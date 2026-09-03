# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The auxiliary-objective logits-processor stage inside FSDPEngineWithLMHead.prepare_model_outputs.

Covers: processor outputs are re-nested per sequence on both the remove-padding and padded paths,
the fused cross-entropy keeps its logits buffer intact (``inplace_backward=False``) whenever a
processor ran, the processor never runs on a no-grad pass, fused kernels are rejected, and the
reference calibration objective produces a gradient through the logits end to end.
"""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import logprobs_from_logits_v2
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.auxiliary_objectives import (
    AuxiliaryLossComposer,
    LoadedAuxiliaryObjective,
    SpecifiedTokenCalibrationObjective,
)

VOCAB = 16
PROMPT_LENS = [3, 2]
RESP_LENS = [4, 3]
DIGITS = [5, 6, 7]  # token ids the calibration objective watches; 6 and 7 are "positive"


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _micro_batch(use_remove_padding: bool, use_fused_kernels: bool = False):
    seqs = [torch.randint(0, VOCAB, (p + r,)) for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True)]
    data = TensorDict(
        {
            "input_ids": _nested(seqs),
            "position_ids": _nested([torch.arange(len(s)) for s in seqs]),
            "prompts": _nested([s[:p] for s, p in zip(seqs, PROMPT_LENS, strict=True)]),
            "responses": _nested([s[p:] for s, p in zip(seqs, PROMPT_LENS, strict=True)]),
            "response_mask": _nested([torch.ones(r) for r in RESP_LENS]),
            # per response position: 1 = positive, 0 = negative, -1 = not a calibrated cell
            "calibration_target": torch.tensor([[1, -1, 0, 1], [0, 1, -1, -1]]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        use_remove_padding=use_remove_padding,
        use_fused_kernels=use_fused_kernels,
        calculate_entropy=False,
        dp_size=1,
        batch_num_tokens=sum(RESP_LENS),
        global_batch_size=2,
    )
    return data


def _engine_stub():
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.use_ulysses_sp = False
    eng.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    return eng


def _raw_output_and_args(data, use_remove_padding, param):
    """Build the model output the engine expects from a parameter so gradients can be checked."""
    total = sum(p + r for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True))
    ids = data["input_ids"]
    if use_remove_padding:
        rolled = torch.roll(ids.values(), shifts=-1, dims=0)
        logits = param[:total].unsqueeze(0)  # (1, total_nnz, vocab)
        args = {
            "input_ids_rmpad_rolled": rolled,
            "temperature_rmpad": torch.ones(total),
            "pad_size": 0,
            "temperature_is_one": True,
        }
    else:
        max_len = max(p + r for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True))
        logits = param[: 2 * max_len].reshape(2, max_len, VOCAB)
        args = {
            "input_ids_rmpad_rolled": torch.roll(ids.values(), shifts=-1, dims=0),
            "temperature": torch.ones(2),
            "temperature_is_one": True,
        }
    return SimpleNamespace(logits=logits), args


def _composer(objective, name="cal"):
    def base(model_output, data, dp_group=None):
        return model_output["log_probs"].values().sum(), {}

    return AuxiliaryLossComposer(
        base_loss_fn=base,
        objectives=[LoadedAuxiliaryObjective(name=name, weight=0.5, objective=objective)],
        loss_agg_mode="token-mean",
    )


@pytest.fixture
def record_inplace(monkeypatch):
    seen = {}

    def fake(logits, labels, inplace_backward=True):
        seen["inplace_backward"] = inplace_backward
        return logprobs_from_logits_v2(logits, labels)

    monkeypatch.setattr(transformer_impl, "logprobs_from_logits", fake)
    return seen


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_processor_outputs_are_nested_and_ce_backward_is_out_of_place(record_inplace, use_remove_padding):
    torch.manual_seed(0)
    data = _micro_batch(use_remove_padding)
    param = torch.randn(64, VOCAB, requires_grad=True)
    raw, args = _raw_output_and_args(data, use_remove_padding, param)
    composer = _composer(SpecifiedTokenCalibrationObjective(token_ids=DIGITS, positive_token_ids=DIGITS[1:]))
    composer.prepare_global_stats(data)

    model_output = _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=composer)

    assert record_inplace["inplace_backward"] is False
    aux = model_output["aux/cal/logprobs"]
    assert aux.is_nested
    assert [t.shape for t in aux.unbind()] == [
        (p + r, len(DIGITS)) for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True)
    ]
    assert aux.values().requires_grad
    # rows are full-vocab log-softmax columns: their mass never exceeds one
    assert (aux.values().exp().sum(-1) <= 1 + 1e-5).all()


def test_plain_loss_function_keeps_in_place_backward(record_inplace):
    data = _micro_batch(use_remove_padding=True)
    param = torch.randn(64, VOCAB, requires_grad=True)
    raw, args = _raw_output_and_args(data, True, param)

    def plain_loss(model_output, data, dp_group=None):
        return model_output["log_probs"].values().sum(), {}

    model_output = _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=plain_loss)
    assert record_inplace["inplace_backward"] is True
    assert not any(k.startswith("aux/") for k in model_output)


def test_processor_does_not_run_without_grad(record_inplace):
    data = _micro_batch(use_remove_padding=True)
    param = torch.randn(64, VOCAB, requires_grad=True)
    raw, args = _raw_output_and_args(data, True, param)
    composer = _composer(SpecifiedTokenCalibrationObjective(token_ids=DIGITS, positive_token_ids=DIGITS[1:]))
    composer.prepare_global_stats(data)
    with torch.no_grad():
        model_output = _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=composer)
    assert not any(k.startswith("aux/") for k in model_output)
    assert record_inplace["inplace_backward"] is True


def test_fused_kernels_reject_processor():
    data = _micro_batch(use_remove_padding=True, use_fused_kernels=True)
    param = torch.randn(64, VOCAB, requires_grad=True)
    raw, args = _raw_output_and_args(data, True, param)
    composer = _composer(SpecifiedTokenCalibrationObjective(token_ids=DIGITS, positive_token_ids=DIGITS[1:]))
    with pytest.raises(NotImplementedError, match="use_fused_kernels"):
        _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=composer)


def test_calibration_objective_end_to_end_gradient(record_inplace):
    torch.manual_seed(1)
    data = _micro_batch(use_remove_padding=True)
    param = torch.randn(64, VOCAB, requires_grad=True)
    raw, args = _raw_output_and_args(data, True, param)
    objective = SpecifiedTokenCalibrationObjective(token_ids=DIGITS, positive_token_ids=DIGITS[1:])
    composer = _composer(objective)
    composer.prepare_global_stats(data)
    assert tu.get_non_tensor_data(data, key="aux_global_stats", default=None)["cal"]["cells"] == 5.0

    eng = _engine_stub()
    model_output = eng.prepare_model_outputs(raw, args, data, logits_processor_func=composer)
    loss, metrics = composer(model_output=model_output, data=data)
    assert metrics["actor/aux/cal/active"].aggregate() == 1.0
    assert metrics["actor/aux/cal/normalizer"].aggregate() == 5.0
    assert 0.0 < metrics["actor/aux/cal/bce"].aggregate() < 20.0

    base_loss = model_output["log_probs"].values().sum()
    (grad_aux,) = torch.autograd.grad(loss - base_loss, param)
    assert torch.isfinite(grad_aux).all()
    assert grad_aux.abs().sum() > 0

    # Response token j of sequence b is predicted at packed row prompt_len_b + j - 1. Sequence 0 has a
    # positive cell at j=0: BCE pulls mass onto the positive digits (negative gradient) and off the
    # remaining watched digit (positive gradient). j=1 is not a cell, so its row gets no gradient.
    pos_row = PROMPT_LENS[0] - 1
    assert (grad_aux[pos_row, DIGITS[1:]] < 0).all()
    assert grad_aux[pos_row, DIGITS[0]] > 0
    assert grad_aux[pos_row + 1].abs().sum() == 0
