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
"""The engine-owned selected-token projection in FSDPEngineWithLMHead.prepare_model_outputs.

Covers: the projection is emitted, per sequence, on both the remove-padding and padded paths and only
when the batch carries ``selected_token_ids``; when it runs, ``log_probs`` comes from the same shared
log-softmax (the fused cross-entropy is bypassed, not run out of place); the sparse backward matches
autograd's reference; fused kernels and out-of-vocabulary ids are rejected; and the reference
calibration objective produces the expected gradients and probabilities end to end in both
normalization modes.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import logprobs_from_logits_v2
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import (
    FSDPEngineWithLMHead,
    _log_probs_and_selected_token_logprobs,
    _LogSoftmaxSparseGather,
)
from verl.workers.utils.auxiliary_objectives import (
    AuxiliaryLossComposer,
    LoadedAuxiliaryObjective,
    SelectedTokenCalibrationObjective,
)

VOCAB = 16
PROMPT_LENS = [3, 2]
RESP_LENS = [4, 3]
SELECTED = [5, 6, 7]  # column order of selected_token_logprobs; 6 and 7 are the "positive" labels


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _micro_batch(use_remove_padding: bool, selected=SELECTED, use_fused_kernels: bool = False):
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
    flags = dict(
        use_remove_padding=use_remove_padding,
        use_fused_kernels=use_fused_kernels,
        calculate_entropy=False,
        dp_size=1,
        batch_num_tokens=sum(RESP_LENS),
        global_batch_size=2,
    )
    if selected:
        flags["selected_token_ids"] = tuple(selected)  # a list would become a per-sample NonTensorStack
    tu.assign_non_tensor(data, **flags)
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
    rolled = torch.roll(ids.values(), shifts=-1, dims=0)
    if use_remove_padding:
        logits = param[:total].unsqueeze(0)  # (1, total_nnz, vocab)
        args = {"input_ids_rmpad_rolled": rolled, "temperature_rmpad": torch.ones(total), "pad_size": 0}
    else:
        max_len = max(p + r for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True))
        logits = param[: 2 * max_len].reshape(2, max_len, VOCAB)
        args = {"input_ids_rmpad_rolled": rolled, "temperature": torch.ones(2)}
    args["temperature_is_one"] = True
    return SimpleNamespace(logits=logits), args


def _composer(objective, name="cal", selected=SELECTED):
    def base(model_output, data, dp_group=None):
        return model_output["log_probs"].values().sum(), {}

    return AuxiliaryLossComposer(
        base_loss_fn=base,
        objectives=[LoadedAuxiliaryObjective(name=name, weight=0.5, objective=objective)],
        loss_agg_mode="token-mean",
        selected_token_ids=tuple(selected),
    )


@pytest.fixture
def record_inplace(monkeypatch):
    """Record whether/how the fused cross-entropy helper was called."""
    seen = {}

    def fake(logits, labels, inplace_backward=True):
        seen["called"] = True
        seen["inplace_backward"] = inplace_backward
        return logprobs_from_logits_v2(logits, labels)

    monkeypatch.setattr(transformer_impl, "logprobs_from_logits", fake)
    return seen


def _reference(logits, labels, ids):
    lsm = torch.log_softmax(logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits, dim=-1)
    return lsm.gather(-1, labels.unsqueeze(-1)).squeeze(-1), lsm.index_select(-1, ids)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_log_softmax_matches_reference_forward_and_backward(dtype):
    torch.manual_seed(0)
    rows = 9
    logits = torch.randn(rows, VOCAB, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, VOCAB, (rows,))
    ids = torch.tensor(SELECTED)

    lp, sel = _log_probs_and_selected_token_logprobs(logits, labels, SELECTED)
    ref_lp, ref_sel = _reference(logits.detach().clone().requires_grad_(True), labels, ids)
    torch.testing.assert_close(lp.float(), ref_lp.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sel, ref_sel.float(), atol=1e-5, rtol=1e-5)

    g_lp, g_sel = torch.randn(rows), torch.randn(rows, len(SELECTED))
    (grad,) = torch.autograd.grad((lp.float() * g_lp).sum() + (sel * g_sel).sum(), logits)
    ref_logits = logits.detach().clone().requires_grad_(True)
    r_lp, r_sel = _reference(ref_logits, labels, ids)
    (ref_grad,) = torch.autograd.grad((r_lp.float() * g_lp).sum() + (r_sel.float() * g_sel).sum(), ref_logits)
    tol = 2e-3 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(grad.float(), ref_grad.float(), atol=tol, rtol=1e-3)


def test_sparse_gather_passes_gradcheck():
    torch.manual_seed(1)
    logits = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
    index = torch.tensor([[0, 2, 3], [1, 2, 3], [6, 2, 3], [4, 2, 3], [2, 2, 3]])  # repeated columns on purpose
    assert torch.autograd.gradcheck(lambda x: _LogSoftmaxSparseGather.apply(x, index), (logits,))


def test_distillation_only_projection_has_no_label_column():
    logits = torch.randn(4, VOCAB, requires_grad=True)
    lp, sel = _log_probs_and_selected_token_logprobs(logits, None, SELECTED)
    assert lp is None and sel.shape == (4, len(SELECTED)) and sel.requires_grad


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_projection_is_nested_per_sequence_and_shares_the_log_softmax(record_inplace, use_remove_padding):
    torch.manual_seed(0)
    data = _micro_batch(use_remove_padding)
    param = torch.randn(64, VOCAB, requires_grad=True)
    raw, args = _raw_output_and_args(data, use_remove_padding, param)

    model_output = _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=None)

    # the fused cross-entropy helper is bypassed entirely when the projection is on
    assert "called" not in record_inplace
    sel = model_output["selected_token_logprobs"]
    assert sel.is_nested
    assert [t.shape for t in sel.unbind()] == [
        (p + r, len(SELECTED)) for p, r in zip(PROMPT_LENS, RESP_LENS, strict=True)
    ]
    assert sel.values().requires_grad and model_output["log_probs"].values().requires_grad
    total = sel.values().shape[0]
    ref_lp, ref_sel = _reference(param[:total], args["input_ids_rmpad_rolled"], torch.tensor(SELECTED))
    torch.testing.assert_close(sel.values(), ref_sel.float())
    torch.testing.assert_close(model_output["log_probs"].values().float(), ref_lp)


def test_no_flag_means_no_projection_and_in_place_backward(record_inplace):
    data = _micro_batch(use_remove_padding=True, selected=None)
    param = torch.randn(64, VOCAB, requires_grad=True)
    raw, args = _raw_output_and_args(data, True, param)
    model_output = _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=None)
    assert "selected_token_logprobs" not in model_output
    assert record_inplace["called"] and record_inplace["inplace_backward"] is True


def test_fused_kernels_reject_projection():
    data = _micro_batch(use_remove_padding=True, use_fused_kernels=True)
    raw, args = _raw_output_and_args(data, True, torch.randn(64, VOCAB))
    with pytest.raises(NotImplementedError, match="use_fused_kernels"):
        _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=None)


def test_out_of_vocabulary_id_is_rejected(record_inplace):
    data = _micro_batch(use_remove_padding=True, selected=[1, VOCAB + 3])
    raw, args = _raw_output_and_args(data, True, torch.randn(64, VOCAB))
    with pytest.raises(ValueError, match="vocabulary"):
        _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=None)


def test_calibration_objective_end_to_end_gradient(record_inplace):
    torch.manual_seed(1)
    data = _micro_batch(use_remove_padding=True)
    param = torch.randn(64, VOCAB, requires_grad=True)
    raw, args = _raw_output_and_args(data, True, param)
    objective = SelectedTokenCalibrationObjective(positive_token_ids=SELECTED[1:])  # token_set vs {5}
    composer = _composer(objective)
    composer.prepare_global_stats(data)
    assert tu.get_non_tensor_data(data, key="aux_global_stats", default=None)["cal"]["cells"] == 5.0

    model_output = _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=None)
    loss, metrics = composer(model_output=model_output, data=data)
    assert metrics["actor/aux/cal/active"].aggregate() == 1.0
    assert metrics["actor/aux/cal/normalizer"].aggregate() == 5.0
    assert 0.0 < metrics["actor/aux/cal/bce"].aggregate() < 20.0
    assert 0.0 < metrics["actor/aux/cal/positive_prob"].aggregate() < 1.0

    base_loss = model_output["log_probs"].values().sum()
    (grad_aux,) = torch.autograd.grad(loss - base_loss, param)
    assert torch.isfinite(grad_aux).all()

    # Response token j of sequence b is predicted at packed row prompt_len_b + j - 1. Sequence 0 has a
    # positive cell at j=0: BCE pulls mass onto the positive labels (negative gradient) and off the
    # negative label (positive gradient). j=1 is not a cell, so its row gets no gradient at all.
    row = PROMPT_LENS[0] - 1
    assert (grad_aux[row, SELECTED[1:]] < 0).all()
    assert grad_aux[row, SELECTED[0]] > 0
    assert grad_aux[row + 1].abs().sum() == 0
    # token_set mode: the full-vocab normalizer cancels in lse_pos - lse_neg, so tokens outside the
    # selected set get only float round-off from this objective, orders of magnitude below the labels
    outside = [t for t in range(VOCAB) if t not in SELECTED]
    assert grad_aux[row, outside].abs().max() < 1e-6 * grad_aux[row, SELECTED].abs().max()


@pytest.mark.parametrize(("normalize_over", "expected_prob"), [("token_set", 2 / 3), ("vocab", 2 / VOCAB)])
def test_calibration_probability_definitions_on_uniform_logits(record_inplace, normalize_over, expected_prob):
    data = _micro_batch(use_remove_padding=True)
    raw, args = _raw_output_and_args(data, True, torch.zeros(64, VOCAB, requires_grad=True))
    objective = SelectedTokenCalibrationObjective(positive_token_ids=SELECTED[1:], normalize_over=normalize_over)
    composer = _composer(objective)
    composer.prepare_global_stats(data)
    model_output = _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=None)
    _, metrics = composer(model_output=model_output, data=data)
    assert math.isclose(metrics["actor/aux/cal/positive_prob"].aggregate(), expected_prob, rel_tol=1e-5)
    # 3 positive cells, 2 negative cells: mean BCE of a constant prediction p
    p = expected_prob
    expected_bce = (3 * -math.log(p) + 2 * -math.log(1 - p)) / 5
    assert math.isclose(metrics["actor/aux/cal/bce"].aggregate(), expected_bce, rel_tol=1e-5)


def test_calibration_objective_validates_ids_against_selected_set():
    with pytest.raises(ValueError, match="unique"):
        SelectedTokenCalibrationObjective(positive_token_ids=[6, 6])
    with pytest.raises(ValueError, match="normalize_over"):
        SelectedTokenCalibrationObjective(positive_token_ids=[6], normalize_over="nope")
    with pytest.raises(ValueError, match="overlap"):
        SelectedTokenCalibrationObjective(positive_token_ids=[6], negative_token_ids=[6])

    data = _micro_batch(use_remove_padding=True)
    raw, args = _raw_output_and_args(data, True, torch.zeros(64, VOCAB, requires_grad=True))
    model_output = _engine_stub().prepare_model_outputs(raw, args, data, logits_processor_func=None)

    not_selected = _composer(SelectedTokenCalibrationObjective(positive_token_ids=[9]))
    not_selected.prepare_global_stats(data)
    with pytest.raises(ValueError, match="not in\n?.*selected_token_logprobs"):
        not_selected(model_output=model_output, data=data)

    no_negatives = _composer(SelectedTokenCalibrationObjective(positive_token_ids=SELECTED))
    no_negatives.prepare_global_stats(data)
    with pytest.raises(ValueError, match="at least one negative"):
        no_negatives(model_output=model_output, data=data)
