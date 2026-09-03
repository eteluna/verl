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
"""Unit tests for the actor-side auxiliary objective composer (no distributed, no model)."""

import textwrap
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import AuxiliaryObjectiveConfig
from verl.workers.utils.auxiliary_objectives import (
    AUX_GLOBAL_STATS_KEY,
    ActorObjectiveResult,
    AuxiliaryLossComposer,
    BaseActorAuxiliaryObjective,
    LoadedAuxiliaryObjective,
    load_auxiliary_objectives,
    maybe_wrap_with_auxiliary_objectives,
)

B, T = 4, 5


def _data(mask=None, dp_size=1):
    data = TensorDict({"mask": torch.ones(B, T) if mask is None else mask}, batch_size=[B])
    tu.assign_non_tensor(data, dp_size=dp_size, batch_num_tokens=B * T, global_batch_size=B)
    return data


def _model_output(seed=0):
    torch.manual_seed(seed)
    return {"log_probs": torch.randn(B, T, requires_grad=True)}


def _base_loss(model_output, data, dp_group=None):
    return model_output["log_probs"].mean(), {"actor/pg_loss": Metric(AggregationType.MEAN, 1.0)}


class SquareObjective(BaseActorAuxiliaryObjective):
    """loss_sum = sum(mask * log_probs^2), normalized by the number of masked elements."""

    required_batch_keys = ("mask",)
    required_model_output_keys = ("log_probs",)

    def prepare_batch(self, data):
        return {"n": data["mask"].sum()}

    def compute(self, *, model_output, data, context):
        m = data["mask"]
        return ActorObjectiveResult(
            loss_sum=(model_output["log_probs"] ** 2 * m).sum(), normalizer="n", metrics={"extra": 2.0}
        )


class ExpObjective(BaseActorAuxiliaryObjective):
    """loss_sum = sum(exp(log_probs)), normalized by sequence count."""

    required_model_output_keys = ("log_probs",)

    def prepare_batch(self, data):
        return {"seqs": float(data.batch_size[0])}

    def compute(self, *, model_output, data, context):
        return ActorObjectiveResult(loss_sum=model_output["log_probs"].exp().sum(), normalizer="seqs")


def _composer(*specs):
    return AuxiliaryLossComposer(base_loss_fn=_base_loss, objectives=list(specs), loss_agg_mode="token-mean")


def _spec(name, obj, weight=1.0):
    return LoadedAuxiliaryObjective(name=name, weight=weight, objective=obj)


def test_empty_config_returns_base_loss_unchanged():
    cfg = SimpleNamespace(strategy="fsdp", auxiliary_objectives=[], loss_agg_mode="token-mean")
    assert maybe_wrap_with_auxiliary_objectives(_base_loss, cfg) is _base_loss


def test_gradient_superposition_matches_manual_sum():
    dp = 3
    data = _data(dp_size=dp)
    composer = _composer(_spec("sq", SquareObjective(), 0.3), _spec("ex", ExpObjective(), 0.7))
    composer.prepare_global_stats(data)

    out = _model_output()
    loss, metrics = composer(model_output=out, data=data)
    (grad,) = torch.autograd.grad(loss, out["log_probs"])

    ref = _model_output()
    n = data["mask"].sum()
    manual = (
        ref["log_probs"].mean()
        + 0.3 * dp * (ref["log_probs"] ** 2).sum() / n
        + 0.7 * dp * ref["log_probs"].exp().sum() / float(B)
    )
    (ref_grad,) = torch.autograd.grad(manual, ref["log_probs"])

    torch.testing.assert_close(loss.detach(), manual.detach())
    torch.testing.assert_close(grad, ref_grad)
    assert metrics["actor/aux/sq/active"].aggregate() == 1.0
    assert metrics["actor/aux/sq/normalizer"].aggregate() == n.item()
    assert metrics["actor/aux/sq/extra"].aggregate() == 2.0
    assert metrics["actor/aux/sq/loss"].aggregation is AggregationType.SUM
    assert "actor/pg_loss" in metrics


def test_normalization_is_invariant_to_micro_batch_split():
    data = _data()
    composer = _composer(_spec("sq", SquareObjective(), 0.5))
    composer.prepare_global_stats(data)
    out = _model_output()

    whole, _ = composer(model_output=out, data=data)
    whole_aux = whole - _base_loss(out, data)[0]

    halves = tu.chunk_tensordict(data, 2)
    split_aux = torch.zeros(())
    for i, half in enumerate(halves):
        assert tu.get_non_tensor_data(half, key=AUX_GLOBAL_STATS_KEY, default=None) is not None
        half_out = {"log_probs": out["log_probs"][i * 2 : (i + 1) * 2]}
        part, _ = composer(model_output=half_out, data=half)
        split_aux = split_aux + (part - _base_loss(half_out, half)[0])
    torch.testing.assert_close(whole_aux, split_aux)


def test_zero_normalizer_is_inactive_and_finite():
    data = _data(mask=torch.zeros(B, T))
    composer = _composer(_spec("sq", SquareObjective(), 1.0))
    composer.prepare_global_stats(data)
    out = _model_output()
    loss, metrics = composer(model_output=out, data=data)
    torch.testing.assert_close(loss, _base_loss(out, data)[0])
    assert metrics["actor/aux/sq/active"].aggregate() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.isfinite(out["log_probs"].grad).all()


def test_no_grad_pass_skips_objectives():
    data = _data()
    composer = _composer(_spec("sq", SquareObjective()))
    composer.prepare_global_stats(data)
    with torch.no_grad():
        loss, metrics = composer(model_output=_model_output(), data=data)
    assert not any(k.startswith("actor/aux/") for k in metrics)
    assert not loss.requires_grad


def test_extra_kwargs_are_delegated_to_base_loss():
    seen = {}

    def base(model_output=None, data=None, dp_group=None, **kw):
        seen.update(kw)
        return {"delegated": True}

    composer = AuxiliaryLossComposer(base_loss_fn=base, objectives=[_spec("sq", SquareObjective())], loss_agg_mode="x")
    assert composer(model_output=None, data=None, student_logits="L") == {"delegated": True}
    assert seen == {"student_logits": "L"}


def test_missing_required_batch_key_fails_in_prepare():
    data = TensorDict({}, batch_size=[B])
    composer = _composer(_spec("sq", SquareObjective()))
    with pytest.raises(KeyError, match="requires batch keys"):
        composer.prepare_global_stats(data)


def test_missing_prepare_stage_is_reported():
    composer = _composer(_spec("sq", SquareObjective()))
    with pytest.raises(RuntimeError, match="prepare_global_stats must run"):
        composer(model_output=_model_output(), data=_data())


class _Broken(BaseActorAuxiliaryObjective):
    def __init__(self, kind):
        self.kind = kind

    def prepare_batch(self, data):
        return {"n": 1.0}

    def compute(self, *, model_output, data, context):
        lp = model_output["log_probs"]
        if self.kind == "detached":
            return ActorObjectiveResult(loss_sum=lp.detach().sum(), normalizer="n")
        if self.kind == "vector":
            return ActorObjectiveResult(loss_sum=lp.sum(-1), normalizer="n")
        if self.kind == "nan":
            return ActorObjectiveResult(loss_sum=lp.sum() * float("nan"), normalizer="n")
        if self.kind == "normalizer":
            return ActorObjectiveResult(loss_sum=lp.sum(), normalizer="missing")
        if self.kind == "collision":
            return ActorObjectiveResult(loss_sum=lp.sum(), normalizer="n", metrics={"loss": 1.0})
        raise AssertionError(self.kind)


@pytest.mark.parametrize(
    ("kind", "exc", "match"),
    [
        ("detached", ValueError, "detached loss"),
        ("vector", ValueError, "scalar tensor"),
        ("nan", ValueError, "not finite"),
        ("normalizer", KeyError, "was not returned by prepare_batch"),
        ("collision", KeyError, "metric collision"),
    ],
)
def test_invalid_results_fail_with_objective_name(kind, exc, match):
    data = _data()
    composer = _composer(_spec("bad", _Broken(kind)))
    composer.prepare_global_stats(data)
    with pytest.raises(exc, match=match) as info:
        composer(model_output=_model_output(), data=data)
    assert "bad" in str(info.value)


class _Processor(BaseActorAuxiliaryObjective):
    uses_logits_processor = True

    def __init__(self, width=3, wrong_leading=False):
        self.width, self.wrong_leading = width, wrong_leading

    def prepare_batch(self, data):
        return {"n": 1.0}

    def process_logits(self, *, logits, data, context):
        rows = logits.shape[0] + (1 if self.wrong_leading else 0)
        return {"cols": logits.new_zeros(rows, self.width)}

    def compute(self, *, model_output, data, context):
        return ActorObjectiveResult(loss_sum=model_output["aux/p/cols"].sum() * 0.0, normalizer="n")


def test_process_logits_namespaces_outputs_and_validates_shape():
    data = _data()
    logits = torch.zeros(7, 32)
    composer = _composer(_spec("p", _Processor()))
    composer.prepare_global_stats(data)
    out = composer.process_logits(logits=logits, data=data)
    assert list(out) == ["aux/p/cols"]
    assert out["aux/p/cols"].shape == (7, 3)

    composer = _composer(_spec("p", _Processor(wrong_leading=True)))
    composer.prepare_global_stats(data)
    with pytest.raises(ValueError, match="leading dim"):
        composer.process_logits(logits=logits, data=data)

    composer = _composer(_spec("p", _Processor(width=32)))
    composer.prepare_global_stats(data)
    with pytest.raises(ValueError, match="not compact"):
        composer.process_logits(logits=logits, data=data)


def test_loader_builds_objectives_from_file_and_rejects_bad_ones(tmp_path):
    src = textwrap.dedent(
        """
        from verl.workers.utils.auxiliary_objectives import ActorObjectiveResult, BaseActorAuxiliaryObjective

        class Obj(BaseActorAuxiliaryObjective):
            def __init__(self, scale):
                self.scale = scale
            def prepare_batch(self, data):
                return {"n": 1.0}
            def compute(self, *, model_output, data, context):
                return ActorObjectiveResult(loss_sum=model_output["log_probs"].sum() * self.scale, normalizer="n")

        def build_objective(scale=1.0):
            return Obj(scale)

        def build_broken():
            return object()
        """
    )
    path = tmp_path / "objectives.py"
    path.write_text(src)

    cfgs = [
        AuxiliaryObjectiveConfig(name="a", path=str(path), weight=0.5, kwargs={"scale": 2.0}),
        AuxiliaryObjectiveConfig(name="b", path=str(path)),
    ]
    loaded = load_auxiliary_objectives(cfgs)
    assert [o.name for o in loaded] == ["a", "b"]
    assert loaded[0].objective.scale == 2.0 and loaded[0].weight == 0.5

    with pytest.raises(ValueError, match="unique"):
        load_auxiliary_objectives([cfgs[0], AuxiliaryObjectiveConfig(name="a", path=str(path))])
    with pytest.raises(TypeError, match="missing attribute"):
        load_auxiliary_objectives([AuxiliaryObjectiveConfig(name="c", path=str(path), factory="build_broken")])

    composer = AuxiliaryLossComposer(_base_loss, loaded, "token-mean")
    assert composer.config_digest() == AuxiliaryLossComposer(_base_loss, loaded, "token-mean").config_digest()


def test_maybe_wrap_rejects_unsupported_backend_and_fused_processor(tmp_path):
    src = textwrap.dedent(
        """
        from verl.workers.utils.auxiliary_objectives import ActorObjectiveResult, BaseActorAuxiliaryObjective

        class P(BaseActorAuxiliaryObjective):
            uses_logits_processor = True
            def process_logits(self, *, logits, data, context):
                return {}
            def compute(self, *, model_output, data, context):
                raise NotImplementedError

        def build_objective():
            return P()
        """
    )
    path = tmp_path / "p.py"
    path.write_text(src)
    entry = AuxiliaryObjectiveConfig(name="p", path=str(path))

    megatron = SimpleNamespace(strategy="megatron", auxiliary_objectives=[entry], loss_agg_mode="token-mean")
    with pytest.raises(NotImplementedError, match="only supported with strategy"):
        maybe_wrap_with_auxiliary_objectives(_base_loss, megatron)

    fused = SimpleNamespace(
        strategy="fsdp", auxiliary_objectives=[entry], loss_agg_mode="token-mean", use_fused_kernels=True, engine=None
    )
    with pytest.raises(NotImplementedError, match="use_fused_kernels"):
        maybe_wrap_with_auxiliary_objectives(_base_loss, fused)

    ok = SimpleNamespace(
        strategy="fsdp2", auxiliary_objectives=[entry], loss_agg_mode="token-mean", use_fused_kernels=False, engine=None
    )
    composer = maybe_wrap_with_auxiliary_objectives(_base_loss, ok)
    assert isinstance(composer, AuxiliaryLossComposer) and composer.has_logits_processors
