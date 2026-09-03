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
"""Unit tests for the actor-side auxiliary objective composer (no model; distributed calls stubbed)."""

import textwrap
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import AuxiliaryObjectiveConfig
from verl.workers.utils import auxiliary_objectives as aux
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
    stat_names = ("n",)

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
    stat_names = ("seqs",)

    def prepare_batch(self, data):
        return {"seqs": float(data.batch_size[0])}

    def compute(self, *, model_output, data, context):
        return ActorObjectiveResult(loss_sum=model_output["log_probs"].exp().sum(), normalizer="seqs")


def _composer(*specs):
    return AuxiliaryLossComposer(base_loss_fn=_base_loss, objectives=list(specs), loss_agg_mode="token-mean")


def _spec(name, obj, weight=1.0, metrics_only=False):
    return LoadedAuxiliaryObjective(name=name, weight=weight, objective=obj, metrics_only=metrics_only)


@pytest.fixture
def fake_dist(monkeypatch):
    """Pretend a single-rank process group exists: all_reduce/all_gather become local no-ops."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, *a, **k: None)

    def all_gather_object(out, obj, *a, **k):
        out[0] = obj

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(aux, "get_device_id", lambda: "cpu")


def _objective_file(tmp_path, body, name="objectives.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return str(path)


SQUARE_SRC = """
    from verl.workers.utils.auxiliary_objectives import ActorObjectiveResult, BaseActorAuxiliaryObjective

    class Obj(BaseActorAuxiliaryObjective):
        required_batch_keys = ("mask",)
        required_model_output_keys = ("log_probs",)
        stat_names = ("n",)
        def __init__(self, scale=1.0):
            self.scale = scale
        def prepare_batch(self, data):
            return {"n": data["mask"].sum()}
        def compute(self, *, model_output, data, context):
            loss_sum = (model_output["log_probs"] ** 2 * data["mask"]).sum() * self.scale
            return ActorObjectiveResult(loss_sum=loss_sum, normalizer="n")

    def build_objective(scale=1.0):
        return Obj(scale)

    def build_needs_entropy():
        o = Obj()
        o.required_model_output_keys = ("log_probs", "entropy")
        return o

    def build_broken():
        return object()
    """


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


def test_stats_are_packed_in_static_schema_order_and_reduced_once(fake_dist, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, *a, **k: calls.append(tensor.clone()))
    data = _data()
    composer = _composer(_spec("sq", SquareObjective()), _spec("ex", ExpObjective()))
    composer.prepare_global_stats(data)
    assert len(calls) == 1
    assert calls[0].tolist() == [B * T, float(B)]  # (sq, n) then (ex, seqs): configuration order
    stats = tu.get_non_tensor_data(data, key=AUX_GLOBAL_STATS_KEY, default=None)
    assert stats == {"sq": {"n": float(B * T)}, "ex": {"seqs": float(B)}}


def test_prepare_batch_must_return_exactly_the_declared_stats():
    class Drifting(SquareObjective):
        def prepare_batch(self, data):
            return {"n": 1.0, "surprise": 2.0}

    composer = _composer(_spec("d", Drifting()))
    with pytest.raises(ValueError, match="schema must be static"):
        composer.prepare_global_stats(_data())


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


def test_metrics_only_objective_reports_but_never_contributes():
    data = _data()
    composer = _composer(_spec("sq", SquareObjective(), weight=0.0, metrics_only=True))
    composer.prepare_global_stats(data)
    out = _model_output()
    loss, metrics = composer(model_output=out, data=data)
    torch.testing.assert_close(loss, _base_loss(out, data)[0])
    assert metrics["actor/aux/sq/loss"].aggregate() > 0
    assert metrics["actor/aux/sq/weighted_loss"].aggregate() == 0.0
    assert metrics["actor/aux/sq/active"].aggregate() == 1.0


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
    stat_names = ("n",)

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
        ("normalizer", KeyError, "not one of its declared stat_names"),
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


def test_loader_builds_objectives_and_skips_zero_weight(tmp_path):
    path = _objective_file(tmp_path, SQUARE_SRC)
    cfgs = [
        AuxiliaryObjectiveConfig(name="a", path=path, weight=0.5, kwargs={"scale": 2.0}),
        AuxiliaryObjectiveConfig(name="b", path=path),
        AuxiliaryObjectiveConfig(name="zero", path=path, weight=0.0),
        AuxiliaryObjectiveConfig(name="probe", path=path, weight=0.0, metrics_only=True),
    ]
    loaded = load_auxiliary_objectives(cfgs)
    assert [o.name for o in loaded] == ["a", "b", "probe"]
    assert loaded[0].objective.scale == 2.0 and loaded[0].weight == 0.5
    assert loaded[2].metrics_only

    with pytest.raises(ValueError, match="unique"):
        load_auxiliary_objectives([cfgs[0], AuxiliaryObjectiveConfig(name="a", path=path)])
    with pytest.raises(TypeError, match="missing attribute"):
        load_auxiliary_objectives([AuxiliaryObjectiveConfig(name="c", path=path, factory="build_broken")])


def test_digest_covers_kwargs_file_and_selected_ids(tmp_path):
    path = _objective_file(tmp_path, SQUARE_SRC)
    base = [AuxiliaryObjectiveConfig(name="a", path=path, kwargs={"scale": 1.0})]
    changed_kwargs = [AuxiliaryObjectiveConfig(name="a", path=path, kwargs={"scale": 2.0})]
    d0 = AuxiliaryLossComposer(_base_loss, load_auxiliary_objectives(base), "m").config_digest()
    assert d0 == AuxiliaryLossComposer(_base_loss, load_auxiliary_objectives(base), "m").config_digest()
    assert d0 != AuxiliaryLossComposer(_base_loss, load_auxiliary_objectives(changed_kwargs), "m").config_digest()
    assert d0 != AuxiliaryLossComposer(_base_loss, load_auxiliary_objectives(base), "m", (1, 2)).config_digest()
    other_file = _objective_file(tmp_path, SQUARE_SRC + "\n    # different bytes\n", name="objectives2.py")
    assert (
        d0
        != AuxiliaryLossComposer(
            _base_loss, load_auxiliary_objectives([AuxiliaryObjectiveConfig(name="a", path=other_file)]), "m"
        ).config_digest()
    )


def test_maybe_wrap_requires_process_group(tmp_path, monkeypatch):
    path = _objective_file(tmp_path, SQUARE_SRC)
    cfg = SimpleNamespace(
        strategy="fsdp",
        auxiliary_objectives=[AuxiliaryObjectiveConfig(name="a", path=path)],
        loss_agg_mode="token-mean",
        selected_token_logprobs=SimpleNamespace(token_ids=[]),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="process group"):
        maybe_wrap_with_auxiliary_objectives(_base_loss, cfg)


def test_maybe_wrap_validates_backend_requirements_and_ranks(tmp_path, fake_dist, monkeypatch):
    path = _objective_file(tmp_path, SQUARE_SRC)
    entry = AuxiliaryObjectiveConfig(name="a", path=path)

    def cfg(**over):
        base = dict(
            strategy="fsdp",
            auxiliary_objectives=[entry],
            loss_agg_mode="token-mean",
            selected_token_logprobs=SimpleNamespace(token_ids=[3, 4]),
        )
        base.update(over)
        return SimpleNamespace(**base)

    with pytest.raises(NotImplementedError, match="only supported with strategy"):
        maybe_wrap_with_auxiliary_objectives(_base_loss, cfg(strategy="megatron"))

    needs_entropy = AuxiliaryObjectiveConfig(name="e", path=path, factory="build_needs_entropy")
    with pytest.raises(ValueError, match="calculate_entropy"):
        maybe_wrap_with_auxiliary_objectives(
            _base_loss, cfg(auxiliary_objectives=[needs_entropy]), available_model_outputs={"log_probs"}
        )

    composer = maybe_wrap_with_auxiliary_objectives(_base_loss, cfg(), available_model_outputs={"log_probs"})
    assert isinstance(composer, AuxiliaryLossComposer)
    assert composer.selected_token_ids == (3, 4)

    def disagreeing_gather(out, obj, *a, **k):
        out[0] = obj
        out[1] = "someone-else"

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", disagreeing_gather)
    with pytest.raises(RuntimeError, match="differently across ranks"):
        maybe_wrap_with_auxiliary_objectives(_base_loss, cfg())
