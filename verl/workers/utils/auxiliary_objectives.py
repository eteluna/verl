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
"""Extensible actor-side auxiliary objectives.

An auxiliary objective is an additive, differentiable term composed with the selected
PPO/GRPO actor loss::

    L_total = L_base + sum_i weight_i * L_i

Each objective declares the batch fields and model outputs it reads, the statistics it normalizes
by, and returns an unweighted loss *sum* per micro-batch. verl owns loading, execution order, global
(mini-batch) normalization, coefficient application, metric namespacing and validation. An empty
objective list leaves the actor loss untouched.

Objectives only ever read ``model_output``. Anything that has to be derived from the full logits is
produced by the engine from configuration, not by plugin callbacks: ``actor.selected_token_logprobs``
makes every actor-update forward emit ``model_output["selected_token_logprobs"]``, the
full-vocabulary log-softmax at a fixed list of token ids, laid out like ``log_probs``.

Lifecycle for one actor mini-batch::

    prepare_global_stats(data)          # once, on the unsplit mini-batch, one SUM over DP
        -> for each micro-batch:
             forward                    # engine emits log_probs [, entropy, selected_token_logprobs]
             -> base loss (pg, entropy, KL)
             -> compute(model_output, data) for every objective, in config order
             backward on the combined scalar

Normalization: for objective ``i`` and micro-batch ``m`` the contribution is
``dp_size * loss_sum_i^(m) / N_i`` where ``N_i`` is the SUM-reduced statistic named by
``ActorObjectiveResult.normalizer``. Summed over micro-batches and averaged over data-parallel ranks
by the gradient reduction this yields the global mini-batch mean, invariant to how the mini-batch was
split, mirroring ``batch_num_tokens`` in :func:`verl.workers.utils.losses.ppo_loss`. This requires the
objective to be additively decomposable over samples; cross-sample terms are out of scope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.utils.import_utils import load_extern_object
from verl.utils.metric import AggregationType, Metric

__all__ = [
    "AUX_API_VERSION",
    "AUX_METRIC_PREFIX",
    "AUX_GLOBAL_STATS_KEY",
    "SELECTED_TOKEN_LOGPROBS_KEY",
    "ActorObjectiveContext",
    "ActorObjectiveResult",
    "ActorAuxiliaryObjective",
    "BaseActorAuxiliaryObjective",
    "LoadedAuxiliaryObjective",
    "AuxiliaryLossComposer",
    "load_auxiliary_objectives",
    "maybe_wrap_with_auxiliary_objectives",
    "SelectedTokenCalibrationObjective",
]

logger = logging.getLogger(__file__)

# Bumped when the objective protocol or the composer's normalization contract changes.
AUX_API_VERSION = 1
# Metrics land under ``actor/aux/<name>/...``.
AUX_METRIC_PREFIX = "actor/aux"
# Non-tensor key on the mini-batch TensorDict carrying ``{objective_name: {stat_name: float}}``.
AUX_GLOBAL_STATS_KEY = "aux_global_stats"
# model_output key of the engine-owned selected-token projection.
SELECTED_TOKEN_LOGPROBS_KEY = "selected_token_logprobs"

_SUPPORTED_STRATEGIES = ("fsdp", "fsdp2")


@dataclass(frozen=True)
class ActorObjectiveContext:
    """Read-only, framework-owned context handed to every objective callback.

    Args:
        name: The objective's configured name.
        global_stats: This objective's ``prepare_batch`` statistics after the SUM reduction over the
            data-parallel group, i.e. totals over the whole actor mini-batch.
        dp_size: Data-parallel world size of the actor.
        loss_agg_mode: The actor's ``loss_agg_mode`` (informational; the composer normalizes by
            ``global_stats[result.normalizer]``).
        batch_num_tokens: Number of loss-mask tokens in the global mini-batch, or None.
        global_batch_size: Number of sequences in the global mini-batch, or None.
        selected_token_ids: ``actor.selected_token_logprobs.token_ids`` in column order of
            ``model_output["selected_token_logprobs"]``; empty when the projection is off.
    """

    name: str
    global_stats: Mapping[str, float]
    dp_size: int
    loss_agg_mode: str
    batch_num_tokens: Optional[int] = None
    global_batch_size: Optional[int] = None
    selected_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ActorObjectiveResult:
    """What ``compute`` returns for one micro-batch.

    Args:
        loss_sum: Unweighted, un-normalized *sum* of the per-element losses of this micro-batch. A
            scalar tensor connected to the actor's autograd graph. When the objective has no
            applicable element in this micro-batch, return a differentiable zero (for example
            ``some_output.sum() * 0.0``).
        normalizer: Name of the declared statistic that counts the applicable elements over the whole
            mini-batch. The composer divides by its SUM-reduced value.
        metrics: Optional plugin metrics. Plain numbers become MEAN metrics; pass a
            :class:`verl.utils.metric.Metric` to choose another aggregation. Keys are namespaced
            under ``actor/aux/<name>/`` by the composer.
    """

    loss_sum: torch.Tensor
    normalizer: str
    metrics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ActorAuxiliaryObjective(Protocol):
    """The objective protocol. Subclass :class:`BaseActorAuxiliaryObjective` for sane defaults.

    Attributes are static declarations verl validates at start-up: ``required_batch_keys`` must be
    present on the actor mini-batch, ``required_model_output_keys`` must be produced by the configured
    forward, and ``stat_names`` is the exact, ordered set of keys ``prepare_batch`` returns on every
    rank (the reduction packs them in that order, so the schema must not depend on data).

    All callbacks must treat their inputs as read-only and must not run distributed collectives,
    call ``backward`` or touch the optimizer; the composer owns those.
    """

    required_batch_keys: tuple[str, ...]
    required_model_output_keys: tuple[str, ...]
    stat_names: tuple[str, ...]

    def prepare_batch(self, data: TensorDict) -> Mapping[str, float]:
        """Return local additive statistics of the *unsplit* mini-batch, one per ``stat_names`` entry."""
        ...

    def compute(
        self, *, model_output: Mapping[str, torch.Tensor], data: TensorDict, context: ActorObjectiveContext
    ) -> ActorObjectiveResult:
        """Compute this micro-batch's contribution."""
        ...


class BaseActorAuxiliaryObjective:
    """Convenience base class implementing the protocol with no-op defaults."""

    required_batch_keys: tuple[str, ...] = ()
    required_model_output_keys: tuple[str, ...] = ()
    stat_names: tuple[str, ...] = ()

    def prepare_batch(self, data: TensorDict) -> Mapping[str, float]:
        """Default: no statistics. Objectives that normalize must declare ``stat_names`` and override."""
        return {}

    def compute(
        self, *, model_output: Mapping[str, torch.Tensor], data: TensorDict, context: ActorObjectiveContext
    ) -> ActorObjectiveResult:
        """Objectives must implement ``compute``."""
        raise NotImplementedError


@dataclass(frozen=True)
class LoadedAuxiliaryObjective:
    """A configured objective instance together with its name, coefficient and mode."""

    name: str
    weight: float
    objective: Any
    metrics_only: bool = False
    digest_payload: Mapping[str, Any] = field(default_factory=dict)


def _validate_objective_instance(name: str, obj: Any) -> None:
    for attr in ("required_batch_keys", "required_model_output_keys", "stat_names"):
        if not hasattr(obj, attr):
            raise TypeError(f"auxiliary objective '{name}' is missing attribute '{attr}'")
        keys = getattr(obj, attr)
        if not isinstance(keys, tuple | list) or not all(isinstance(k, str) for k in keys):
            raise TypeError(f"auxiliary objective '{name}'.{attr} must be a tuple of str, got {keys!r}")
    if len(set(obj.stat_names)) != len(obj.stat_names):
        raise ValueError(f"auxiliary objective '{name}'.stat_names has duplicates: {obj.stat_names}")
    for method in ("prepare_batch", "compute"):
        if not callable(getattr(obj, method, None)):
            raise TypeError(f"auxiliary objective '{name}' is missing method '{method}'")


def _file_sha256(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unreadable"


def load_auxiliary_objectives(configs) -> list[LoadedAuxiliaryObjective]:
    """Instantiate objectives from ``actor.auxiliary_objectives`` config entries.

    Each entry names a python file (``path``), a factory in it (``factory``), a ``weight``, ``kwargs``
    forwarded to the factory and ``metrics_only``. An entry with ``weight == 0`` and
    ``metrics_only=False`` is skipped entirely: a zero coefficient would still retain the graph and
    run the reduction. Names must be unique and weights finite.
    """
    loaded: list[LoadedAuxiliaryObjective] = []
    seen: set[str] = set()
    for cfg in configs or []:
        name = str(cfg.name)
        if not name or name in seen:
            raise ValueError(f"auxiliary objective names must be non-empty and unique, got {name!r}")
        seen.add(name)
        weight = float(cfg.weight)
        if isinstance(cfg.weight, bool) or not math.isfinite(weight):
            raise ValueError(f"auxiliary objective '{name}' weight must be a finite number, got {cfg.weight!r}")
        metrics_only = bool(getattr(cfg, "metrics_only", False))
        if weight == 0.0 and not metrics_only:
            logger.warning(
                "auxiliary objective '%s' has weight 0 and metrics_only=false; skipping it entirely "
                "(set metrics_only=true to keep its metrics)",
                name,
            )
            continue
        factory = load_extern_object(cfg.path, cfg.factory)
        if not callable(factory):
            raise TypeError(f"auxiliary objective '{name}': {cfg.factory} in {cfg.path} is not callable")
        kwargs = dict(cfg.kwargs) if cfg.kwargs else {}
        obj = factory(**kwargs)
        _validate_objective_instance(name, obj)
        loaded.append(
            LoadedAuxiliaryObjective(
                name=name,
                weight=weight,
                objective=obj,
                metrics_only=metrics_only,
                digest_payload={
                    "api_version": AUX_API_VERSION,
                    "name": name,
                    "weight": weight,
                    "metrics_only": metrics_only,
                    "path": os.path.abspath(cfg.path),
                    "file_sha256": _file_sha256(cfg.path),
                    "factory": cfg.factory,
                    "kwargs": kwargs,
                    "stat_names": list(obj.stat_names),
                    "required_batch_keys": list(obj.required_batch_keys),
                    "required_model_output_keys": list(obj.required_model_output_keys),
                },
            )
        )
    return loaded


class AuxiliaryLossComposer:
    """Wraps the selected actor loss and composes configured objectives with it.

    The engine treats it as the loss callable and additionally calls :meth:`prepare_global_stats`
    once per mini-batch before micro-batch splitting.
    """

    def __init__(
        self,
        base_loss_fn: Callable,
        objectives: list[LoadedAuxiliaryObjective],
        loss_agg_mode: str,
        selected_token_ids: tuple[int, ...] = (),
    ):
        self.base_loss_fn = base_loss_fn
        self.objectives = list(objectives)
        self.loss_agg_mode = loss_agg_mode
        self.selected_token_ids = tuple(int(t) for t in selected_token_ids)
        names = [o.name for o in self.objectives]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate auxiliary objective names: {names}")
        # Static packing schema for the one SUM reduction: identical on every rank by construction.
        self._stat_slots: list[tuple[str, str]] = [
            (spec.name, stat) for spec in self.objectives for stat in spec.objective.stat_names
        ]

    # ---- introspection ----------------------------------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        """Objective names in execution order."""
        return tuple(o.name for o in self.objectives)

    def config_digest(self) -> str:
        """Stable digest of the resolved objective configuration, compared across ranks at start-up."""
        payload = {
            "selected_token_ids": list(self.selected_token_ids),
            "objectives": [o.digest_payload for o in self.objectives],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def validate_requirements(self, available_model_outputs: set[str]) -> None:
        """Fail at start-up when an objective needs a model output the configured forward will not emit."""
        hints = {
            "entropy": "set actor.calculate_entropy=true",
            "sum_pi_squared": "set actor.calculate_sum_pi_squared=true",
            SELECTED_TOKEN_LOGPROBS_KEY: "set actor.selected_token_logprobs.token_ids",
            "log_probs": "the configured forward skips log_probs (distillation-only mode)",
        }
        for spec in self.objectives:
            missing = [k for k in spec.objective.required_model_output_keys if k not in available_model_outputs]
            if missing:
                advice = "; ".join(hints.get(k, f"'{k}' is not a known model output") for k in missing)
                raise ValueError(
                    f"auxiliary objective '{spec.name}' requires model outputs {missing} which the configured "
                    f"actor forward does not produce ({advice}); available: {sorted(available_model_outputs)}"
                )

    # ---- stage 1: global normalization statistics -------------------------------------------------------
    def prepare_global_stats(self, data: TensorDict, dp_group=None) -> None:
        """Collect every objective's declared statistics on the unsplit mini-batch and SUM-reduce them once.

        Values are packed in the static ``(objective, stat_name)`` order every rank derives from the
        configuration, so ranks never disagree on the tensor shape or on which slot holds which count.
        The result is stored as the non-tensor ``aux_global_stats`` entry so it survives micro-batch
        splitting, exactly like ``batch_num_tokens``. Objectives never issue collectives themselves.
        """
        if not self.objectives:
            return
        for spec in self.objectives:
            missing = [k for k in spec.objective.required_batch_keys if k not in data.keys()]
            if missing:
                raise KeyError(f"auxiliary objective '{spec.name}' requires batch keys {missing} which are absent")
        by_name = {spec.name: spec for spec in self.objectives}
        collected: dict[str, Mapping[str, Any]] = {}
        for spec in self.objectives:
            stats = dict(spec.objective.prepare_batch(data) or {})
            declared = set(spec.objective.stat_names)
            if set(stats) != declared:
                raise ValueError(
                    f"auxiliary objective '{spec.name}'.prepare_batch returned keys {sorted(stats)} but declared "
                    f"stat_names {sorted(declared)}; the schema must be static"
                )
            collected[spec.name] = stats
        values = []
        for name, stat in self._stat_slots:
            value = collected[name][stat]
            if isinstance(value, torch.Tensor):
                if value.requires_grad or value.numel() != 1:
                    raise ValueError(f"auxiliary objective '{name}' stat '{stat}' must be a detached scalar")
                value = value.item()
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"auxiliary objective '{name}' stat '{stat}' is not finite: {value}")
            values.append(value)
        packed = torch.tensor(values, dtype=torch.float64)
        if self._stat_slots and torch.distributed.is_initialized():
            packed = packed.to(get_device_id())
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            packed = packed.cpu()
        reduced: dict[str, dict[str, float]] = {spec.name: {} for spec in self.objectives}
        for (name, stat), value in zip(self._stat_slots, packed.tolist(), strict=True):
            reduced[name][stat] = float(value)
        del by_name
        tu.assign_non_tensor(data, **{AUX_GLOBAL_STATS_KEY: reduced})

    # ---- stage 2: the loss ------------------------------------------------------------------------------
    def __call__(self, model_output=None, data: TensorDict = None, dp_group=None, **kwargs):
        """Compose the base loss with every configured objective.

        Extra keyword arguments (for example ``student_logits`` from the distillation logits-processor
        path) are delegated verbatim to the base loss so existing specialised callers keep working.
        """
        if kwargs:
            return self.base_loss_fn(model_output=model_output, data=data, dp_group=dp_group, **kwargs)
        base_loss, metrics = self.base_loss_fn(model_output=model_output, data=data, dp_group=dp_group)
        if not self.objectives or not torch.is_grad_enabled():
            return base_loss, metrics
        metrics = dict(metrics)
        contexts = self._contexts(data)
        total = base_loss
        for spec in self.objectives:
            missing = [k for k in spec.objective.required_model_output_keys if k not in model_output]
            if missing:
                raise KeyError(f"auxiliary objective '{spec.name}' requires model outputs {missing} which are absent")
            context = contexts[spec.name]
            if spec.metrics_only:
                with torch.no_grad():
                    result = spec.objective.compute(model_output=model_output, data=data, context=context)
                contribution, active, normalizer_value = self._normalize(spec, result, context, require_grad=False)
                weighted = torch.zeros_like(contribution)
            else:
                result = spec.objective.compute(model_output=model_output, data=data, context=context)
                contribution, active, normalizer_value = self._normalize(spec, result, context, require_grad=True)
                weighted = spec.weight * contribution
                total = total + weighted
            prefix = f"{AUX_METRIC_PREFIX}/{spec.name}"
            self._put_metric(metrics, f"{prefix}/loss", Metric(AggregationType.SUM, contribution.detach()))
            self._put_metric(metrics, f"{prefix}/weighted_loss", Metric(AggregationType.SUM, weighted.detach()))
            self._put_metric(metrics, f"{prefix}/normalizer", Metric(AggregationType.MEAN, normalizer_value))
            self._put_metric(metrics, f"{prefix}/active", Metric(AggregationType.MEAN, active))
            for key, value in (result.metrics or {}).items():
                metric = value if isinstance(value, Metric) else Metric(AggregationType.MEAN, value)
                self._put_metric(metrics, f"{prefix}/{key}", metric)
        return total, metrics

    # ---- helpers ------------------------------------------------------------------------------------------
    def _contexts(self, data: TensorDict) -> dict[str, ActorObjectiveContext]:
        stats = tu.get_non_tensor_data(data=data, key=AUX_GLOBAL_STATS_KEY, default=None)
        if stats is None:
            raise RuntimeError(
                "aux_global_stats is missing from the micro-batch: prepare_global_stats must run on the "
                "unsplit mini-batch before the forward (the FSDP engine does this in forward_backward_batch)"
            )
        dp_size = int(tu.get_non_tensor_data(data=data, key="dp_size", default=1))
        batch_num_tokens = tu.get_non_tensor_data(data=data, key="batch_num_tokens", default=None)
        global_batch_size = tu.get_non_tensor_data(data=data, key="global_batch_size", default=None)
        return {
            spec.name: ActorObjectiveContext(
                name=spec.name,
                global_stats=dict(stats.get(spec.name, {})),
                dp_size=dp_size,
                loss_agg_mode=self.loss_agg_mode,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
                selected_token_ids=self.selected_token_ids,
            )
            for spec in self.objectives
        }

    @staticmethod
    def _normalize(
        spec: LoadedAuxiliaryObjective, result: ActorObjectiveResult, context: ActorObjectiveContext, require_grad: bool
    ):
        name = spec.name
        if not isinstance(result, ActorObjectiveResult):
            raise TypeError(
                f"auxiliary objective '{name}'.compute must return ActorObjectiveResult, got {type(result)}"
            )
        loss_sum = result.loss_sum
        if not isinstance(loss_sum, torch.Tensor) or loss_sum.numel() != 1:
            raise ValueError(f"auxiliary objective '{name}' loss_sum must be a scalar tensor")
        if not torch.isfinite(loss_sum).all():
            raise ValueError(f"auxiliary objective '{name}' loss_sum is not finite: {loss_sum.item()}")
        if result.normalizer not in spec.objective.stat_names:
            raise KeyError(
                f"auxiliary objective '{name}' normalizer '{result.normalizer}' is not one of its declared "
                f"stat_names {list(spec.objective.stat_names)}"
            )
        normalizer = float(context.global_stats[result.normalizer])
        if normalizer < 0:
            raise ValueError(f"auxiliary objective '{name}' normalizer '{result.normalizer}' is negative")
        loss_sum = loss_sum.reshape(())
        if normalizer > 0:
            if require_grad and not loss_sum.requires_grad:
                raise ValueError(
                    f"auxiliary objective '{name}' returned a detached loss while active; the objective would "
                    "silently contribute no gradient"
                )
            return loss_sum * (float(context.dp_size) / normalizer), 1.0, normalizer
        if loss_sum.detach().abs().item() != 0.0:
            raise ValueError(
                f"auxiliary objective '{name}' has a zero normalizer over the mini-batch but a non-zero loss_sum"
            )
        return loss_sum * 0.0, 0.0, 0.0

    @staticmethod
    def _put_metric(metrics: dict, key: str, value: Metric) -> None:
        if key in metrics:
            raise KeyError(f"auxiliary objective metric collision on '{key}'")
        metrics[key] = value


def maybe_wrap_with_auxiliary_objectives(
    base_loss_fn: Callable, actor_config, available_model_outputs: Optional[set[str]] = None
) -> Callable:
    """Return ``base_loss_fn`` unchanged when no objectives are configured, else the composer.

    Must be called after the actor's process group exists (the cross-rank configuration handshake is
    mandatory, not best-effort). Fails at initialization on an unsupported training backend, on a
    configuration that differs across ranks, or on an objective whose required model outputs the
    configured forward will not produce.
    """
    configs = list(getattr(actor_config, "auxiliary_objectives", None) or [])
    if not configs:
        return base_loss_fn
    strategy = getattr(actor_config, "strategy", None)
    if strategy not in _SUPPORTED_STRATEGIES:
        raise NotImplementedError(
            f"actor.auxiliary_objectives is only supported with strategy in {_SUPPORTED_STRATEGIES}, got {strategy!r}"
        )
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "actor.auxiliary_objectives must be initialized after the actor process group exists so the "
            "configuration can be checked across ranks"
        )
    selected = getattr(actor_config, "selected_token_logprobs", None)
    selected_ids = tuple(int(t) for t in (getattr(selected, "token_ids", None) or []))
    objectives = load_auxiliary_objectives(configs)
    composer = AuxiliaryLossComposer(
        base_loss_fn=base_loss_fn,
        objectives=objectives,
        loss_agg_mode=getattr(actor_config, "loss_agg_mode", ""),
        selected_token_ids=selected_ids,
    )
    if available_model_outputs is not None:
        composer.validate_requirements(set(available_model_outputs))
    digests = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(digests, composer.config_digest())
    if len(set(digests)) != 1:
        raise RuntimeError(
            "actor.auxiliary_objectives resolves differently across ranks; the configuration must be identical"
        )
    return composer


class SelectedTokenCalibrationObjective(BaseActorAuxiliaryObjective):
    """Reference objective: binary cross-entropy calibration of the mass on a set of token ids.

    Reads the engine-owned ``model_output["selected_token_logprobs"]`` (configure
    ``actor.selected_token_logprobs.token_ids`` to include every id used here). At every response
    position where ``data[target_key]`` is 0 or 1 it forms the logit of the positive set against the
    negative set and applies ``binary_cross_entropy_with_logits``; positions marked -1 are ignored.

    ``normalize_over`` selects what the positive mass is measured against:

    - ``"token_set"`` (default): the negatives are ``negative_token_ids`` (default: every selected id not
      in the positive set), so the probability is conditional on the label alphabet. This is what a
      yes/no reranker reads: ``log_softmax(logits[[no, yes]])[yes]``.
    - ``"vocab"``: the negatives are the whole rest of the vocabulary, so the probability is the raw
      full-vocabulary mass of the positive tokens.

    Batch contract: ``data[target_key]`` is a ``(bsz, response_len)`` tensor (padded or nested) with
    values in ``{-1, 0, 1}``. How it gets there (dataset field, reward-side transform) is up to the
    caller.
    """

    required_model_output_keys = (SELECTED_TOKEN_LOGPROBS_KEY,)
    stat_names = ("cells",)

    def __init__(
        self,
        positive_token_ids,
        negative_token_ids=None,
        normalize_over: str = "token_set",
        target_key: str = "calibration_target",
    ):
        self.positive_token_ids = tuple(int(t) for t in positive_token_ids)
        self.negative_token_ids = None if negative_token_ids is None else tuple(int(t) for t in negative_token_ids)
        if not self.positive_token_ids or len(set(self.positive_token_ids)) != len(self.positive_token_ids):
            raise ValueError("positive_token_ids must be a non-empty list of unique ints")
        if self.negative_token_ids is not None and set(self.negative_token_ids) & set(self.positive_token_ids):
            raise ValueError("negative_token_ids must not overlap positive_token_ids")
        if normalize_over not in ("token_set", "vocab"):
            raise ValueError(f"normalize_over must be 'token_set' or 'vocab', got {normalize_over!r}")
        self.normalize_over = normalize_over
        self.target_key = target_key
        self.required_batch_keys = (target_key, "responses", "prompts")

    def _columns(self, context: ActorObjectiveContext) -> tuple[list[int], list[int]]:
        ids = list(context.selected_token_ids)
        missing = [t for t in self.positive_token_ids if t not in ids]
        if missing:
            raise ValueError(
                f"auxiliary objective '{context.name}': positive_token_ids {missing} are not in "
                f"actor.selected_token_logprobs.token_ids {ids}"
            )
        pos = [ids.index(t) for t in self.positive_token_ids]
        if self.normalize_over == "vocab":
            return pos, []
        negatives = self.negative_token_ids
        if negatives is None:
            negatives = tuple(t for t in ids if t not in self.positive_token_ids)
        missing = [t for t in negatives if t not in ids]
        if missing:
            raise ValueError(
                f"auxiliary objective '{context.name}': negative_token_ids {missing} are not in "
                f"actor.selected_token_logprobs.token_ids {ids}"
            )
        if not negatives:
            raise ValueError(
                f"auxiliary objective '{context.name}': normalize_over='token_set' needs at least one negative "
                "token id (add one to selected_token_logprobs.token_ids or pass negative_token_ids)"
            )
        return pos, [ids.index(t) for t in negatives]

    def _targets(self, data: TensorDict) -> torch.Tensor:
        target = data[self.target_key]
        if target.is_nested:
            target = target.to_padded_tensor(-1)
        return target

    def prepare_batch(self, data: TensorDict) -> Mapping[str, float]:
        """Count calibrated cells over the unsplit mini-batch."""
        return {"cells": float((self._targets(data) >= 0).sum().item())}

    def compute(self, *, model_output, data, context) -> ActorObjectiveResult:
        """BCE-with-logits between the positive-set logit and the binary targets at calibrated cells."""
        from verl.workers.utils.padding import no_padding_2_padding

        pos_cols, neg_cols = self._columns(context)
        lp = no_padding_2_padding(model_output[SELECTED_TOKEN_LOGPROBS_KEY], data)  # (bsz, response_len, M)
        target = self._targets(data).to(lp.device)
        if target.shape[1] < lp.shape[1]:
            pad = torch.full(
                (target.shape[0], lp.shape[1] - target.shape[1]), -1, dtype=target.dtype, device=target.device
            )
            target = torch.cat([target, pad], dim=1)
        target = target[:, : lp.shape[1]]
        mask = (target >= 0).float()
        valid = mask > 0
        lse_pos = torch.logsumexp(lp.index_select(-1, torch.as_tensor(pos_cols, device=lp.device)), dim=-1)
        if neg_cols:
            lse_neg = torch.logsumexp(lp.index_select(-1, torch.as_tensor(neg_cols, device=lp.device)), dim=-1)
        else:
            # Rest of the vocabulary: log(1 - exp(lse_pos)). Padded rows carry zeros (exp(0) = 1) and a
            # saturated row would give log(0): neutralise both before the log so no NaN/inf enters the
            # masked sum. The clamp only touches rows with p_pos > 1 - 1e-7.
            neutral = torch.full_like(lse_pos, -math.log(2.0))
            lse_pos = torch.where(valid, lse_pos, neutral).clamp(max=-1e-7)
            lse_neg = torch.log(-torch.expm1(lse_pos))
        z = lse_pos - lse_neg
        y = (target > 0).float()
        bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        loss_sum = (bce * mask).sum()
        n_local = mask.sum().clamp(min=1.0)
        return ActorObjectiveResult(
            loss_sum=loss_sum,
            normalizer="cells",
            metrics={
                "bce": (bce.detach() * mask).sum() / n_local,
                "positive_prob": (torch.sigmoid(z.detach()) * mask).sum() / n_local,
                "cells_per_seq": mask.sum() / max(lp.shape[0], 1),
            },
        )
