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

Each objective declares the batch fields and model outputs it reads, may optionally reduce
the transient logits of the same actor forward into compact, namespaced outputs, and returns an
unweighted loss *sum* plus the name of the statistic that normalizes it. verl owns loading,
execution order, global (mini-batch) normalization, coefficient application, metric
namespacing and validation. An empty objective list leaves the actor loss untouched.

Lifecycle for one actor mini-batch::

    prepare_global_stats(data)                    # once, on the unsplit mini-batch, SUM over DP
        -> for each micro-batch:
             forward -> process_logits(logits)    # optional, same forward, logits still transient
                     -> base loss (pg, entropy, KL)
                     -> compute(model_output, data) for every objective, in config order
             backward on the combined scalar

Normalization: for objective ``i`` and micro-batch ``m`` the contribution is
``dp_size * loss_sum_i^(m) / N_i`` where ``N_i`` is the SUM-reduced statistic named by
``ActorObjectiveResult.normalizer``. Summed over micro-batches and averaged over data-parallel ranks
by the gradient reduction this yields the global mini-batch mean, invariant to how the mini-batch was
split, mirroring ``batch_num_tokens`` in :func:`verl.workers.utils.losses.ppo_loss`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import torch
import torch.distributed
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.utils.import_utils import load_extern_object
from verl.utils.metric import AggregationType, Metric

__all__ = [
    "AUX_METRIC_PREFIX",
    "AUX_OUTPUT_PREFIX",
    "AUX_GLOBAL_STATS_KEY",
    "ActorObjectiveContext",
    "ActorObjectiveResult",
    "ActorAuxiliaryObjective",
    "BaseActorAuxiliaryObjective",
    "LoadedAuxiliaryObjective",
    "AuxiliaryLossComposer",
    "load_auxiliary_objectives",
    "maybe_wrap_with_auxiliary_objectives",
    "SpecifiedTokenCalibrationObjective",
]

# Metrics land under ``actor/aux/<name>/...``; processor outputs under ``aux/<name>/<key>`` in model_output.
AUX_METRIC_PREFIX = "actor/aux"
AUX_OUTPUT_PREFIX = "aux"
# Non-tensor key on the mini-batch TensorDict carrying ``{objective_name: {stat_name: float}}``.
AUX_GLOBAL_STATS_KEY = "aux_global_stats"

_SUPPORTED_STRATEGIES = ("fsdp", "fsdp2")


@dataclass(frozen=True)
class ActorObjectiveContext:
    """Read-only, framework-owned context handed to every objective callback.

    Args:
        name: The objective's configured name; its processor outputs live at ``aux/<name>/<key>``.
        global_stats: This objective's ``prepare_batch`` statistics after the SUM reduction over the
            data-parallel group, i.e. totals over the whole actor mini-batch.
        dp_size: Data-parallel world size of the actor.
        loss_agg_mode: The actor's ``loss_agg_mode`` (informational; the composer normalizes by
            ``global_stats[result.normalizer]``).
        batch_num_tokens: Number of loss-mask tokens in the global mini-batch, or None.
        global_batch_size: Number of sequences in the global mini-batch, or None.
    """

    name: str
    global_stats: Mapping[str, float]
    dp_size: int
    loss_agg_mode: str
    batch_num_tokens: Optional[int] = None
    global_batch_size: Optional[int] = None


@dataclass(frozen=True)
class ActorObjectiveResult:
    """What ``compute`` returns for one micro-batch.

    Args:
        loss_sum: Unweighted, un-normalized *sum* of the per-element losses of this micro-batch. A
            scalar tensor connected to the actor's autograd graph. When the objective has no
            applicable element in this micro-batch, return a differentiable zero (for example
            ``some_output.sum() * 0.0``).
        normalizer: Name of the ``prepare_batch`` statistic that counts the applicable elements over
            the whole mini-batch. The composer divides by its SUM-reduced value.
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

    All callbacks must treat their inputs as read-only and must not run distributed collectives,
    call ``backward`` or touch the optimizer; the composer owns those.
    """

    required_batch_keys: tuple[str, ...]
    required_model_output_keys: tuple[str, ...]
    uses_logits_processor: bool

    def prepare_batch(self, data: TensorDict) -> Mapping[str, float]:
        """Return local additive statistics (counts, sums) of the *unsplit* mini-batch, no gradients."""
        ...

    def process_logits(
        self, *, logits: torch.Tensor, data: TensorDict, context: ActorObjectiveContext
    ) -> Mapping[str, torch.Tensor]:
        """Reduce transient ``(total_nnz, vocab)`` logits into compact ``(total_nnz, ...)`` tensors."""
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
    uses_logits_processor: bool = False

    def prepare_batch(self, data: TensorDict) -> Mapping[str, float]:
        """Default: no statistics. Objectives that normalize must override."""
        return {}

    def process_logits(
        self, *, logits: torch.Tensor, data: TensorDict, context: ActorObjectiveContext
    ) -> Mapping[str, torch.Tensor]:
        """Default: not a logits processor."""
        raise NotImplementedError(f"{type(self).__name__} does not declare uses_logits_processor")

    def compute(
        self, *, model_output: Mapping[str, torch.Tensor], data: TensorDict, context: ActorObjectiveContext
    ) -> ActorObjectiveResult:
        """Objectives must implement ``compute``."""
        raise NotImplementedError


@dataclass(frozen=True)
class LoadedAuxiliaryObjective:
    """A configured objective instance together with its name and coefficient."""

    name: str
    weight: float
    objective: Any
    digest_payload: Mapping[str, Any] = field(default_factory=dict)


def _validate_objective_instance(name: str, obj: Any) -> None:
    for attr in ("required_batch_keys", "required_model_output_keys", "uses_logits_processor"):
        if not hasattr(obj, attr):
            raise TypeError(f"auxiliary objective '{name}' is missing attribute '{attr}'")
    for method in ("prepare_batch", "compute"):
        if not callable(getattr(obj, method, None)):
            raise TypeError(f"auxiliary objective '{name}' is missing method '{method}'")
    if obj.uses_logits_processor and not callable(getattr(obj, "process_logits", None)):
        raise TypeError(f"auxiliary objective '{name}' declares uses_logits_processor but has no process_logits")
    for attr in ("required_batch_keys", "required_model_output_keys"):
        keys = getattr(obj, attr)
        if not isinstance(keys, tuple | list) or not all(isinstance(k, str) for k in keys):
            raise TypeError(f"auxiliary objective '{name}'.{attr} must be a tuple of str, got {keys!r}")


def load_auxiliary_objectives(configs) -> list[LoadedAuxiliaryObjective]:
    """Instantiate objectives from ``actor.auxiliary_objectives`` config entries.

    Each entry names a python file (``path``), a factory in it (``factory``), a ``weight`` and
    ``kwargs`` forwarded to the factory. Names must be unique and weights finite; both are also
    checked by the config dataclass, so this is the second line of defense for hand-built configs.
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
                digest_payload={"name": name, "weight": weight, "path": cfg.path, "factory": cfg.factory},
            )
        )
    return loaded


class AuxiliaryLossComposer:
    """Wraps the selected actor loss and composes configured objectives with it.

    The engine treats it as the loss callable and additionally calls two optional stages when
    present: :meth:`prepare_global_stats` once per mini-batch before micro-batch splitting, and
    :meth:`process_logits` per micro-batch while the logits are still alive.
    """

    def __init__(self, base_loss_fn: Callable, objectives: list[LoadedAuxiliaryObjective], loss_agg_mode: str):
        self.base_loss_fn = base_loss_fn
        self.objectives = list(objectives)
        self.loss_agg_mode = loss_agg_mode
        names = [o.name for o in self.objectives]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate auxiliary objective names: {names}")

    # ---- introspection ----------------------------------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        """Objective names in execution order."""
        return tuple(o.name for o in self.objectives)

    @property
    def has_logits_processors(self) -> bool:
        """Whether any objective declared ``uses_logits_processor``."""
        return any(o.objective.uses_logits_processor for o in self.objectives)

    def config_digest(self) -> str:
        """Stable digest of the ordered objective configuration, compared across ranks at init."""
        payload = [o.digest_payload for o in self.objectives]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    # ---- stage 1: global normalization statistics -------------------------------------------------------
    def prepare_global_stats(self, data: TensorDict, dp_group=None) -> None:
        """Collect every objective's local statistics on the unsplit mini-batch and SUM-reduce them once.

        The result is stored as the non-tensor ``aux_global_stats`` entry so it survives micro-batch
        splitting, exactly like ``batch_num_tokens``. Objectives never issue collectives themselves.
        """
        if not self.objectives:
            return
        for spec in self.objectives:
            missing = [k for k in spec.objective.required_batch_keys if k not in data.keys()]
            if missing:
                raise KeyError(f"auxiliary objective '{spec.name}' requires batch keys {missing} which are absent")
        local: list[tuple[str, str, float]] = []
        for spec in self.objectives:
            stats = spec.objective.prepare_batch(data) or {}
            for stat_name, value in stats.items():
                if isinstance(value, torch.Tensor):
                    if value.requires_grad or value.numel() != 1:
                        raise ValueError(
                            f"auxiliary objective '{spec.name}' stat '{stat_name}' must be a detached scalar"
                        )
                    value = value.item()
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"auxiliary objective '{spec.name}' stat '{stat_name}' is not finite: {value}")
                local.append((spec.name, stat_name, value))
        packed = torch.tensor([v for _, _, v in local], dtype=torch.float64)
        if torch.distributed.is_initialized() and packed.numel() > 0:
            packed = packed.to(get_device_id())
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            packed = packed.cpu()
        reduced: dict[str, dict[str, float]] = {spec.name: {} for spec in self.objectives}
        for (name, stat_name, _), value in zip(local, packed.tolist(), strict=True):
            reduced[name][stat_name] = float(value)
        tu.assign_non_tensor(data, **{AUX_GLOBAL_STATS_KEY: reduced})

    # ---- stage 2: optional same-forward logits processing ----------------------------------------------
    def process_logits(self, *, logits: torch.Tensor, data: TensorDict) -> dict[str, torch.Tensor]:
        """Run every declared processor on the transient logits and namespace the compact outputs.

        ``logits`` are exactly the logits ``log_probs`` is computed from (already divided by the
        rollout temperature), laid out as ``(total_nnz, vocab)``. Each returned tensor must have
        ``total_nnz`` as its leading dimension; the engine re-nests them per sequence.
        """
        outputs: dict[str, torch.Tensor] = {}
        if not self.has_logits_processors:
            return outputs
        contexts = self._contexts(data)
        for spec in self.objectives:
            if not spec.objective.uses_logits_processor:
                continue
            produced = spec.objective.process_logits(logits=logits, data=data, context=contexts[spec.name])
            for key, value in (produced or {}).items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"auxiliary objective '{spec.name}' processor output '{key}' is not a tensor")
                if value.dim() == 0 or value.shape[0] != logits.shape[0]:
                    raise ValueError(
                        f"auxiliary objective '{spec.name}' processor output '{key}' must have leading dim "
                        f"{logits.shape[0]} (total_nnz), got {tuple(value.shape)}"
                    )
                if value.shape[1:].numel() > logits.shape[-1] // 4:
                    raise ValueError(
                        f"auxiliary objective '{spec.name}' processor output '{key}' is not compact: "
                        f"{tuple(value.shape[1:])} per token exceeds vocab/4"
                    )
                outputs[f"{AUX_OUTPUT_PREFIX}/{spec.name}/{key}"] = value
        return outputs

    # ---- stage 3: the loss ------------------------------------------------------------------------------
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
            result = spec.objective.compute(model_output=model_output, data=data, context=context)
            contribution, active, normalizer_value = self._normalize(spec, result, context)
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
            )
            for spec in self.objectives
        }

    @staticmethod
    def _normalize(spec: LoadedAuxiliaryObjective, result: ActorObjectiveResult, context: ActorObjectiveContext):
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
        if result.normalizer not in context.global_stats:
            raise KeyError(
                f"auxiliary objective '{name}' normalizer '{result.normalizer}' was not returned by prepare_batch; "
                f"available: {sorted(context.global_stats)}"
            )
        normalizer = float(context.global_stats[result.normalizer])
        if normalizer < 0:
            raise ValueError(f"auxiliary objective '{name}' normalizer '{result.normalizer}' is negative")
        loss_sum = loss_sum.reshape(())
        if normalizer > 0:
            if not loss_sum.requires_grad:
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


def maybe_wrap_with_auxiliary_objectives(base_loss_fn: Callable, actor_config) -> Callable:
    """Return ``base_loss_fn`` unchanged when no objectives are configured, else the composer.

    Fails fast at initialization on an unsupported training backend or when a declared logits
    processor cannot be honoured because fused kernels never materialize the logits.
    """
    configs = list(getattr(actor_config, "auxiliary_objectives", None) or [])
    if not configs:
        return base_loss_fn
    strategy = getattr(actor_config, "strategy", None)
    if strategy not in _SUPPORTED_STRATEGIES:
        raise NotImplementedError(
            f"actor.auxiliary_objectives is only supported with strategy in {_SUPPORTED_STRATEGIES}, got {strategy!r}"
        )
    objectives = load_auxiliary_objectives(configs)
    composer = AuxiliaryLossComposer(
        base_loss_fn=base_loss_fn, objectives=objectives, loss_agg_mode=getattr(actor_config, "loss_agg_mode", "")
    )
    if composer.has_logits_processors:
        engine = getattr(actor_config, "engine", None)
        use_fused = bool(getattr(actor_config, "use_fused_kernels", False)) or bool(
            getattr(engine, "use_fused_kernels", False)
        )
        if use_fused:
            names = [o.name for o in objectives if o.objective.uses_logits_processor]
            raise NotImplementedError(
                f"auxiliary objectives {names} declare a logits processor, which needs the full logits; "
                "use_fused_kernels=True never materializes them. Disable fused kernels or the processor."
            )
    if torch.distributed.is_initialized():
        digests = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(digests, composer.config_digest())
        if len(set(digests)) != 1:
            raise RuntimeError("actor.auxiliary_objectives differs across ranks; the configuration must be identical")
    return composer


class SpecifiedTokenCalibrationObjective(BaseActorAuxiliaryObjective):
    """Reference objective: binary cross-entropy calibration of the mass on a set of token IDs.

    At every response position where ``data[target_key] >= 0`` the objective reads the current-policy
    probability mass assigned to ``positive_token_ids`` (out of ``token_ids``) and pulls it toward
    the binary target. Positions with a negative target are ignored. The reduction happens in
    ``process_logits`` so only ``len(token_ids)`` values per token leave the forward; the full logits
    are never retained.

    Batch contract: ``data[target_key]`` is a ``(bsz, response_len)`` tensor (padded or nested) with
    values in ``{-1, 0, 1}``. How it gets there (dataset field, reward-side transform) is up to the
    caller.
    """

    uses_logits_processor = True
    required_model_output_keys = ()

    def __init__(self, token_ids, positive_token_ids, target_key: str = "calibration_target", eps: float = 1e-6):
        token_ids = [int(t) for t in token_ids]
        if len(set(token_ids)) != len(token_ids) or not token_ids:
            raise ValueError("token_ids must be a non-empty list of unique ints")
        self.token_ids = token_ids
        self.positive_columns = [token_ids.index(int(t)) for t in positive_token_ids]
        if not self.positive_columns:
            raise ValueError("positive_token_ids must select at least one of token_ids")
        self.target_key = target_key
        self.eps = float(eps)
        self.required_batch_keys = (target_key, "responses", "prompts")

    def _targets(self, data: TensorDict) -> torch.Tensor:
        target = data[self.target_key]
        if target.is_nested:
            target = target.to_padded_tensor(-1)
        return target

    def prepare_batch(self, data: TensorDict) -> Mapping[str, float]:
        """Count calibrated cells over the unsplit mini-batch."""
        return {"cells": float((self._targets(data) >= 0).sum().item())}

    def process_logits(self, *, logits, data, context):
        """Gather the requested columns of the full-vocabulary log-softmax."""
        ids = torch.as_tensor(self.token_ids, device=logits.device)
        lp = logits.index_select(-1, ids) - torch.logsumexp(logits, dim=-1, keepdim=True)
        return {"logprobs": lp.float()}

    def compute(self, *, model_output, data, context) -> ActorObjectiveResult:
        """BCE between the positive-column mass and the binary targets at calibrated cells."""
        from verl.workers.utils.padding import no_padding_2_padding

        lp = no_padding_2_padding(
            model_output[f"{AUX_OUTPUT_PREFIX}/{context.name}/logprobs"], data
        )  # (bsz, response_len, M)
        target = self._targets(data).to(lp.device)
        if target.shape[1] < lp.shape[1]:
            pad = torch.full(
                (target.shape[0], lp.shape[1] - target.shape[1]), -1, dtype=target.dtype, device=target.device
            )
            target = torch.cat([target, pad], dim=1)
        target = target[:, : lp.shape[1]]
        mask = (target >= 0).float()
        cols = torch.as_tensor(self.positive_columns, device=lp.device)
        p = torch.exp(torch.logsumexp(lp.index_select(-1, cols), dim=-1)).clamp(self.eps, 1 - self.eps)
        y = (target > 0).float()
        bce = -(y * torch.log(p) + (1 - y) * torch.log1p(-p))
        loss_sum = (bce * mask).sum()
        n_local = mask.sum().clamp(min=1.0)
        return ActorObjectiveResult(
            loss_sum=loss_sum,
            normalizer="cells",
            metrics={"bce": (bce.detach() * mask).sum() / n_local, "cells_per_seq": mask.sum() / max(lp.shape[0], 1)},
        )
