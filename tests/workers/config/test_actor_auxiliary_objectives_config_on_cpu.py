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
"""Config parsing for ``actor.auxiliary_objectives`` and ``actor.selected_token_logprobs``."""

import pytest

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import AuxiliaryObjectiveConfig, FSDPActorConfig, SelectedTokenLogprobsConfig


def _actor_kwargs(objectives=(), selected=None):
    kwargs = {
        "strategy": "fsdp",
        "ppo_micro_batch_size_per_gpu": 1,
        "rollout_n": 1,
        "auxiliary_objectives": list(objectives),
    }
    if selected is not None:
        kwargs["selected_token_logprobs"] = selected
    return kwargs


def _via_hydra(objectives=(), selected=None):
    """The production path: hydra instantiates the ``_target_`` and hands entries over as plain dicts."""
    cfg = {
        "_target_": "verl.workers.config.FSDPActorConfig",
        "optim": {"_target_": "verl.workers.config.FSDPOptimizerConfig", "lr": 0.1},
        **_actor_kwargs(objectives, selected),
    }
    if selected is not None:
        cfg["selected_token_logprobs"] = {"_target_": "verl.workers.config.SelectedTokenLogprobsConfig", **selected}
    return omega_conf_to_dataclass(cfg)


def test_defaults_are_off():
    for cfg in (_via_hydra(), FSDPActorConfig(**_actor_kwargs())):
        assert cfg.auxiliary_objectives == []
        assert isinstance(cfg.selected_token_logprobs, SelectedTokenLogprobsConfig)
        assert cfg.selected_token_logprobs.token_ids == []


@pytest.mark.parametrize("via_hydra", [True, False])
def test_entries_become_dataclasses(via_hydra):
    raw = [
        {"name": "cal", "path": "/w/obj.py", "weight": 0.05, "kwargs": {"positive_token_ids": [1, 2]}},
        {"name": "b", "path": "/w/b.py", "metrics_only": True, "weight": 0},
    ]
    selected = {"token_ids": [1, 2, 3]}
    cfg = _via_hydra(raw, selected) if via_hydra else FSDPActorConfig(**_actor_kwargs(raw, selected))
    assert all(isinstance(o, AuxiliaryObjectiveConfig) for o in cfg.auxiliary_objectives)
    assert cfg.auxiliary_objectives[0].weight == 0.05
    assert dict(cfg.auxiliary_objectives[0].kwargs) == {"positive_token_ids": [1, 2]}
    assert cfg.auxiliary_objectives[1].factory == "build_objective"
    assert cfg.auxiliary_objectives[1].metrics_only is True
    assert list(cfg.selected_token_logprobs.token_ids) == [1, 2, 3]


def test_duplicate_names_rejected():
    raw = [{"name": "a", "path": "/w/a.py"}, {"name": "a", "path": "/w/b.py"}]
    with pytest.raises(ValueError, match="unique"):
        FSDPActorConfig(**_actor_kwargs(raw))


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "", "path": "/w/a.py"},
        {"name": "a", "path": ""},
        {"name": "a", "path": "/w/a.py", "weight": float("inf")},
        {"name": "a", "path": "/w/a.py", "weight": True},
        {"name": "a", "path": "/w/a.py", "metrics_only": "yes"},
    ],
)
def test_invalid_entry_rejected(bad):
    with pytest.raises(ValueError):
        AuxiliaryObjectiveConfig(**bad)


@pytest.mark.parametrize("bad_ids", [[1, 1], [-1], [True], [1.5], list(range(1025))])
def test_invalid_selected_token_ids_rejected(bad_ids):
    with pytest.raises(ValueError):
        SelectedTokenLogprobsConfig(token_ids=bad_ids)


def test_weight_zero_is_allowed_in_config():
    entry = AuxiliaryObjectiveConfig(name="a", path="/w/a.py", weight=0)
    assert entry.weight == 0 and entry.metrics_only is False
