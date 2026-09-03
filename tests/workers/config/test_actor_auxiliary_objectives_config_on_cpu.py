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
"""Config parsing for ``actor.auxiliary_objectives``."""

import pytest

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import AuxiliaryObjectiveConfig, FSDPActorConfig


def _actor_kwargs(objectives):
    return {
        "strategy": "fsdp",
        "ppo_micro_batch_size_per_gpu": 1,
        "rollout_n": 1,
        "auxiliary_objectives": objectives,
    }


def _via_hydra(objectives):
    """The production path: hydra instantiates the ``_target_`` and hands entries over as plain dicts."""
    return omega_conf_to_dataclass(
        {
            "_target_": "verl.workers.config.FSDPActorConfig",
            "optim": {"_target_": "verl.workers.config.FSDPOptimizerConfig", "lr": 0.1},
            **_actor_kwargs(objectives),
        }
    )


def test_default_is_empty_and_loss_path_untouched():
    assert _via_hydra([]).auxiliary_objectives == []
    assert FSDPActorConfig(**_actor_kwargs([])).auxiliary_objectives == []


@pytest.mark.parametrize("via_hydra", [True, False])
def test_entries_become_dataclasses(via_hydra):
    raw = [
        {"name": "cal", "path": "/w/obj.py", "weight": 0.05, "kwargs": {"token_ids": [1, 2]}},
        {"name": "b", "path": "/w/b.py"},
    ]
    cfg = _via_hydra(raw) if via_hydra else FSDPActorConfig(**_actor_kwargs(raw))
    assert all(isinstance(o, AuxiliaryObjectiveConfig) for o in cfg.auxiliary_objectives)
    assert cfg.auxiliary_objectives[0].weight == 0.05
    assert dict(cfg.auxiliary_objectives[0].kwargs) == {"token_ids": [1, 2]}
    assert cfg.auxiliary_objectives[1].factory == "build_objective"
    assert cfg.auxiliary_objectives[1].weight == 1.0


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
    ],
)
def test_invalid_entry_rejected(bad):
    with pytest.raises(ValueError):
        AuxiliaryObjectiveConfig(**bad)


def test_weight_zero_is_allowed():
    entry = AuxiliaryObjectiveConfig(name="a", path="/w/a.py", weight=0)
    assert entry.weight == 0
