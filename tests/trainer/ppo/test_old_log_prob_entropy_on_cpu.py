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

from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("ray")

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import should_calculate_actor_entropy
from verl.utils import tensordict_utils as tu


class _LogProbWorker:
    def __init__(self):
        self.calculate_entropy = None

    def compute_log_prob(self, batch):
        self.calculate_entropy = tu.get(batch, "calculate_entropy")
        output = {"log_probs": torch.ones(2, 3)}
        if self.calculate_entropy:
            output["entropy"] = torch.full((2, 3), 0.5)
        return tu.get_tensordict(output, non_tensor_dict={"metrics": {"mfu": 0.25}})


@pytest.mark.parametrize(
    ("calculate_entropy", "entropy_coeff", "expected"),
    [(False, 0.0, False), (True, 0.0, True), (False, 0.01, True)],
)
def test_should_calculate_actor_entropy(calculate_entropy, entropy_coeff, expected):
    actor_config = OmegaConf.create({"calculate_entropy": calculate_entropy, "entropy_coeff": entropy_coeff})

    assert should_calculate_actor_entropy(actor_config) is expected


@pytest.mark.parametrize("calculate_entropy", [False, True])
def test_compute_old_log_prob_only_returns_requested_entropy(calculate_entropy):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": calculate_entropy,
                    "entropy_coeff": 0.0,
                    "calculate_sum_pi_squared": False,
                }
            }
        }
    )
    trainer.actor_rollout_wg = _LogProbWorker()
    batch = DataProto.from_single_dict({"input_ids": torch.ones(2, 3, dtype=torch.long)})

    with (
        patch("verl.trainer.ppo.ray_trainer.left_right_2_no_padding", side_effect=lambda data: data),
        patch("verl.trainer.ppo.ray_trainer.no_padding_2_padding", side_effect=lambda tensor, _: tensor),
    ):
        old_log_prob, mfu = trainer._compute_old_log_prob(batch)

    assert trainer.actor_rollout_wg.calculate_entropy is calculate_entropy
    assert ("entropys" in old_log_prob.batch) is calculate_entropy
    assert torch.equal(old_log_prob.batch["old_log_probs"], torch.ones(2, 3))
    assert mfu == 0.25
