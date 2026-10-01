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

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from verl.utils.config import validate_config
from verl.workers.config import SelectedTokenLogprobsConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.logprobs import (
    SELECTED_TOKEN_LOGPROBS_KEY,
    SelectedTokenLogprobs,
    build_selected_token_logprobs,
    finalize_selected_token_logprobs,
)

_DENSE = np.array([[-0.1, -1.1], [-0.2, -1.2], [-0.3, -1.3], [-0.4, -1.4]], dtype=np.float32)


def _payload(
    *,
    response_ids: tuple[int, ...] = (11, 12, 13),
    positions: np.ndarray | None = None,
    logprobs: np.ndarray | None = None,
    logprobs_mode: str = "processed_logprobs",
) -> SelectedTokenLogprobs:
    if positions is None:
        positions = np.array([0, 2], dtype=np.int32)
    if logprobs is None:
        logprobs = np.array([[-0.1, -1.1], [-0.2, -1.2]], dtype=np.float32)
    return SelectedTokenLogprobs(
        token_ids=(101, 7),
        response_ids=response_ids,
        positions=positions,
        logprobs=logprobs,
        logprobs_mode=logprobs_mode,
        backend="vllm",
        backend_version="0.24.0",
    )


def _build(dense: np.ndarray, configured_positions, **overrides) -> SelectedTokenLogprobs:
    values = {
        "token_ids": [101, 7],
        "response_ids": tuple(range(11, 11 + dense.shape[0])),
        "logprobs_mode": "processed_logprobs",
        "backend": "vllm",
        "backend_version": "0.24.0",
    }
    values.update(overrides)
    return build_selected_token_logprobs(dense, configured_positions, **values)


def _selected_config(**overrides) -> SelectedTokenLogprobsConfig:
    values = {"token_ids": [101, 7], "positions": [0, 2], "max_payload_bytes_per_sample": 1024}
    values.update(overrides)
    return SelectedTokenLogprobsConfig(**values)


def _rollout_config(**overrides) -> RolloutConfig:
    values = {"name": "vllm", "response_length": 8, "selected_token_logprobs": _selected_config()}
    values.update(overrides)
    return RolloutConfig(**values)


def test_payload_key_is_stable():
    assert SELECTED_TOKEN_LOGPROBS_KEY == "selected_token_logprobs"


def test_payload_preserves_caller_order_and_owns_readonly_contiguous_arrays():
    source_positions = np.arange(4, dtype=np.int32)[::2]
    source_logprobs = np.asfortranarray(np.array([[-0.1, -1.1], [-0.2, -1.2]], dtype=np.float32))

    payload = _payload(positions=source_positions, logprobs=source_logprobs)

    assert payload.token_ids == (101, 7)
    assert payload.response_token_count == 3
    assert payload.positions.flags.c_contiguous
    assert payload.logprobs.flags.c_contiguous
    assert not payload.positions.flags.writeable
    assert not payload.logprobs.flags.writeable
    assert not np.shares_memory(payload.positions, source_positions)
    assert not np.shares_memory(payload.logprobs, source_logprobs)

    source_positions[0] = 1
    source_logprobs[0, 0] = -99.0
    assert payload.positions.tolist() == [0, 2]
    assert payload.logprobs[0, 0] == pytest.approx(-0.1)


def test_payload_represents_present_but_empty_positions_with_typed_arrays():
    payload = _payload(
        response_ids=(),
        positions=np.empty((0,), dtype=np.int32),
        logprobs=np.empty((0, 2), dtype=np.float32),
    )

    assert payload.response_token_count == 0
    assert payload.positions.shape == (0,)
    assert payload.positions.dtype == np.int32
    assert payload.logprobs.shape == (0, 2)
    assert payload.logprobs.dtype == np.float32


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("positions", np.array([0, 2], dtype=np.int64), "dtype int32"),
        ("logprobs", np.zeros((2, 2), dtype=np.float64), "dtype float32"),
        ("positions", np.zeros((1, 2), dtype=np.int32), "shape \\[P\\]"),
        ("logprobs", np.zeros((2,), dtype=np.float32), "shape \\[P, M\\]"),
        ("positions", np.array([0, 0], dtype=np.int32), "strictly increasing"),
        ("positions", np.array([0, 3], dtype=np.int32), "smaller than the response length"),
        ("logprobs", np.array([[-0.1, np.nan], [-0.2, -1.2]], dtype=np.float32), "must not contain NaN"),
        ("logprobs", np.array([[-0.1, np.inf], [-0.2, -1.2]], dtype=np.float32), "must be non-positive"),
        ("logprobs", np.array([[-0.1, 0.01], [-0.2, -1.2]], dtype=np.float32), "must be non-positive"),
    ],
)
def test_payload_rejects_invalid_arrays(field, value, match):
    with pytest.raises(ValidationError, match=match):
        _payload(**{field: value})


@pytest.mark.parametrize("logprobs_mode", ["raw_logits", "processed_logits"])
def test_payload_allows_positive_values_under_logit_modes(logprobs_mode):
    payload = _payload(logprobs=np.array([[3.5, -1.1], [0.0, 12.0]], dtype=np.float32), logprobs_mode=logprobs_mode)

    assert payload.logprobs_mode == logprobs_mode
    assert payload.logprobs[1, 1] == pytest.approx(12.0)


def test_payload_rejects_matrix_shape_mismatch():
    with pytest.raises(ValidationError, match="logprobs must have shape"):
        _payload(logprobs=np.zeros((1, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": True},
        {"token_ids": (True, 7)},
        {"response_ids": (True, 12, 13)},
        {"logprobs_mode": ""},
    ],
)
def test_payload_rejects_invalid_scalars(overrides):
    values = {
        "token_ids": (101, 7),
        "response_ids": (11, 12, 13),
        "positions": np.array([0, 2], dtype=np.int32),
        "logprobs": np.zeros((2, 2), dtype=np.float32),
        "logprobs_mode": "processed_logprobs",
        "backend": "vllm",
        "backend_version": "0.24.0",
    }
    values.update(overrides)
    with pytest.raises(ValidationError):
        SelectedTokenLogprobs(**values)


def test_build_keeps_every_row_when_positions_are_not_configured():
    payload = _build(_DENSE, None)

    assert payload.response_ids == (11, 12, 13, 14)
    assert payload.positions.tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(payload.logprobs, _DENSE)
    assert not payload.logprobs.flags.writeable


def test_build_validates_dense_matrix_then_gathers_realized_positions():
    payload = _build(_DENSE, [0, 2, 5])

    assert payload.response_token_count == 4
    assert payload.positions.tolist() == [0, 2]
    np.testing.assert_array_equal(payload.logprobs, _DENSE[[0, 2]])


def test_build_validates_unselected_dense_rows_before_gathering():
    dense = np.array([[-0.1, -1.1], [np.nan, -1.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="dense_logprobs must not contain NaN"):
        _build(dense, [0])


def test_build_rejects_positive_logprob_in_unselected_dense_row():
    dense = np.array([[-0.1, -1.1], [0.01, -1.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="dense_logprobs must be non-positive"):
        _build(dense, [0])

    np.testing.assert_allclose(_build(dense, [0], logprobs_mode="raw_logits").logprobs, dense[[0]])


def test_build_rejects_row_count_that_does_not_match_response_ids():
    with pytest.raises(ValueError, match="row count must match response_ids"):
        _build(_DENSE, None, response_ids=(11, 12))


def test_build_filters_python_positions_before_int32_cast():
    payload = _build(np.array([[-0.1, -1.1]], dtype=np.float32), [0, 2**40])

    assert payload.positions.tolist() == [0]


def test_build_returns_typed_empty_when_response_is_shorter_than_every_position():
    payload = _build(np.empty((0, 2), dtype=np.float32), [3, 7])

    assert payload.response_token_count == 0
    assert payload.positions.shape == (0,)
    assert payload.logprobs.shape == (0, 2)


def test_finalize_filters_rows_to_exact_final_response_prefix():
    payload = _build(_DENSE, [0, 2, 5])

    finalized = finalize_selected_token_logprobs(
        payload,
        configured_positions=[0, 2, 5],
        configured_token_ids=[101, 7],
        final_response_ids=[11, 12],
    )

    assert finalized.response_ids == (11, 12)
    assert finalized.positions.tolist() == [0]
    np.testing.assert_array_equal(finalized.logprobs, _DENSE[[0]])
    assert finalized.logprobs_mode == "processed_logprobs"
    assert not finalized.positions.flags.writeable
    assert not finalized.logprobs.flags.writeable


def test_finalize_truncates_dense_payload_to_final_prefix():
    finalized = finalize_selected_token_logprobs(
        _build(_DENSE, None),
        configured_positions=None,
        configured_token_ids=[101, 7],
        final_response_ids=[11, 12, 13],
    )

    assert finalized.positions.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(finalized.logprobs, _DENSE[:3])


def test_finalize_rejects_non_prefix_final_response():
    with pytest.raises(ValueError, match="exact prefix"):
        finalize_selected_token_logprobs(
            _payload(),
            configured_positions=[0, 2],
            configured_token_ids=[101, 7],
            final_response_ids=[11, 99],
        )


def test_finalize_rejects_payload_token_ids_that_do_not_match_configured_order():
    with pytest.raises(ValueError, match="exactly match configured_token_ids"):
        finalize_selected_token_logprobs(
            _payload(),
            configured_positions=[0, 2],
            configured_token_ids=[7, 101],
            final_response_ids=[11, 12, 13],
        )


@pytest.mark.parametrize("configured_positions", [[0, 2], None])
def test_finalize_rejects_missing_configured_in_range_position(configured_positions):
    incomplete = _payload(positions=np.array([0], dtype=np.int32), logprobs=np.array([[-0.1, -1.1]], dtype=np.float32))

    with pytest.raises(ValueError, match="missing configured in-range"):
        finalize_selected_token_logprobs(
            incomplete,
            configured_positions=configured_positions,
            configured_token_ids=[101, 7],
            final_response_ids=[11, 12, 13],
        )


def test_selected_config_defaults_to_disabled():
    config = SelectedTokenLogprobsConfig()

    assert config.enabled is False
    assert config.token_ids is None
    assert config.positions is None


def test_selected_config_positions_require_token_ids():
    with pytest.raises(ValueError, match="positions requires token_ids"):
        SelectedTokenLogprobsConfig(token_ids=None, positions=[0])


def test_selected_config_accepts_dense_capture_without_positions():
    config = SelectedTokenLogprobsConfig(token_ids=[9, 3])

    assert config.enabled is True
    assert config.positions is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"token_ids": []}, "non-empty"),
        ({"token_ids": [True]}, "Python integers"),
        ({"token_ids": [-1]}, "non-negative"),
        ({"token_ids": [7, 7]}, "unique"),
        ({"token_ids": list(range(129))}, "logprob_token_ids limit"),
        ({"positions": []}, "non-empty"),
        ({"positions": [True]}, "Python integers"),
        ({"positions": [-1]}, "non-negative"),
        ({"positions": [2, 1]}, "strictly increasing"),
        ({"positions": [1, 1]}, "strictly increasing"),
    ],
)
def test_selected_config_rejects_invalid_token_ids_and_positions(overrides, match):
    with pytest.raises(ValueError, match=match):
        _selected_config(**overrides)


def test_selected_config_preserves_unsorted_caller_token_order():
    assert _selected_config(token_ids=[9, 3, 7]).token_ids == [9, 3, 7]


@pytest.mark.parametrize("value", [0, -1, True])
def test_selected_config_byte_limit_is_a_positive_python_integer(value):
    with pytest.raises(ValueError, match="positive Python integer"):
        _selected_config(max_payload_bytes_per_sample=value)


def test_rollout_config_accepts_dataclass_dict_and_dictconfig():
    dataclass_config = _selected_config()
    from_dataclass = _rollout_config(selected_token_logprobs=dataclass_config)
    from_dict = _rollout_config(selected_token_logprobs={"token_ids": [101, 7], "positions": [0, 2]})
    from_dictconfig = _rollout_config(selected_token_logprobs=OmegaConf.create({"token_ids": [101, 7]}))

    assert from_dataclass.selected_token_logprobs is dataclass_config
    assert isinstance(from_dict.selected_token_logprobs, SelectedTokenLogprobsConfig)
    assert isinstance(from_dictconfig.selected_token_logprobs, SelectedTokenLogprobsConfig)
    assert from_dictconfig.selected_token_logprobs.positions is None


@pytest.mark.parametrize("logprobs_mode", ["processed_logprobs", "raw_logprobs", "raw_logits"])
def test_rollout_config_follows_any_engine_logprobs_mode(logprobs_mode):
    config = _rollout_config(logprobs_mode=logprobs_mode)

    assert config.logprobs_mode == logprobs_mode
    assert config.selected_token_logprobs.enabled


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"name": "sglang"}, ValueError, "only by rollout.name='vllm'"),
        ({"multi_turn": {"enable": True}}, NotImplementedError, "multi-turn"),
        ({"agent": {"default_agent_loop": "tool_agent"}}, NotImplementedError, "single_turn_agent"),
        ({"agent": {"agent_loop_config_path": "custom-agent.yaml"}}, NotImplementedError, "agent_loop_config_path"),
        ({"agent": {"agent_loop_manager_class": "custom.Manager"}}, NotImplementedError, "agent_loop_manager_class"),
        ({"mtp": {"enable_rollout": True}}, NotImplementedError, "speculative decoding"),
        ({"disaggregation": {"enabled": True}}, NotImplementedError, "disaggregation"),
    ],
)
def test_rollout_config_feature_gates(overrides, error, match):
    with pytest.raises(error, match=match):
        _rollout_config(**overrides)


def test_rollout_config_rejects_position_outside_configured_response_length():
    with pytest.raises(ValueError, match="smaller than rollout.response_length"):
        _rollout_config(response_length=8, selected_token_logprobs=_selected_config(positions=[0, 8]))


def test_rollout_config_rejects_payload_over_byte_limit_for_configured_positions():
    # P=2 and M=2 owns int32[P] + float32[P,M] = 24 bytes.
    with pytest.raises(ValueError, match="exceeds max_payload_bytes_per_sample"):
        _rollout_config(selected_token_logprobs=_selected_config(max_payload_bytes_per_sample=23))


def test_rollout_config_sizes_dense_payload_from_response_length():
    # P=response_length=8 and M=2 owns 8*4 + 8*2*4 = 96 bytes.
    _rollout_config(selected_token_logprobs=_selected_config(positions=None, max_payload_bytes_per_sample=96))
    with pytest.raises(ValueError, match="exceeds max_payload_bytes_per_sample"):
        _rollout_config(selected_token_logprobs=_selected_config(positions=None, max_payload_bytes_per_sample=95))


def test_canonical_rollout_yaml_exposes_disabled_typed_config():
    config = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml").selected_token_logprobs

    assert config._target_ == "verl.workers.config.SelectedTokenLogprobsConfig"
    assert config.token_ids is None
    assert config.positions is None
    assert config.max_payload_bytes_per_sample == 4194304


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _invalid_reward_topology_config(
    *, reward_model_enabled: bool, enable_resource_pool: bool, custom_path, num_workers: int = 1
):
    return SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
        actor_rollout_ref=SimpleNamespace(rollout=_AttrDict(selected_token_logprobs=_AttrDict(token_ids=[101]))),
        algorithm=_AttrDict(rollout_correction=None),
        reward=_AttrDict(
            num_workers=num_workers,
            reward_model=_AttrDict(enable=reward_model_enabled, enable_resource_pool=enable_resource_pool),
            custom_reward_function=_AttrDict(path=custom_path),
        ),
    )


def test_validate_config_rejects_colocated_reward_model_before_rollout():
    config = _invalid_reward_topology_config(reward_model_enabled=True, enable_resource_pool=False, custom_path="r.py")

    with pytest.raises(NotImplementedError, match="require streaming reward workers"):
        validate_config(config, use_reference_policy=False, use_critic=False)


def test_validate_config_rejects_builtin_discriminative_reward_path_before_rollout():
    config = _invalid_reward_topology_config(reward_model_enabled=True, enable_resource_pool=True, custom_path=None)

    with pytest.raises(NotImplementedError, match="built-in discriminative reward model path"):
        validate_config(config, use_reference_policy=False, use_critic=False)


def test_validate_config_requires_a_streaming_reward_worker():
    config = _invalid_reward_topology_config(
        reward_model_enabled=False, enable_resource_pool=False, custom_path=None, num_workers=0
    )

    with pytest.raises(ValueError, match="reward.num_workers to be a positive Python integer"):
        validate_config(config, use_reference_policy=False, use_critic=False)


@pytest.mark.parametrize(
    "rollout_correction",
    [
        {"rollout_is": "token", "rollout_rs": None, "bypass_mode": False},
        {"rollout_is": None, "rollout_rs": "token_k1", "bypass_mode": False},
        {"rollout_is": None, "rollout_rs": None, "bypass_mode": True},
    ],
)
def test_validate_config_allows_active_rollout_correction(rollout_correction):
    # The feature follows rollout.logprobs_mode instead of forcing it, so the
    # sampled rollout_log_probs that rollout correction consumes are unchanged.
    actor = SimpleNamespace(use_dynamic_bsz=True, use_kl_loss=False, validate=lambda *_args: None)
    rollout = _AttrDict(
        selected_token_logprobs=_AttrDict(token_ids=[7]),
        calculate_log_probs=True,
        val_kwargs=SimpleNamespace(do_sample=False),
        name="vllm",
    )
    config = SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
        actor_rollout_ref=SimpleNamespace(actor=actor, rollout=rollout, model=_AttrDict(lora={}, lora_rank=0)),
        algorithm=_AttrDict(rollout_correction=_AttrDict(rollout_correction), use_kl_in_reward=False),
        data=_AttrDict(train_batch_size=1, val_batch_size=None),
        reward=_AttrDict(
            num_workers=1,
            reward_model=_AttrDict(enable=False, enable_resource_pool=False),
            custom_reward_function=_AttrDict(path=None),
        ),
    )

    validate_config(config, use_reference_policy=False, use_critic=False)
