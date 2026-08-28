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
from verl.workers.config import SpecifiedTokenLogprobsConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.logprobs import (
    SPECIFIED_TOKEN_LOGPROBS_KEY,
    SpecifiedTokenLogprobs,
    build_specified_token_logprobs,
    finalize_specified_token_logprobs,
)


def _payload(
    *,
    response_token_count: int = 3,
    position_indices: np.ndarray | None = None,
    logprobs: np.ndarray | None = None,
) -> SpecifiedTokenLogprobs:
    if position_indices is None:
        position_indices = np.array([0, 2], dtype=np.int32)
    if logprobs is None:
        logprobs = np.array([[-0.1, -1.1], [-0.2, -1.2]], dtype=np.float32)
    return SpecifiedTokenLogprobs(
        token_ids=(101, 7),
        response_token_count=response_token_count,
        position_indices=position_indices,
        logprobs=logprobs,
        backend="vllm",
        backend_version="0.24.0",
        policy_version=4,
        model_revision="revision-a",
    )


def _specified_config(**overrides) -> SpecifiedTokenLogprobsConfig:
    values = {
        "token_ids": [101, 7],
        "positions": [0, 2],
        "consumers": ["reward"],
        "max_capture_positions": 8,
        "max_requested_positions": 4,
        "max_payload_bytes_per_sample": 1024,
    }
    values.update(overrides)
    return SpecifiedTokenLogprobsConfig(**values)


def _rollout_config(**overrides) -> RolloutConfig:
    values = {
        "name": "vllm",
        "response_length": 8,
        "logprobs_mode": "raw_logprobs",
        "specified_token_logprobs": _specified_config(),
    }
    values.update(overrides)
    return RolloutConfig(**values)


def test_payload_key_is_stable():
    assert SPECIFIED_TOKEN_LOGPROBS_KEY == "specified_token_logprobs"


def test_payload_preserves_caller_order_and_owns_readonly_contiguous_arrays():
    source_positions = np.arange(4, dtype=np.int32)[::2]
    source_logprobs = np.asfortranarray(np.array([[-0.1, -1.1], [-0.2, -1.2]], dtype=np.float32))

    payload = _payload(position_indices=source_positions, logprobs=source_logprobs)

    assert payload.token_ids == (101, 7)
    assert payload.position_indices.flags.c_contiguous
    assert payload.logprobs.flags.c_contiguous
    assert not payload.position_indices.flags.writeable
    assert not payload.logprobs.flags.writeable
    assert not np.shares_memory(payload.position_indices, source_positions)
    assert not np.shares_memory(payload.logprobs, source_logprobs)

    source_positions[0] = 1
    source_logprobs[0, 0] = -99.0
    assert payload.position_indices.tolist() == [0, 2]
    assert payload.logprobs[0, 0] == pytest.approx(-0.1)


def test_payload_represents_present_but_empty_positions_with_typed_arrays():
    payload = _payload(
        response_token_count=0,
        position_indices=np.empty((0,), dtype=np.int32),
        logprobs=np.empty((0, 2), dtype=np.float32),
    )

    assert payload.position_indices.shape == (0,)
    assert payload.position_indices.dtype == np.int32
    assert payload.logprobs.shape == (0, 2)
    assert payload.logprobs.dtype == np.float32


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("position_indices", np.array([0, 2], dtype=np.int64), "dtype int32"),
        ("logprobs", np.zeros((2, 2), dtype=np.float64), "dtype float32"),
        ("position_indices", np.zeros((1, 2), dtype=np.int32), "shape \\[P\\]"),
        ("logprobs", np.zeros((2,), dtype=np.float32), "shape \\[P, M\\]"),
        ("position_indices", np.array([0, 0], dtype=np.int32), "strictly increasing"),
        ("position_indices", np.array([0, 3], dtype=np.int32), "response_token_count"),
        (
            "logprobs",
            np.array([[-0.1, np.nan], [-0.2, -1.2]], dtype=np.float32),
            "must not contain NaN",
        ),
        (
            "logprobs",
            np.array([[-0.1, np.inf], [-0.2, -1.2]], dtype=np.float32),
            "must be non-positive",
        ),
        (
            "logprobs",
            np.array([[-0.1, 0.01], [-0.2, -1.2]], dtype=np.float32),
            "must be non-positive",
        ),
    ],
)
def test_payload_rejects_invalid_arrays(field, value, match):
    kwargs = {field: value}
    with pytest.raises(ValidationError, match=match):
        _payload(**kwargs)


def test_payload_rejects_matrix_shape_mismatch():
    with pytest.raises(ValidationError, match="logprobs must have shape"):
        _payload(logprobs=np.zeros((1, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": True},
        {"token_ids": (True, 7)},
        {"response_token_count": True},
        {"policy_version": True},
    ],
)
def test_payload_rejects_boolean_strict_integers(overrides):
    values = {
        "token_ids": (101, 7),
        "response_token_count": 3,
        "position_indices": np.array([0, 2], dtype=np.int32),
        "logprobs": np.zeros((2, 2), dtype=np.float32),
        "backend": "vllm",
        "backend_version": "0.24.0",
        "policy_version": 0,
    }
    values.update(overrides)
    with pytest.raises(ValidationError):
        SpecifiedTokenLogprobs(**values)


def test_build_validates_dense_matrix_then_gathers_realized_positions():
    dense = np.array(
        [[-0.1, -1.1], [-0.2, -1.2], [-0.3, -1.3], [-0.4, -1.4]],
        dtype=np.float32,
    )

    payload = build_specified_token_logprobs(
        dense,
        [0, 2, 5],
        token_ids=[101, 7],
        backend="vllm",
        backend_version="0.24.0",
        policy_version=2,
    )

    assert payload.response_token_count == 4
    assert payload.position_indices.tolist() == [0, 2]
    np.testing.assert_array_equal(payload.logprobs, dense[[0, 2]])


def test_build_validates_unselected_dense_rows_before_gathering():
    dense = np.array([[-0.1, -1.1], [np.nan, -1.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="dense_logprobs must not contain NaN"):
        build_specified_token_logprobs(
            dense,
            [0],
            token_ids=[101, 7],
            backend="vllm",
            backend_version="0.24.0",
            policy_version=0,
        )


def test_build_rejects_positive_logprob_in_unselected_dense_row():
    dense = np.array([[-0.1, -1.1], [0.01, -1.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="dense_logprobs must be non-positive"):
        build_specified_token_logprobs(
            dense,
            [0],
            token_ids=[101, 7],
            backend="vllm",
            backend_version="0.24.0",
            policy_version=0,
        )


def test_build_filters_python_positions_before_int32_cast():
    dense = np.array([[-0.1]], dtype=np.float32)

    payload = build_specified_token_logprobs(
        dense,
        [0, 2**40],
        token_ids=[101],
        backend="vllm",
        backend_version="0.24.0",
        policy_version=0,
    )

    assert payload.position_indices.tolist() == [0]


def test_build_returns_typed_empty_when_response_is_shorter_than_every_position():
    payload = build_specified_token_logprobs(
        np.empty((0, 2), dtype=np.float32),
        [3, 7],
        token_ids=[101, 7],
        backend="vllm",
        backend_version="0.24.0",
        policy_version=0,
    )

    assert payload.response_token_count == 0
    assert payload.position_indices.shape == (0,)
    assert payload.logprobs.shape == (0, 2)


def test_finalize_filters_rows_to_exact_final_response_prefix():
    dense = np.array(
        [[-0.1, -1.1], [-0.2, -1.2], [-0.3, -1.3], [-0.4, -1.4]],
        dtype=np.float32,
    )
    payload = build_specified_token_logprobs(
        dense,
        [0, 2, 5],
        token_ids=[101, 7],
        backend="vllm",
        backend_version="0.24.0",
        policy_version=2,
    )

    finalized = finalize_specified_token_logprobs(
        payload,
        configured_positions=[0, 2, 5],
        configured_token_ids=[101, 7],
        backend_response_ids=[11, 12, 13, 14],
        final_response_ids=[11, 12],
    )

    assert finalized.response_token_count == 2
    assert finalized.position_indices.tolist() == [0]
    np.testing.assert_array_equal(finalized.logprobs, dense[[0]])
    assert not finalized.position_indices.flags.writeable
    assert not finalized.logprobs.flags.writeable


def test_finalize_rejects_non_prefix_final_response():
    with pytest.raises(ValueError, match="exact prefix"):
        finalize_specified_token_logprobs(
            _payload(),
            configured_positions=[0, 2],
            configured_token_ids=[101, 7],
            backend_response_ids=[11, 12, 13],
            final_response_ids=[11, 99],
        )


def test_finalize_rejects_payload_token_ids_that_do_not_match_configured_order():
    with pytest.raises(ValueError, match="exactly match configured_token_ids"):
        finalize_specified_token_logprobs(
            _payload(),
            configured_positions=[0, 2],
            configured_token_ids=[7, 101],
            backend_response_ids=[11, 12, 13],
            final_response_ids=[11, 12, 13],
        )


def test_finalize_rejects_missing_configured_in_range_position():
    incomplete = _payload(
        position_indices=np.array([0], dtype=np.int32),
        logprobs=np.array([[-0.1, -1.1]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="missing configured in-range"):
        finalize_specified_token_logprobs(
            incomplete,
            configured_positions=[0, 2],
            configured_token_ids=[101, 7],
            backend_response_ids=[11, 12, 13],
            final_response_ids=[11, 12, 13],
        )


def test_specified_config_defaults_to_disabled():
    config = SpecifiedTokenLogprobsConfig()

    assert config.enabled is False
    assert config.token_ids is None
    assert config.positions is None
    assert config.consumers == ["reward"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_ids": [101], "positions": None},
        {"token_ids": None, "positions": [0]},
    ],
)
def test_specified_config_requires_token_ids_and_positions_together(kwargs):
    with pytest.raises(ValueError, match="must be set together"):
        SpecifiedTokenLogprobsConfig(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"token_ids": []}, "non-empty"),
        ({"token_ids": [True]}, "Python integers"),
        ({"token_ids": [-1]}, "non-negative"),
        ({"token_ids": [7, 7]}, "unique"),
        ({"token_ids": list(range(129))}, "exact-token limit"),
        ({"positions": []}, "non-empty"),
        ({"positions": [True]}, "Python integers"),
        ({"positions": [-1]}, "non-negative"),
        ({"positions": [2, 1]}, "strictly increasing"),
        ({"positions": [1, 1]}, "strictly increasing"),
        ({"positions": [0, 1, 2], "max_requested_positions": 2}, "max_requested_positions"),
    ],
)
def test_specified_config_rejects_invalid_token_ids_and_positions(overrides, match):
    with pytest.raises(ValueError, match=match):
        _specified_config(**overrides)


def test_specified_config_preserves_unsorted_caller_token_order():
    config = _specified_config(token_ids=[9, 3, 7])

    assert config.token_ids == [9, 3, 7]


def test_specified_config_registry_fails_closed_for_unknown_or_unimplemented_consumer():
    with pytest.raises(ValueError, match="unknown specified_token_logprobs consumers"):
        _specified_config(consumers=["unknown"])
    with pytest.raises(NotImplementedError, match="actor_auxiliary"):
        _specified_config(consumers=["actor_auxiliary"])


@pytest.mark.parametrize(
    "field_name",
    ["max_capture_positions", "max_requested_positions", "max_payload_bytes_per_sample"],
)
@pytest.mark.parametrize("value", [0, -1, True])
def test_specified_config_limits_are_positive_python_integers(field_name, value):
    with pytest.raises(ValueError, match="positive Python integer"):
        _specified_config(**{field_name: value})


def test_rollout_config_accepts_dataclass_dict_and_dictconfig():
    dataclass_config = _specified_config()
    from_dataclass = _rollout_config(specified_token_logprobs=dataclass_config)
    from_dict = _rollout_config(
        specified_token_logprobs={
            "token_ids": [101, 7],
            "positions": [0, 2],
            "max_capture_positions": 8,
        }
    )
    from_dictconfig = _rollout_config(
        specified_token_logprobs=OmegaConf.create(
            {
                "token_ids": [101, 7],
                "positions": [0, 2],
                "max_capture_positions": 8,
            }
        )
    )

    assert from_dataclass.specified_token_logprobs is dataclass_config
    assert isinstance(from_dict.specified_token_logprobs, SpecifiedTokenLogprobsConfig)
    assert isinstance(from_dictconfig.specified_token_logprobs, SpecifiedTokenLogprobsConfig)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"name": "sglang"}, ValueError, "only by rollout.name='vllm'"),
        ({"logprobs_mode": "processed_logprobs"}, ValueError, "requires rollout.logprobs_mode"),
        ({"multi_turn": {"enable": True}}, NotImplementedError, "multi-turn"),
        ({"agent": {"default_agent_loop": "tool_agent"}}, NotImplementedError, "single_turn_agent"),
        (
            {"agent": {"agent_loop_config_path": "custom-agent.yaml"}},
            NotImplementedError,
            "agent_loop_config_path",
        ),
        (
            {"agent": {"agent_loop_manager_class": "custom.Manager"}},
            NotImplementedError,
            "agent_loop_manager_class",
        ),
        ({"mtp": {"enable_rollout": True}}, NotImplementedError, "speculative decoding"),
        ({"disaggregation": {"enabled": True}}, NotImplementedError, "disaggregation"),
        ({"trace": {"backend": "mlflow"}}, NotImplementedError, "rollout tracing"),
        ({"response_length": 9}, ValueError, "max_capture_positions"),
    ],
)
def test_rollout_config_feature_gates(overrides, error, match):
    with pytest.raises(error, match=match):
        _rollout_config(**overrides)


def test_rollout_config_rejects_position_outside_configured_response_length():
    with pytest.raises(ValueError, match="smaller than rollout.response_length"):
        _rollout_config(
            response_length=8,
            specified_token_logprobs=_specified_config(positions=[0, 8]),
        )


def test_rollout_config_rejects_sparse_ndarray_payload_over_byte_limit():
    # P=2 and M=2 owns int32[P] + float32[P,M] = 24 bytes.
    with pytest.raises(ValueError, match="sparse ndarray payload"):
        _rollout_config(
            specified_token_logprobs=_specified_config(max_payload_bytes_per_sample=23),
        )


def test_canonical_rollout_yaml_exposes_disabled_typed_config():
    config = OmegaConf.load("verl/trainer/config/rollout/rollout.yaml").specified_token_logprobs

    assert config._target_ == "verl.workers.config.SpecifiedTokenLogprobsConfig"
    assert config.token_ids is None
    assert config.positions is None
    assert config.consumers == ["reward"]


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _invalid_reward_topology_config(
    *, reward_model_enabled: bool, enable_resource_pool: bool, custom_path, num_workers: int = 1
):
    return SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
        actor_rollout_ref=SimpleNamespace(rollout=_AttrDict(specified_token_logprobs=_AttrDict(token_ids=[101]))),
        algorithm=_AttrDict(rollout_correction=None),
        reward=_AttrDict(
            num_workers=num_workers,
            reward_model=_AttrDict(
                enable=reward_model_enabled,
                enable_resource_pool=enable_resource_pool,
            ),
            custom_reward_function=_AttrDict(path=custom_path),
        ),
    )


def test_validate_config_rejects_colocated_reward_model_before_rollout():
    config = _invalid_reward_topology_config(
        reward_model_enabled=True,
        enable_resource_pool=False,
        custom_path="reward.py",
    )

    with pytest.raises(NotImplementedError, match="require streaming reward workers"):
        validate_config(config, use_reference_policy=False, use_critic=False)


def test_validate_config_rejects_builtin_discriminative_reward_path_before_rollout():
    config = _invalid_reward_topology_config(
        reward_model_enabled=True,
        enable_resource_pool=True,
        custom_path=None,
    )

    with pytest.raises(NotImplementedError, match="built-in discriminative reward model path"):
        validate_config(config, use_reference_policy=False, use_critic=False)


def test_validate_config_requires_a_streaming_reward_worker():
    config = _invalid_reward_topology_config(
        reward_model_enabled=False,
        enable_resource_pool=False,
        custom_path=None,
        num_workers=0,
    )

    with pytest.raises(ValueError, match="reward.num_workers to be a positive Python integer"):
        validate_config(config, use_reference_policy=False, use_critic=False)
