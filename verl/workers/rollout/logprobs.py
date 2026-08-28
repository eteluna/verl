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

"""Typed rollout payload for log probabilities of explicitly requested tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator

SPECIFIED_TOKEN_LOGPROBS_KEY = "specified_token_logprobs"
SPECIFIED_TOKEN_LOGPROBS_MODE = "raw_logprobs"
SPECIFIED_TOKEN_LOGPROBS_NORMALIZATION = "full_vocab_log_softmax"


def _copy_readonly_array(value: object, *, dtype: np.dtype, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field_name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != dtype:
        raise ValueError(f"{field_name} must have dtype {dtype}, got {value.dtype}")
    copied = np.array(value, dtype=dtype, order="C", copy=True)
    copied.setflags(write=False)
    return copied


def _validate_configured_positions(configured_positions: Sequence[int]) -> tuple[int, ...]:
    if isinstance(configured_positions, str | bytes) or not isinstance(configured_positions, Sequence):
        raise TypeError("configured_positions must be a sequence of Python integers")

    positions = tuple(configured_positions)
    for position in positions:
        if type(position) is not int:
            raise TypeError(f"configured_positions must contain only Python integers; got {type(position).__name__}")
        if position < 0:
            raise ValueError(f"configured_positions must be non-negative, got {position}")
    if any(left >= right for left, right in zip(positions, positions[1:], strict=False)):
        raise ValueError("configured_positions must be unique and strictly increasing")
    return positions


def _validate_response_ids(response_ids: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    if isinstance(response_ids, str | bytes) or not isinstance(response_ids, Sequence):
        raise TypeError(f"{field_name} must be a sequence of Python integers")

    normalized = tuple(response_ids)
    for token_id in normalized:
        if type(token_id) is not int:
            raise TypeError(f"{field_name} must contain only Python integers; got {type(token_id).__name__}")
        if token_id < 0:
            raise ValueError(f"{field_name} must contain non-negative token IDs, got {token_id}")
    return normalized


class SpecifiedTokenLogprobs(BaseModel):
    """Sparse response-position rows for caller-specified token IDs.

    ``token_ids`` defines the column order. ``position_indices`` defines the
    response-token rows, and ``logprobs`` has shape ``[P, M]`` for ``P``
    realized positions and ``M`` requested token IDs. A present payload with no
    realized positions is represented by arrays with shapes ``[0]`` and
    ``[0, M]``; ``None`` is reserved for a disabled request.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    schema_version: Literal[1] = 1
    token_ids: tuple[StrictInt, ...]
    response_token_count: StrictInt
    position_indices: np.ndarray
    logprobs: np.ndarray
    normalization: Literal["full_vocab_log_softmax"] = SPECIFIED_TOKEN_LOGPROBS_NORMALIZATION
    mode: Literal["raw_logprobs"] = SPECIFIED_TOKEN_LOGPROBS_MODE
    backend: str
    backend_version: str
    policy_version: StrictInt
    model_revision: Optional[str] = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("position_indices", mode="before")
    @classmethod
    def _copy_position_indices(cls, value: object) -> np.ndarray:
        return _copy_readonly_array(value, dtype=np.dtype(np.int32), field_name="position_indices")

    @field_validator("logprobs", mode="before")
    @classmethod
    def _copy_logprobs(cls, value: object) -> np.ndarray:
        return _copy_readonly_array(value, dtype=np.dtype(np.float32), field_name="logprobs")

    @model_validator(mode="after")
    def _validate_invariants(self) -> SpecifiedTokenLogprobs:
        if not self.token_ids:
            raise ValueError("token_ids must be non-empty")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must be non-negative")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("token_ids must be unique while preserving caller order")
        if self.response_token_count < 0:
            raise ValueError("response_token_count must be non-negative")
        if self.policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.backend_version:
            raise ValueError("backend_version must be non-empty")
        if self.model_revision is not None and not self.model_revision:
            raise ValueError("model_revision must be non-empty when provided")

        if self.position_indices.ndim != 1:
            raise ValueError(f"position_indices must have shape [P], got ndim={self.position_indices.ndim}")
        if self.logprobs.ndim != 2:
            raise ValueError(f"logprobs must have shape [P, M], got ndim={self.logprobs.ndim}")

        expected_shape = (self.position_indices.shape[0], len(self.token_ids))
        if self.logprobs.shape != expected_shape:
            raise ValueError(f"logprobs must have shape {expected_shape}, got {self.logprobs.shape}")
        if np.isnan(self.logprobs).any():
            raise ValueError("logprobs must not contain NaN")
        if np.any(self.logprobs > 0):
            raise ValueError("full-vocabulary-normalized logprobs must be non-positive")

        positions = self.position_indices
        if positions.size:
            if int(positions[0]) < 0:
                raise ValueError("position_indices must be non-negative")
            if np.any(positions[:-1] >= positions[1:]):
                raise ValueError("position_indices must be unique and strictly increasing")
            if int(positions[-1]) >= self.response_token_count:
                raise ValueError(
                    "position_indices must be smaller than response_token_count, "
                    f"got last position {int(positions[-1])} and count {self.response_token_count}"
                )
        return self


def build_specified_token_logprobs(
    dense_logprobs: np.ndarray,
    configured_positions: Sequence[int],
    *,
    token_ids: Sequence[int],
    backend: str,
    backend_version: str,
    policy_version: int,
    model_revision: Optional[str] = None,
) -> SpecifiedTokenLogprobs:
    """Validate a dense backend matrix, then gather configured response rows.

    The full ``[T, M]`` matrix is validated before gathering so missing cells in
    unselected backend rows cannot be hidden. Positions outside the realized
    response length are filtered as Python integers before the remaining values
    are represented as ``int32``.
    """

    if not isinstance(dense_logprobs, np.ndarray):
        raise TypeError(f"dense_logprobs must be a numpy.ndarray, got {type(dense_logprobs).__name__}")
    if dense_logprobs.dtype != np.float32:
        raise ValueError(f"dense_logprobs must have dtype float32, got {dense_logprobs.dtype}")
    if dense_logprobs.ndim != 2:
        raise ValueError(f"dense_logprobs must have shape [T, M], got ndim={dense_logprobs.ndim}")
    if np.isnan(dense_logprobs).any():
        raise ValueError("dense_logprobs must not contain NaN")
    if np.any(dense_logprobs > 0):
        raise ValueError("full-vocabulary-normalized dense_logprobs must be non-positive")

    positions = _validate_configured_positions(configured_positions)
    response_token_count, num_requested_tokens = dense_logprobs.shape
    if num_requested_tokens != len(token_ids):
        raise ValueError(
            f"dense_logprobs column count must match token_ids, got {num_requested_tokens} and {len(token_ids)}"
        )

    realized_positions = tuple(position for position in positions if position < response_token_count)
    if realized_positions and realized_positions[-1] > np.iinfo(np.int32).max:
        raise ValueError(f"realized response position exceeds int32 range: {realized_positions[-1]}")

    position_indices = np.asarray(realized_positions, dtype=np.int32)
    if realized_positions:
        gathered_logprobs = dense_logprobs[list(realized_positions), :]
    else:
        gathered_logprobs = np.empty((0, num_requested_tokens), dtype=np.float32)

    return SpecifiedTokenLogprobs(
        token_ids=tuple(token_ids),
        response_token_count=response_token_count,
        position_indices=position_indices,
        logprobs=np.asarray(gathered_logprobs, dtype=np.float32),
        backend=backend,
        backend_version=backend_version,
        policy_version=policy_version,
        model_revision=model_revision,
    )


def finalize_specified_token_logprobs(
    payload: SpecifiedTokenLogprobs,
    configured_positions: Sequence[int],
    configured_token_ids: Sequence[int],
    backend_response_ids: Sequence[int],
    final_response_ids: Sequence[int],
) -> SpecifiedTokenLogprobs:
    """Align a backend payload to the final response-token prefix."""

    positions = _validate_configured_positions(configured_positions)
    token_ids = _validate_response_ids(configured_token_ids, field_name="configured_token_ids")
    backend_ids = _validate_response_ids(backend_response_ids, field_name="backend_response_ids")
    final_ids = _validate_response_ids(final_response_ids, field_name="final_response_ids")

    if payload.token_ids != token_ids:
        raise ValueError(
            "payload token_ids must exactly match configured_token_ids in caller order, "
            f"got {payload.token_ids} and {token_ids}"
        )
    if payload.response_token_count != len(backend_ids):
        raise ValueError(
            "payload response_token_count must match backend_response_ids length, "
            f"got {payload.response_token_count} and {len(backend_ids)}"
        )
    if len(final_ids) > len(backend_ids) or backend_ids[: len(final_ids)] != final_ids:
        raise ValueError("final_response_ids must be an exact prefix of backend_response_ids")

    expected_backend_positions = tuple(position for position in positions if position < len(backend_ids))
    actual_positions = tuple(int(position) for position in payload.position_indices)
    missing_backend_positions = [
        position for position in expected_backend_positions if position not in actual_positions
    ]
    if missing_backend_positions:
        raise ValueError(f"payload is missing configured in-range backend positions: {missing_backend_positions}")
    if actual_positions != expected_backend_positions:
        raise ValueError(
            "payload position_indices must exactly match configured positions realized by the backend, "
            f"got {actual_positions} and {expected_backend_positions}"
        )

    expected_final_positions = tuple(position for position in positions if position < len(final_ids))
    missing_final_positions = [position for position in expected_final_positions if position not in actual_positions]
    if missing_final_positions:
        raise ValueError(f"payload is missing configured in-range final positions: {missing_final_positions}")

    row_by_position = {position: row for row, position in enumerate(actual_positions)}
    selected_rows = [row_by_position[position] for position in expected_final_positions]
    if selected_rows:
        final_logprobs = payload.logprobs[selected_rows, :]
    else:
        final_logprobs = np.empty((0, len(payload.token_ids)), dtype=np.float32)

    return SpecifiedTokenLogprobs(
        schema_version=payload.schema_version,
        token_ids=payload.token_ids,
        response_token_count=len(final_ids),
        position_indices=np.asarray(expected_final_positions, dtype=np.int32),
        logprobs=np.asarray(final_logprobs, dtype=np.float32),
        normalization=payload.normalization,
        mode=payload.mode,
        backend=payload.backend,
        backend_version=payload.backend_version,
        policy_version=payload.policy_version,
        model_revision=payload.model_revision,
    )
