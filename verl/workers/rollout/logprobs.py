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

"""Typed rollout payload for log probabilities of caller-selected token IDs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator

SELECTED_TOKEN_LOGPROBS_KEY = "selected_token_logprobs"


def _copy_readonly_array(value: object, *, dtype: np.dtype, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field_name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != dtype:
        raise ValueError(f"{field_name} must have dtype {dtype}, got {value.dtype}")
    copied = np.array(value, dtype=dtype, order="C", copy=True)
    copied.setflags(write=False)
    return copied


def _validate_configured_positions(configured_positions: Optional[Sequence[int]]) -> Optional[tuple[int, ...]]:
    """``None`` means every response position; otherwise a strictly increasing tuple."""
    if configured_positions is None:
        return None
    if isinstance(configured_positions, str | bytes) or not isinstance(configured_positions, Sequence):
        raise TypeError("configured_positions must be None or a sequence of Python integers")

    positions = tuple(configured_positions)
    for position in positions:
        if type(position) is not int:
            raise TypeError(f"configured_positions must contain only Python integers; got {type(position).__name__}")
        if position < 0:
            raise ValueError(f"configured_positions must be non-negative, got {position}")
    if any(left >= right for left, right in zip(positions, positions[1:], strict=False)):
        raise ValueError("configured_positions must be unique and strictly increasing")
    return positions


def _validate_token_ids(token_ids: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    if isinstance(token_ids, str | bytes) or not isinstance(token_ids, Sequence):
        raise TypeError(f"{field_name} must be a sequence of Python integers")

    normalized = tuple(token_ids)
    for token_id in normalized:
        if type(token_id) is not int:
            raise TypeError(f"{field_name} must contain only Python integers; got {type(token_id).__name__}")
        if token_id < 0:
            raise ValueError(f"{field_name} must contain non-negative token IDs, got {token_id}")
    return normalized


def _values_are_logprobs(logprobs_mode: str) -> bool:
    """vLLM ``*_logprobs`` modes emit log-softmax values; ``*_logits`` modes emit raw scores."""
    return logprobs_mode.endswith("logprobs")


def _expected_positions(positions: Optional[tuple[int, ...]], response_token_count: int) -> tuple[int, ...]:
    if positions is None:
        return tuple(range(response_token_count))
    return tuple(position for position in positions if position < response_token_count)


class SelectedTokenLogprobs(BaseModel):
    """Log probabilities of caller-selected token IDs at response positions.

    ``token_ids`` defines the column order and ``response_ids`` the response
    the rows describe. ``positions`` holds the zero-based response positions
    of the rows (every position when the request was dense) and ``logprobs``
    has shape ``[P, M]``. ``logprobs_mode`` records the engine mode the values
    were produced under; ``*_logits`` modes are carried unchanged. A present
    payload with no rows uses shapes ``[0]`` and ``[0, M]``; ``None`` is
    reserved for a disabled request.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    schema_version: Literal[1] = 1
    token_ids: tuple[StrictInt, ...]
    response_ids: tuple[StrictInt, ...]
    positions: np.ndarray
    logprobs: np.ndarray
    logprobs_mode: str
    backend: str
    backend_version: str

    @property
    def response_token_count(self) -> int:
        return len(self.response_ids)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("positions", mode="before")
    @classmethod
    def _copy_positions(cls, value: object) -> np.ndarray:
        return _copy_readonly_array(value, dtype=np.dtype(np.int32), field_name="positions")

    @field_validator("logprobs", mode="before")
    @classmethod
    def _copy_logprobs(cls, value: object) -> np.ndarray:
        return _copy_readonly_array(value, dtype=np.dtype(np.float32), field_name="logprobs")

    @model_validator(mode="after")
    def _validate_invariants(self) -> SelectedTokenLogprobs:
        if not self.token_ids:
            raise ValueError("token_ids must be non-empty")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must be non-negative")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("token_ids must be unique while preserving caller order")
        if any(token_id < 0 for token_id in self.response_ids):
            raise ValueError("response_ids must be non-negative")
        if not self.logprobs_mode:
            raise ValueError("logprobs_mode must be non-empty")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.backend_version:
            raise ValueError("backend_version must be non-empty")

        if self.positions.ndim != 1:
            raise ValueError(f"positions must have shape [P], got ndim={self.positions.ndim}")
        if self.logprobs.ndim != 2:
            raise ValueError(f"logprobs must have shape [P, M], got ndim={self.logprobs.ndim}")

        expected_shape = (self.positions.shape[0], len(self.token_ids))
        if self.logprobs.shape != expected_shape:
            raise ValueError(f"logprobs must have shape {expected_shape}, got {self.logprobs.shape}")
        if np.isnan(self.logprobs).any():
            raise ValueError("logprobs must not contain NaN")
        if _values_are_logprobs(self.logprobs_mode) and np.any(self.logprobs > 0):
            raise ValueError(f"logprobs must be non-positive under logprobs_mode={self.logprobs_mode!r}")

        positions = self.positions
        if positions.size:
            if int(positions[0]) < 0:
                raise ValueError("positions must be non-negative")
            if np.any(positions[:-1] >= positions[1:]):
                raise ValueError("positions must be unique and strictly increasing")
            if int(positions[-1]) >= self.response_token_count:
                raise ValueError(
                    "positions must be smaller than the response length, "
                    f"got last position {int(positions[-1])} and {self.response_token_count} response tokens"
                )
        return self


def build_selected_token_logprobs(
    dense_logprobs: np.ndarray,
    configured_positions: Optional[Sequence[int]],
    *,
    token_ids: Sequence[int],
    response_ids: Sequence[int],
    logprobs_mode: str,
    backend: str,
    backend_version: str,
) -> SelectedTokenLogprobs:
    """Validate a dense ``[T, M]`` backend matrix, then keep the configured rows.

    ``configured_positions=None`` keeps every row. The full matrix is validated
    before gathering so a bad cell in an unselected row cannot be hidden.
    Positions beyond the realized response length are dropped as Python
    integers before the remaining values are represented as ``int32``.
    """

    if not isinstance(dense_logprobs, np.ndarray):
        raise TypeError(f"dense_logprobs must be a numpy.ndarray, got {type(dense_logprobs).__name__}")
    if dense_logprobs.dtype != np.float32:
        raise ValueError(f"dense_logprobs must have dtype float32, got {dense_logprobs.dtype}")
    if dense_logprobs.ndim != 2:
        raise ValueError(f"dense_logprobs must have shape [T, M], got ndim={dense_logprobs.ndim}")
    if np.isnan(dense_logprobs).any():
        raise ValueError("dense_logprobs must not contain NaN")
    if _values_are_logprobs(logprobs_mode) and np.any(dense_logprobs > 0):
        raise ValueError(f"dense_logprobs must be non-positive under logprobs_mode={logprobs_mode!r}")

    positions = _validate_configured_positions(configured_positions)
    response_ids = _validate_token_ids(response_ids, field_name="response_ids")
    response_token_count, num_requested_tokens = dense_logprobs.shape
    if num_requested_tokens != len(token_ids):
        raise ValueError(
            f"dense_logprobs column count must match token_ids, got {num_requested_tokens} and {len(token_ids)}"
        )
    if response_token_count != len(response_ids):
        raise ValueError(
            f"dense_logprobs row count must match response_ids, got {response_token_count} and {len(response_ids)}"
        )

    kept_positions = _expected_positions(positions, response_token_count)
    if kept_positions and kept_positions[-1] > np.iinfo(np.int32).max:
        raise ValueError(f"realized response position exceeds int32 range: {kept_positions[-1]}")

    if positions is None:
        kept_logprobs = dense_logprobs
    elif kept_positions:
        kept_logprobs = dense_logprobs[list(kept_positions), :]
    else:
        kept_logprobs = np.empty((0, num_requested_tokens), dtype=np.float32)

    return SelectedTokenLogprobs(
        token_ids=tuple(token_ids),
        response_ids=response_ids,
        positions=np.asarray(kept_positions, dtype=np.int32),
        logprobs=np.asarray(kept_logprobs, dtype=np.float32),
        logprobs_mode=logprobs_mode,
        backend=backend,
        backend_version=backend_version,
    )


def finalize_selected_token_logprobs(
    payload: SelectedTokenLogprobs,
    *,
    configured_positions: Optional[Sequence[int]],
    configured_token_ids: Sequence[int],
    final_response_ids: Sequence[int],
) -> SelectedTokenLogprobs:
    """Align a backend payload to the final response, which may be a truncated prefix."""

    positions = _validate_configured_positions(configured_positions)
    token_ids = _validate_token_ids(configured_token_ids, field_name="configured_token_ids")
    final_ids = _validate_token_ids(final_response_ids, field_name="final_response_ids")

    if payload.token_ids != token_ids:
        raise ValueError(
            "payload token_ids must exactly match configured_token_ids in caller order, "
            f"got {payload.token_ids} and {token_ids}"
        )
    backend_ids = payload.response_ids
    if len(final_ids) > len(backend_ids) or backend_ids[: len(final_ids)] != final_ids:
        raise ValueError("final_response_ids must be an exact prefix of the payload response_ids")

    expected_backend_positions = _expected_positions(positions, len(backend_ids))
    actual_positions = tuple(int(position) for position in payload.positions)
    missing_backend_positions = sorted(set(expected_backend_positions) - set(actual_positions))
    if missing_backend_positions:
        raise ValueError(f"payload is missing configured in-range backend positions: {missing_backend_positions}")
    if actual_positions != expected_backend_positions:
        raise ValueError(
            "payload positions must exactly match the configured positions realized by the backend, "
            f"got {actual_positions} and {expected_backend_positions}"
        )

    expected_final_positions = _expected_positions(positions, len(final_ids))
    row_by_position = {position: row for row, position in enumerate(actual_positions)}
    selected_rows = [row_by_position[position] for position in expected_final_positions]
    if selected_rows:
        final_logprobs = payload.logprobs[selected_rows, :]
    else:
        final_logprobs = np.empty((0, len(payload.token_ids)), dtype=np.float32)

    return SelectedTokenLogprobs(
        schema_version=payload.schema_version,
        token_ids=payload.token_ids,
        response_ids=final_ids,
        positions=np.asarray(expected_final_positions, dtype=np.int32),
        logprobs=np.asarray(final_logprobs, dtype=np.float32),
        logprobs_mode=payload.logprobs_mode,
        backend=payload.backend,
        backend_version=payload.backend_version,
    )
