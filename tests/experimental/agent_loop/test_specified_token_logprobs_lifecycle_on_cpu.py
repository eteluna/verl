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
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.logprobs import SPECIFIED_TOKEN_LOGPROBS_KEY, SpecifiedTokenLogprobs
from verl.workers.rollout.replica import TokenOutput


def _payload(
    *,
    response_token_count: int,
    positions: tuple[int, ...],
    rows: tuple[tuple[float, ...], ...],
    token_ids: tuple[int, ...] = (101, 202),
) -> SpecifiedTokenLogprobs:
    if rows:
        logprobs = np.asarray(rows, dtype=np.float32)
    else:
        logprobs = np.empty((0, len(token_ids)), dtype=np.float32)
    return SpecifiedTokenLogprobs(
        token_ids=token_ids,
        response_token_count=response_token_count,
        position_indices=np.asarray(positions, dtype=np.int32),
        logprobs=logprobs,
        backend="vllm",
        backend_version="test",
        policy_version=3,
    )


class _FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools=None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        return [1, 2]


class _FakeServerManager:
    def __init__(self, output: TokenOutput):
        self.output = output

    async def generate(self, *args, **kwargs) -> TokenOutput:
        del args, kwargs
        return self.output


def _single_turn_loop(
    token_output: TokenOutput,
    *,
    response_length: int,
    configured_positions: tuple[int, ...] | None,
) -> SingleTurnAgentLoop:
    if configured_positions is None:
        specified_token_config = {"token_ids": None, "positions": None}
    else:
        assert token_output.specified_token_logprobs is not None
        specified_token_config = {
            "token_ids": list(token_output.specified_token_logprobs.token_ids),
            "positions": list(configured_positions),
        }
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 8,
                    "response_length": response_length,
                    "specified_token_logprobs": specified_token_config,
                    "multi_turn": {"tool_config_path": None},
                },
                "model": {"path": "dummy-model", "tokenizer_path": "dummy-model"},
            },
            "data": {"apply_chat_template_kwargs": {}, "tool_config_path": None},
        }
    )
    return SingleTurnAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=_FakeServerManager(token_output),
        tokenizer=_FakeTokenizer(),
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
    )


async def _run_single_turn(loop: SingleTurnAgentLoop) -> AgentLoopOutput:
    return await loop.run(
        sampling_params={},
        raw_prompt=[{"role": "user", "content": "hello"}],
    )


def _agent_output(
    specified_token_logprobs: SpecifiedTokenLogprobs,
    *,
    reward_score: float | None = None,
) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[10, 11, 12],
        response_mask=[1, 1, 1],
        specified_token_logprobs=specified_token_logprobs,
        reward_score=reward_score,
        metrics=AgentLoopMetrics(),
        extra_fields={"kept": "value"},
        mm_processor_kwargs={},
    )


def test_agent_loop_output_as_dict_excludes_raw_specified_token_logprobs():
    payload = _payload(response_token_count=1, positions=(0,), rows=((-0.1, -0.2),))
    output = _agent_output(payload)

    fields = output.as_dict()

    assert SPECIFIED_TOKEN_LOGPROBS_KEY not in fields
    assert output.specified_token_logprobs is payload


@pytest.mark.asyncio
async def test_single_turn_disabled_path_does_not_set_payload_field():
    token_output = TokenOutput(token_ids=[10, 11], extra_fields={})
    loop = _single_turn_loop(token_output, response_length=4, configured_positions=None)

    output = await _run_single_turn(loop)

    assert output.response_ids == [10, 11]
    assert output.specified_token_logprobs is None
    assert SPECIFIED_TOKEN_LOGPROBS_KEY not in output.model_fields_set


@pytest.mark.asyncio
async def test_single_turn_finalizes_specified_token_logprobs_after_response_truncation():
    token_output = TokenOutput(
        token_ids=[10, 11, 12, 13, 14],
        specified_token_logprobs=_payload(
            response_token_count=5,
            positions=(0, 2),
            rows=((-0.1, -0.2), (-0.3, -0.4)),
        ),
        extra_fields={},
    )
    loop = _single_turn_loop(token_output, response_length=3, configured_positions=(0, 2))

    output = await _run_single_turn(loop)

    assert output.response_ids == [10, 11, 12]
    assert output.specified_token_logprobs is not None
    assert output.specified_token_logprobs.response_token_count == 3
    np.testing.assert_array_equal(output.specified_token_logprobs.position_indices, np.array([0, 2], np.int32))
    np.testing.assert_allclose(
        output.specified_token_logprobs.logprobs,
        np.array([[-0.1, -0.2], [-0.3, -0.4]], np.float32),
    )


@pytest.mark.asyncio
async def test_single_turn_allows_positions_unavailable_in_short_response():
    token_output = TokenOutput(
        token_ids=[10],
        specified_token_logprobs=_payload(
            response_token_count=1,
            positions=(0,),
            rows=((-0.1, -0.2),),
        ),
        extra_fields={},
    )
    loop = _single_turn_loop(token_output, response_length=4, configured_positions=(0, 2))

    output = await _run_single_turn(loop)

    assert output.specified_token_logprobs is not None
    assert output.specified_token_logprobs.response_token_count == 1
    np.testing.assert_array_equal(output.specified_token_logprobs.position_indices, np.array([0], np.int32))


@pytest.mark.asyncio
async def test_single_turn_preserves_present_empty_payload_for_short_response():
    token_output = TokenOutput(
        token_ids=[10],
        specified_token_logprobs=_payload(response_token_count=1, positions=(), rows=()),
        extra_fields={},
    )
    loop = _single_turn_loop(token_output, response_length=4, configured_positions=(2,))

    output = await _run_single_turn(loop)

    assert output.specified_token_logprobs is not None
    assert output.specified_token_logprobs.response_token_count == 1
    assert output.specified_token_logprobs.position_indices.shape == (0,)
    assert output.specified_token_logprobs.logprobs.shape == (0, 2)


@pytest.mark.asyncio
async def test_single_turn_rejects_missing_realized_position():
    token_output = TokenOutput(
        token_ids=[10, 11, 12],
        specified_token_logprobs=_payload(
            response_token_count=3,
            positions=(0,),
            rows=((-0.1, -0.2),),
        ),
        extra_fields={},
    )
    loop = _single_turn_loop(token_output, response_length=4, configured_positions=(0, 2))

    with pytest.raises(ValueError, match="missing configured in-range backend positions"):
        await _run_single_turn(loop)


class _CaptureRemoteMethod:
    def __init__(self):
        self.data = None

    async def remote(self, data):
        self.data = data
        reward_extra_info = data.non_tensor_batch["extra_info"][0]
        reward_extra_info["reward_side_mutation"] = True
        return {
            "reward_score": 1.0,
            "reward_extra_info": {SPECIFIED_TOKEN_LOGPROBS_KEY: "must not escape the consumer"},
        }


class _CaptureRewardHandle:
    def __init__(self):
        self.compute_score = _CaptureRemoteMethod()


def _reward_worker(*, handles, reward_model_enabled: bool = False, custom_reward_path: str | None = None):
    worker = object.__new__(AgentLoopWorker)
    worker.reward_loop_worker_handles = handles
    worker.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": reward_model_enabled},
                "custom_reward_function": {"path": custom_reward_path},
            }
        }
    )
    worker._compute_multi_modal_inputs = lambda output, input_ids: {}
    worker._compute_position_ids = lambda input_ids, attention_mask, multi_modal_inputs, mm_kwargs: torch.arange(
        input_ids.shape[-1]
    ).unsqueeze(0)
    return worker


@pytest.mark.asyncio
async def test_compute_score_hands_off_copy_without_leaking_payload_to_output():
    payload = _payload(response_token_count=3, positions=(0, 2), rows=((-0.1, -0.2), (-0.3, -0.4)))
    output = _agent_output(payload)
    handle = _CaptureRewardHandle()
    worker = _reward_worker(handles=[handle])
    original_extra_info = {"original": "value"}

    await AgentLoopWorker._compute_score(worker, [output], kwargs={"extra_info": original_extra_info})

    captured_extra_info = handle.compute_score.data.non_tensor_batch["extra_info"][0]
    assert captured_extra_info is not original_extra_info
    assert captured_extra_info["original"] == "value"
    assert captured_extra_info["reward_side_mutation"] is True
    assert SPECIFIED_TOKEN_LOGPROBS_KEY in captured_extra_info
    np.testing.assert_array_equal(
        captured_extra_info[SPECIFIED_TOKEN_LOGPROBS_KEY]["position_indices"], np.array([0, 2], np.int32)
    )
    assert original_extra_info == {"original": "value"}
    assert SPECIFIED_TOKEN_LOGPROBS_KEY not in output.extra_fields
    assert output.specified_token_logprobs is None
    assert output.reward_score == 1.0
    assert SPECIFIED_TOKEN_LOGPROBS_KEY not in output.extra_fields["reward_extra_info"]


@pytest.mark.asyncio
async def test_compute_score_rejects_payload_without_streaming_reward_workers():
    payload = _payload(response_token_count=3, positions=(0,), rows=((-0.1, -0.2),))
    worker = _reward_worker(handles=None)

    with pytest.raises(NotImplementedError, match="require streaming reward workers"):
        await AgentLoopWorker._compute_score(worker, [_agent_output(payload)], kwargs={})


@pytest.mark.asyncio
async def test_compute_score_rejects_payload_when_reward_is_already_present():
    payload = _payload(response_token_count=3, positions=(0,), rows=((-0.1, -0.2),))
    worker = _reward_worker(handles=[_CaptureRewardHandle()])

    with pytest.raises(RuntimeError, match="already supplied reward_score"):
        await AgentLoopWorker._compute_score(worker, [_agent_output(payload, reward_score=0.5)], kwargs={})


@pytest.mark.asyncio
async def test_compute_score_rejects_discriminative_reward_model_bypass():
    payload = _payload(response_token_count=3, positions=(0,), rows=((-0.1, -0.2),))
    worker = _reward_worker(handles=[_CaptureRewardHandle()], reward_model_enabled=True)

    with pytest.raises(NotImplementedError, match="built-in discriminative reward model path"):
        await AgentLoopWorker._compute_score(worker, [_agent_output(payload)], kwargs={})


@pytest.mark.asyncio
async def test_agent_loop_rejects_non_single_turn_consumer_path():
    worker = object.__new__(AgentLoopWorker)
    worker.rollout_config = SimpleNamespace(specified_token_logprobs=SimpleNamespace(enabled=True))

    with pytest.raises(NotImplementedError, match="only the single_turn_agent loop"):
        await AgentLoopWorker._run_agent_loop(
            worker,
            {},
            {"step": 0, "sample_index": 0, "rollout_n": 0, "validate": False},
            agent_name="tool_agent",
            trace=False,
        )


@pytest.mark.asyncio
async def test_fully_async_client_rejects_specified_token_logprobs():
    client = object.__new__(FullyAsyncLLMServerClient)
    client.config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"specified_token_logprobs": {"token_ids": [101], "positions": [0]}}}}
    )

    with pytest.raises(NotImplementedError, match="FullyAsync/partial-resume"):
        await client.generate(request_id="request", prompt_ids=[1], sampling_params={})


def test_fully_async_client_rejects_specified_token_logprobs_at_initialization():
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"specified_token_logprobs": {"token_ids": [101], "positions": [0]}}}}
    )

    with pytest.raises(NotImplementedError, match="FullyAsync/partial-resume"):
        FullyAsyncLLMServerClient(config=config)
