# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.logprobs import SPECIFIED_TOKEN_LOGPROBS_KEY, finalize_specified_token_logprobs
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. build the initial prompt with Continuous Token (the only tokenization path).
        # Multimodal inputs require a VL builder + processor; fail loudly otherwise.
        self._assert_mm_supported(bool(multi_modal_data))
        prompt_ids = await self.ct_build_initial_tokens(
            messages,
            images=images,
            videos=videos,
            audios=audios,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            token_output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1

        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            token_output.token_ids,
            [],
            [] if token_output.log_probs else None,
            assistant_logprobs=token_output.log_probs if token_output.log_probs else None,
        )
        response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
        final_response_ids = response_ids[: self.response_length]

        specified_output_fields: dict[str, Any] = {}
        specified_token_config = self.rollout_config.get("specified_token_logprobs")
        if specified_token_config is not None and specified_token_config.get("token_ids") is not None:
            if token_output.specified_token_logprobs is None:
                raise RuntimeError("Rollout backend did not return specified-token logprobs for an enabled request.")
            specified_output_fields[SPECIFIED_TOKEN_LOGPROBS_KEY] = finalize_specified_token_logprobs(
                token_output.specified_token_logprobs,
                configured_positions=specified_token_config.positions,
                configured_token_ids=specified_token_config.token_ids,
                backend_response_ids=token_output.token_ids,
                final_response_ids=final_response_ids,
            )

        agent_output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=final_response_ids,
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                token_output.routed_experts[: len(prompt_ids) + self.response_length]
                if token_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=token_output.extra_fields,
            **specified_output_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        agent_output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return agent_output
