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

import json
from types import SimpleNamespace

import numpy as np
import pytest

# vllm is not part of the `cpu` extra (it conflicts with the cpu torch world), so
# cpu_unit_tests skips this module; vllm.yml runs it in the vllm venv.
pytest.importorskip("vllm")

from verl.utils.config import validate_config
from verl.workers.rollout.vllm_rollout.utils import (
    _resolve_vllm_weight_sync_local_rank,
    build_cli_args_from_config,
    vLLMColocateWorkerExtension,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    _build_vllm_specified_token_logprobs,
    _extract_vllm_logprob_rows,
    _prepare_vllm_logprob_request,
    _require_policy_version,
    _validate_specified_token_max_tokens,
    vLLMHttpServer,
)


class TestBuildCliArgsFromConfig:
    """Tests for CLI argument serialization from config dictionaries."""

    def test_string_value(self):
        """String values become '--key value'."""
        config = {"model": "gpt2"}
        result = build_cli_args_from_config(config)
        assert result == ["--model", "gpt2"]

    def test_integer_value(self):
        """Integer values are converted to strings."""
        config = {"tensor-parallel-size": 4}
        result = build_cli_args_from_config(config)
        assert result == ["--tensor-parallel-size", "4"]

    def test_float_value(self):
        """Float values are converted to strings."""
        config = {"temperature": 0.7}
        result = build_cli_args_from_config(config)
        assert result == ["--temperature", "0.7"]

    def test_bool_true(self):
        """Bool True adds flag without value."""
        config = {"enable-prefix-caching": True}
        result = build_cli_args_from_config(config)
        assert result == ["--enable-prefix-caching"]

    def test_bool_false(self):
        """Optional[bool] args emit '--no-key' for an explicit False."""
        config = {"enable-prefix-caching": False}
        result = build_cli_args_from_config(config)
        assert result == ["--no-enable-prefix-caching"]

    def test_bool_false_plain_bool_omitted(self):
        """Bool False on a plain-bool arg is skipped (parser default is False)."""
        config = {"enforce_eager": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_bool_false_union_str_arg_omitted(self):
        """Bool False on a `bool | str | None` arg is skipped (string flag, no --no- form)."""
        config = {"hf_token": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_bool_false_underscore_key(self):
        """Underscore keys emit the negative flag in the key's own spelling."""
        config = {"enable_prefix_caching": False}
        result = build_cli_args_from_config(config)
        assert result == ["--no-enable_prefix_caching"]

    def test_bool_false_non_engine_arg_omitted(self):
        """Bool False on args unknown to AsyncEngineArgs is omitted."""
        config = {"disable-log-requests": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_none_value(self):
        """None values are skipped."""
        config = {"lora-path": None}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_values(self):
        """List values are expanded into multiple arguments."""
        config = {"cudagraph-capture-sizes": [1, 2, 4, 8]}
        result = build_cli_args_from_config(config)
        assert result == ["--cudagraph-capture-sizes", "1", "2", "4", "8"]

    def test_empty_list(self):
        """Empty lists are skipped (vLLM nargs='+' requires at least one value)."""
        config = {"cudagraph-capture-sizes": []}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_with_strings(self):
        """List of strings is properly expanded."""
        config = {"allowed-origins": ["http://localhost", "http://example.com"]}
        result = build_cli_args_from_config(config)
        assert result == ["--allowed-origins", "http://localhost", "http://example.com"]

    def test_dict_value(self):
        """Dict values are JSON serialized."""
        config = {"extra-config": {"key": "value", "nested": True}}
        result = build_cli_args_from_config(config)
        assert result[0] == "--extra-config"
        # JSON output may have different key ordering, so parse and compare
        assert json.loads(result[1]) == {"key": "value", "nested": True}

    def test_mixed_config(self):
        """Test a realistic mixed configuration."""
        config = {
            "tensor-parallel-size": 4,
            "enable-prefix-caching": True,
            "disable-log-requests": False,
            "lora-path": None,
            "cudagraph-capture-sizes": [1, 2, 4, 8],
            "max-model-len": 2048,
        }
        result = build_cli_args_from_config(config)

        # Check expected args are present
        assert "--tensor-parallel-size" in result
        assert "4" in result
        assert "--enable-prefix-caching" in result
        assert "--cudagraph-capture-sizes" in result
        assert "1" in result
        assert "8" in result
        assert "--max-model-len" in result
        assert "2048" in result

        # Check skipped values are not present
        assert "--disable-log-requests" not in result
        assert "--lora-path" not in result

    def test_preserves_order(self):
        """Arguments should preserve dictionary order (Python 3.7+)."""
        config = {"first": "a", "second": "b", "third": "c"}
        result = build_cli_args_from_config(config)
        assert result == ["--first", "a", "--second", "b", "--third", "c"]

    def test_empty_config(self):
        """Empty config returns empty list."""
        config = {}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_single_element_list(self):
        """Single element list works correctly."""
        config = {"sizes": [42]}
        result = build_cli_args_from_config(config)
        assert result == ["--sizes", "42"]


class TestCliArgsVllmParserRoundTrip:
    """Serialized args must round-trip through vLLM's serve CLI parser."""

    @staticmethod
    def _build_parser():
        import vllm.entrypoints.cli.serve as serve_mod
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(description="test")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        for cmd in serve_mod.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        return parser

    def test_explicit_false_survives_parsing(self):
        """Explicit False survives parse_args and AsyncEngineArgs.from_cli_args."""
        from vllm.engine.arg_utils import AsyncEngineArgs

        parser = self._build_parser()
        config = {
            "skip_tokenizer_init": False,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "enable_sleep_mode": True,
            "enforce_eager": False,
            "disable_log_stats": False,
        }
        argv = ["serve", "dummy-model"] + build_cli_args_from_config(config)
        namespace = parser.parse_args(args=argv)
        engine_args = AsyncEngineArgs.from_cli_args(namespace)
        assert engine_args.enable_prefix_caching is False
        assert engine_args.enable_chunked_prefill is True
        assert engine_args.enable_sleep_mode is True
        assert engine_args.skip_tokenizer_init is False
        assert engine_args.enforce_eager is False
        assert engine_args.disable_log_stats is False


class TestVllmColocateZmqHandle:
    def test_dp_local_rank_offsets_tensor_parallel_rank(self):
        """DP workers on the same node must not reuse the same TP-local socket."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 3
        assert _resolve_vllm_weight_sync_local_rank(3, parallel_config) == 3

    def test_single_dp_keeps_local_rank(self):
        """The old single-DP handle layout remains unchanged."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank_local=0,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 1

    def test_uses_global_dp_rank_when_local_rank_is_unset(self):
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=None,
            data_parallel_rank=3,
        )

        assert _resolve_vllm_weight_sync_local_rank(0, parallel_config) == 2

    def test_zmq_handle_uses_resolved_dp_rank(self, monkeypatch):
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )
        worker = SimpleNamespace(
            local_rank=1,
            model_runner=SimpleNamespace(
                vllm_config=SimpleNamespace(parallel_config=parallel_config),
            ),
        )
        monkeypatch.setenv("VERL_REPLICA_RANK", "2")
        monkeypatch.setenv("VERL_RAY_JOB_ID", "job-123")

        handle = vLLMColocateWorkerExtension._get_zmq_handle(worker)

        assert handle == "ipc:///tmp/rl-colocate-zmq-job-123-replica-2-rank-3.sock"


def _logprob(value: float) -> SimpleNamespace:
    return SimpleNamespace(logprob=value)


def _exact_logprob_rows():
    # Deliberately vary insertion order. The parser must use token IDs and the
    # caller's requested-token order, never backend mapping order.
    return [
        {3: _logprob(-3.1), 10: _logprob(-0.1), 7: _logprob(-7.1)},
        {20: _logprob(-0.2), 7: _logprob(-7.2), 3: _logprob(-3.2), 10: _logprob(-10.2)},
    ]


@pytest.mark.parametrize("exact_enabled", [False, True])
@pytest.mark.parametrize("want_sampled_logprobs", [False, True])
def test_specified_token_request_and_parse_2x2(exact_enabled, want_sampled_logprobs):
    requested_token_ids = (7, 10, 3) if exact_enabled else None
    params = {"temperature": 0.5, "logprobs": want_sampled_logprobs}

    sampled_requested = _prepare_vllm_logprob_request(params, requested_token_ids)

    assert sampled_requested is want_sampled_logprobs
    assert params["temperature"] == 0.5
    if exact_enabled:
        assert params["logprobs"] is None
        assert params["logprob_token_ids"] == [7, 10, 3]
        rows = _exact_logprob_rows()
    else:
        assert params["logprobs"] == (0 if want_sampled_logprobs else None)
        assert "logprob_token_ids" not in params
        rows = [{10: _logprob(-0.1)}, {20: _logprob(-0.2)}] if want_sampled_logprobs else None

    sampled_logprobs, dense_logprobs = _extract_vllm_logprob_rows(
        [10, 20],
        rows,
        want_sampled_logprobs=sampled_requested,
        requested_token_ids=requested_token_ids,
    )

    if want_sampled_logprobs:
        assert sampled_logprobs == pytest.approx([-0.1, -0.2])
    else:
        assert sampled_logprobs is None
    if exact_enabled:
        assert dense_logprobs is not None
        assert dense_logprobs.dtype == np.float32
        np.testing.assert_allclose(
            dense_logprobs,
            np.array([[-7.1, -0.1, -3.1], [-7.2, -10.2, -3.2]], dtype=np.float32),
        )
    else:
        assert dense_logprobs is None


@pytest.mark.parametrize(
    ("reserved_field", "reserved_value"),
    [
        ("logprob_token_ids", [1]),
        ("flat_logprobs", False),
        ("structured_outputs", None),
    ],
)
def test_specified_token_request_rejects_caller_reserved_fields(reserved_field, reserved_value):
    params = {"logprobs": True, reserved_field: reserved_value}

    with pytest.raises(ValueError, match=reserved_field):
        _prepare_vllm_logprob_request(params, (7, 3))


def test_disabled_feature_does_not_claim_or_rewrite_exact_token_fields():
    params = {
        "logprobs": True,
        "logprob_token_ids": [7],
        "flat_logprobs": True,
        "structured_outputs": None,
    }

    assert _prepare_vllm_logprob_request(params, None) is True
    assert params == {
        "logprobs": 0,
        "logprob_token_ids": [7],
        "flat_logprobs": True,
        "structured_outputs": None,
    }


def test_specified_token_parser_validates_unretained_dense_rows():
    rows = _exact_logprob_rows()
    del rows[1][3]

    # A later sparse gather might retain only response position zero, but the
    # backend adapter must still reject the missing cell in dense row one.
    with pytest.raises(RuntimeError, match="response position 1.*token ID 3"):
        _extract_vllm_logprob_rows(
            [10, 20],
            rows,
            want_sampled_logprobs=False,
            requested_token_ids=(7, 10, 3),
        )


def test_specified_token_parser_rejects_missing_sampled_cell_only_when_requested():
    rows = [{7: _logprob(-7.1), 3: _logprob(-3.1)}]

    sampled_logprobs, dense_logprobs = _extract_vllm_logprob_rows(
        [10],
        rows,
        want_sampled_logprobs=False,
        requested_token_ids=(7, 3),
    )
    assert sampled_logprobs is None
    assert dense_logprobs is not None

    with pytest.raises(RuntimeError, match="sampled-token logprob.*token ID 10"):
        _extract_vllm_logprob_rows(
            [10],
            rows,
            want_sampled_logprobs=True,
            requested_token_ids=(7, 3),
        )


@pytest.mark.parametrize("row_count", [0, 2])
def test_specified_token_parser_requires_one_dense_row_per_response_token(row_count):
    rows = [{7: _logprob(-7.1)} for _ in range(row_count)]

    with pytest.raises(RuntimeError, match="different number of logprob rows"):
        _extract_vllm_logprob_rows(
            [10],
            rows,
            want_sampled_logprobs=False,
            requested_token_ids=(7,),
        )


def test_vllm_payload_builder_handles_short_and_empty_responses_with_provenance():
    short_payload = _build_vllm_specified_token_logprobs(
        np.array([[-7.1, -3.1]], dtype=np.float32),
        [0, 2],
        token_ids=(7, 3),
        policy_version=12,
        model_revision="model-commit",
    )
    empty_payload = _build_vllm_specified_token_logprobs(
        np.empty((0, 2), dtype=np.float32),
        [0, 2],
        token_ids=(7, 3),
        policy_version=12,
        model_revision="model-commit",
    )

    assert short_payload.backend == "vllm"
    assert short_payload.backend_version
    assert short_payload.policy_version == 12
    assert short_payload.model_revision == "model-commit"
    assert short_payload.response_token_count == 1
    assert short_payload.position_indices.tolist() == [0]
    assert short_payload.logprobs.shape == (1, 2)
    assert empty_payload.response_token_count == 0
    assert empty_payload.position_indices.shape == (0,)
    assert empty_payload.logprobs.shape == (0, 2)


@pytest.mark.parametrize("invalid_policy_version", [None, -1, True, 1.0, "1"])
def test_specified_token_policy_version_fails_closed(invalid_policy_version):
    with pytest.raises(RuntimeError, match="policy version"):
        _require_policy_version(invalid_policy_version)


def test_specified_token_policy_version_accepts_non_negative_python_int():
    assert _require_policy_version(0) == 0
    assert _require_policy_version(9) == 9


def test_specified_token_explicit_max_tokens_cannot_exceed_response_or_capture_budget():
    with pytest.raises(ValueError, match="rollout.response_length"):
        _validate_specified_token_max_tokens(9, response_length=8, max_capture_positions=10)
    with pytest.raises(ValueError, match="max_capture_positions"):
        _validate_specified_token_max_tokens(9, response_length=10, max_capture_positions=8)

    _validate_specified_token_max_tokens(8, response_length=8, max_capture_positions=8)


class _AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _streaming_reward_config():
    return _AttrDict(
        num_workers=1,
        reward_model=_AttrDict(enable=False, enable_resource_pool=False),
        custom_reward_function=_AttrDict(path=None),
    )


def _server_for_config_validation(*, token_ids, engine_kwargs=None):
    server = object.__new__(vLLMHttpServer)
    server.config = _AttrDict(
        max_model_len=32,
        specified_token_logprobs=SimpleNamespace(enabled=token_ids is not None, token_ids=token_ids),
        engine_kwargs=engine_kwargs or {},
    )
    server.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(max_position_embeddings=32, vocab_size=100, _commit_hash="model-commit")
    )
    return server


class _AbortingEngine:
    def __init__(self):
        self.sampling_params = None

    async def generate(self, *, sampling_params, **kwargs):
        del kwargs
        self.sampling_params = sampling_params
        yield SimpleNamespace(outputs=[])


@pytest.mark.asyncio
async def test_generate_abort_returns_present_empty_specified_token_payload(monkeypatch):
    monkeypatch.delenv("VERL_RL_INSIGHT_ENABLE", raising=False)
    server = object.__new__(vLLMHttpServer)
    server._disaggregation_role = "null"
    server._pd_decode_peers = []
    server.config = _AttrDict(
        max_model_len=32,
        prompt_length=8,
        response_length=8,
        full_determinism=False,
        repetition_penalty=1.0,
        ignore_eos=False,
        specified_token_logprobs=SimpleNamespace(
            enabled=True,
            token_ids=[7, 3],
            positions=[0, 2],
            max_capture_positions=8,
        ),
    )
    server.model_config = SimpleNamespace(
        processor=None,
        hf_config=SimpleNamespace(_commit_hash="model-commit"),
    )
    server.global_steps = 5
    server.replica_rank = 0
    server.lora_as_adapter = False
    server.engine = _AbortingEngine()

    output = await server.generate(
        prompt_ids=[1, 2],
        sampling_params={"logprobs": False, "max_tokens": 2},
        request_id="abort-test",
    )

    assert output.stop_reason == "aborted"
    assert output.specified_token_logprobs is not None
    assert output.specified_token_logprobs.response_token_count == 0
    assert output.specified_token_logprobs.position_indices.shape == (0,)
    assert output.specified_token_logprobs.logprobs.shape == (0, 2)
    assert output.specified_token_logprobs.policy_version == 5
    assert output.specified_token_logprobs.model_revision == "model-commit"
    assert server.engine.sampling_params.logprobs is None
    assert server.engine.sampling_params.logprob_token_ids == [7, 3]


def test_specified_token_config_validation_checks_vocab_bounds():
    server = _server_for_config_validation(token_ids=[7, 100])

    with pytest.raises(ValueError, match="out-of-range values.*100"):
        server._validate_configs()


@pytest.mark.parametrize(
    ("engine_key", "engine_value"),
    [
        ("speculative_config", {"method": "ngram"}),
        ("speculative-config", {"method": "ngram"}),
        ("spec_method", "ngram"),
        ("spec-method", "ngram"),
        ("spec_model", "draft-model"),
        ("spec-model", "draft-model"),
        ("spec_tokens", 3),
        ("spec-tokens", 3),
    ],
)
def test_specified_token_config_validation_rejects_engine_speculative_config(engine_key, engine_value):
    server = _server_for_config_validation(
        token_ids=[7],
        engine_kwargs={"vllm": {engine_key: engine_value}},
    )

    with pytest.raises(NotImplementedError, match="speculative decoding"):
        server._validate_configs()


@pytest.mark.parametrize("engine_key", ["logprobs_mode", "logprobs-mode"])
def test_specified_token_config_validation_rejects_engine_logprobs_mode_override(engine_key):
    server = _server_for_config_validation(
        token_ids=[7],
        engine_kwargs={"vllm": {engine_key: "processed_logprobs"}},
    )

    with pytest.raises(ValueError, match="do not allow an override"):
        server._validate_configs()


def test_specified_token_config_validation_requires_vllm_020(monkeypatch):
    import verl.workers.rollout.vllm_rollout.vllm_async_server as server_module

    server = _server_for_config_validation(token_ids=[7])
    monkeypatch.setattr(server_module, "_VLLM_VERSION", server_module.version.parse("0.19.0"))

    with pytest.raises(RuntimeError, match="vLLM>=0.20.0"):
        server._validate_configs()


def test_specified_token_config_validation_checks_sampling_params_capability(monkeypatch):
    import verl.workers.rollout.vllm_rollout.vllm_async_server as server_module

    server = _server_for_config_validation(token_ids=[7])
    monkeypatch.setattr(server_module, "_vllm_supports_logprob_token_ids", lambda: False)

    with pytest.raises(RuntimeError, match="does not expose logprob_token_ids"):
        server._validate_configs()


def test_disabled_feature_skips_exact_token_capability_gates(monkeypatch):
    import verl.workers.rollout.vllm_rollout.vllm_async_server as server_module

    server = _server_for_config_validation(token_ids=None)
    monkeypatch.setattr(server_module, "_VLLM_VERSION", server_module.version.parse("0.1.0"))
    monkeypatch.setattr(server_module, "_vllm_supports_logprob_token_ids", lambda: False)

    server._validate_configs()


@pytest.mark.parametrize(
    "rollout_correction",
    [
        {"rollout_is": "token", "rollout_rs": None, "bypass_mode": False},
        {"rollout_is": None, "rollout_rs": "token_k1", "bypass_mode": False},
        {"rollout_is": None, "rollout_rs": None, "bypass_mode": True},
    ],
)
def test_validate_config_rejects_active_correction_with_specified_token_logprobs(rollout_correction):
    config = SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
        actor_rollout_ref=SimpleNamespace(
            rollout=_AttrDict(
                specified_token_logprobs=_AttrDict(token_ids=[7]),
                calculate_log_probs=True,
            )
        ),
        algorithm=_AttrDict(rollout_correction=_AttrDict(rollout_correction)),
        reward=_streaming_reward_config(),
    )

    with pytest.raises(ValueError, match="engine-global.*sampled rollout_log_probs"):
        validate_config(config, use_reference_policy=False, use_critic=False)


def test_validate_config_allows_sampled_logprobs_without_active_correction():
    actor = SimpleNamespace(
        use_dynamic_bsz=True,
        use_kl_loss=False,
        validate=lambda *_args: None,
    )
    rollout = _AttrDict(
        specified_token_logprobs=_AttrDict(token_ids=[7]),
        calculate_log_probs=True,
        val_kwargs=SimpleNamespace(do_sample=False),
        name="vllm",
    )
    config = SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
        actor_rollout_ref=SimpleNamespace(
            actor=actor,
            rollout=rollout,
            model=_AttrDict(lora={}, lora_rank=0),
        ),
        algorithm=_AttrDict(
            rollout_correction=_AttrDict(rollout_is=None, rollout_rs=None, bypass_mode=False),
            use_kl_in_reward=False,
        ),
        data=_AttrDict(train_batch_size=1, val_batch_size=None),
        reward=_streaming_reward_config(),
    )

    validate_config(config, use_reference_policy=False, use_critic=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
