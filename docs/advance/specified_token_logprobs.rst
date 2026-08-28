Specified-token rollout log probabilities
===========================================

Last updated: 08/28/2026.

VERL can optionally capture full-vocabulary-normalized log probabilities for a
small, caller-specified set of token IDs at explicit response positions. The
initial implementation supports single-turn vLLM rollouts and passes the sparse
payload transiently to a streaming reward function.

Configuration
-------------

The feature is disabled when ``token_ids`` and ``positions`` are ``null``. To
enable it, set both lists and use vLLM's engine-global ``raw_logprobs`` mode:

.. code-block:: yaml

   actor_rollout_ref:
     rollout:
       name: vllm
       logprobs_mode: raw_logprobs
       multi_turn:
         enable: false
       specified_token_logprobs:
         token_ids: [101, 202, 303]
         mode: raw_logprobs
         positions: [0, 3, 7]
         consumers: [reward]
         max_capture_positions: 4096
         max_requested_positions: 128
         max_payload_bytes_per_sample: 1048576

``token_ids`` preserves caller order. ``positions`` contains unique, strictly
increasing, zero-based positions in the final unpadded response. For a response
of length ``T``, VERL returns exactly the configured positions smaller than
``T``. Positions made unavailable by a short response are omitted; a missing
backend row or token value at an available position is an error. If all
configured positions are unavailable, the feature returns a present typed
payload with shapes ``[0]`` and ``[0, M]`` rather than treating the feature as
disabled.

Reward function access
----------------------

The configured streaming reward function receives the payload under
``extra_info["specified_token_logprobs"]``. The numeric arrays are contiguous
``numpy.ndarray`` objects: ``position_indices`` is ``int32`` with shape ``[P]``
and ``logprobs`` is ``float32`` with shape ``[P, M]``.

.. code-block:: python

   def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
       evidence = extra_info["specified_token_logprobs"]
       positions = evidence["position_indices"]
       label_logprobs = evidence["logprobs"]
       # Compute a task-specific reward from positions and label_logprobs.
       return {"score": float(label_logprobs[0, 0]) if len(positions) else 0.0}

The payload also records its schema version, requested token IDs, realized
response length, normalization and mode, rollout backend and version, policy
version, and optional model revision. It is copied only into the reward call and
is removed before actor, reference-policy, or critic processing.

Compatibility and cost
----------------------

The first implementation intentionally rejects multi-turn loops, custom agent
loop registries or managers, FullyAsync partial-resume clients, prefill/decode disaggregation,
speculative decoding, structured outputs, rollout tracing, colocated reward
computation, the built-in discriminative reward-model path, and the future
``actor_auxiliary`` consumer. At least one streaming reward worker is required.
It also rejects rollout correction while the feature is active because
``logprobs_mode`` is engine-global and therefore changes the semantics of
sampled-token rollout log probabilities as well.

``calculate_log_probs`` remains independently supported: vLLM returns the
sampled token together with the explicitly requested token IDs, and VERL
recovers the sampled-token scalar by token ID.

Sparse positions reduce the server-to-reward payload to ``P x M`` values, but
they do not reduce vLLM's per-step capture work. The backend still produces and
VERL validates a dense ``T x M`` result before gathering the configured
positions. The capture-position, requested-position, and per-sample payload
limits are hard errors; VERL never silently truncates evidence to meet them.

The design contract and planned follow-up consumers are described in the
`RFC <https://docs.google.com/document/d/1SPZEfh0lqhLbN4YNDqfSZCXKCSlMmWsMvt_06M9c4SI/edit>`_.
