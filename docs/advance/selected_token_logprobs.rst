Selected-token rollout log probabilities
========================================

Last updated: 09/03/2026.

verl can optionally capture, during rollout, the log probabilities of a small
caller-selected set of token IDs at every response position (or at a fixed list
of positions) and hand them to the reward function. Typical consumers are
generative rankers and classifiers that read their score off specific tokens,
for example a ``yes``/``no`` relevance judgement or per-item rating digits,
where the reward wants the policy's probability mass rather than the single
token that happened to be sampled.

The first revision supports single-turn vLLM rollouts and the streaming reward
function.

Configuration
-------------

The feature is disabled while ``token_ids`` is ``null``. Setting it enables
capture; ``positions`` optionally restricts which response positions are kept:

.. code-block:: yaml

   actor_rollout_ref:
     rollout:
       name: vllm
       selected_token_logprobs:
         token_ids: [9454, 2753]   # e.g. the "yes" and "no" token IDs
         positions: null           # every response position, or e.g. [0, 3, 7]
         max_payload_bytes_per_sample: 4194304

``token_ids`` preserves caller order and holds at most 128 IDs (vLLM's
``logprob_token_ids`` limit). ``positions`` is either ``null``, which keeps every
position of the final unpadded response, or a unique, strictly increasing list
of zero-based positions. For a response of length ``T``, verl returns exactly
the configured positions smaller than ``T``: positions made unavailable by a
short response are omitted, while a missing backend row or token value at an
available position is an error. If no configured position is available, the
payload is still present with shapes ``[0]`` and ``[0, M]`` rather than
treating the feature as disabled.

The values follow ``rollout.logprobs_mode``, which the feature does not change.
With verl's default ``processed_logprobs`` they are log-softmax values of the
logits after temperature and other sampling processors; ``raw_logprobs`` gives
the unprocessed distribution; the ``*_logits`` modes carry unnormalised scores.
The mode in force, including an ``engine_kwargs`` override, is recorded in the
payload. Because the engine-global mode is untouched, the feature composes with
rollout correction and everything else that reads ``rollout_log_probs``.

Reward function access
----------------------

The configured streaming reward function receives the payload under
``extra_info["selected_token_logprobs"]``:

- ``token_ids``: the configured IDs, defining the column order.
- ``response_ids``: the final response tokens the rows belong to.
- ``positions``: ``int32`` array of shape ``[P]``, the response positions of the rows.
- ``logprobs``: ``float32`` array of shape ``[P, M]``.
- ``logprobs_mode``, ``backend``, ``backend_version``, ``schema_version``: provenance.

Positions may also be selected in the reward by token, using ``response_ids``,
which covers selectors that a fixed position list cannot express:

.. code-block:: python

   import numpy as np

   YES, NO = 9454, 2753

   def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
       evidence = extra_info["selected_token_logprobs"]
       logprobs = evidence["logprobs"]  # float32 [P, M], columns follow token_ids
       positions = evidence["positions"]  # int32 [P]
       response_ids = evidence["response_ids"]
       # Score the first position whose sampled token is "yes" or "no".
       for row, position in enumerate(positions):
           if response_ids[position] in (YES, NO):
               p_yes, p_no = np.exp(logprobs[row])
               return float(p_yes / (p_yes + p_no))
       return 0.0

The payload is copied only into the reward call. It is removed before actor,
reference-policy, or critic processing and excluded from rollout traces.

Compatibility and cost
----------------------

The first implementation rejects multi-turn and tool agent loops, custom agent
loop registries or managers, FullyAsync partial-resume clients, prefill/decode
disaggregation, speculative decoding, structured outputs, colocated reward
computation, and the built-in discriminative reward-model path. At least one
streaming reward worker is required.

``calculate_log_probs`` remains independently supported: vLLM returns the
sampled token together with the selected token IDs, and verl recovers the
sampled-token scalar by token ID.

vLLM gathers ``M`` extra columns per decode step without enabling top-k
capture, and the payload transported to the reward is ``P x M`` ``float32``
values plus ``P`` ``int32`` positions per sample. The per-sample byte limit is
checked at configuration time from ``response_length`` (or the position list)
and is a hard error; verl never silently truncates the evidence. Restrict
``positions`` when only a few positions matter.

The training-side counterpart, ``actor.selected_token_logprobs``, exposes the
same columns from the actor forward pass for auxiliary objectives; the two
share the token-ID contract but are configured independently.
