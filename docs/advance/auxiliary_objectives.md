# Actor-Side Auxiliary Objectives

Last updated: 09/03/2026.

`actor.auxiliary_objectives` lets you add differentiable terms to the actor update without patching
`ppo_loss` or the training engine:

```
L_total = L_policy (pg + entropy + KL, unchanged)  +  sum_i weight_i * L_i
```

Each objective is a small Python class you point the config at. verl loads it, runs it in a fixed
order, normalizes it over the whole mini-batch, applies the coefficient once, and publishes its
metrics under `actor/aux/<name>/`. With the default empty list nothing changes.

Typical uses: calibration of the probability the policy assigns to a few label tokens, a supervised
term on part of the batch, an auxiliary-prediction loss. A reward cannot express these: it only changes
the advantages of the sampled tokens, so it cannot directly optimize a function of the current full
output distribution.

Objectives only read `model_output`. When a term needs more than `log_probs`, the engine produces it
from configuration (see *Selected-token log-probabilities* below); plugins never touch the full logits.

## Configuration

```yaml
actor_rollout_ref:
  actor:
    selected_token_logprobs:
      token_ids: [15, 16, 17, 18, 19]  # engine emits model_output["selected_token_logprobs"], (…, 5)
    auxiliary_objectives:
      - name: label_calibration          # unique; metrics land under actor/aux/label_calibration/
        path: /workspace/objectives.py   # loaded like custom_reward_function.path
        factory: build_objective         # callable in that file, called with **kwargs
        weight: 0.05                     # applied exactly once by verl
        metrics_only: false              # true = run under no_grad for metrics, never in the loss
        kwargs:
          positive_token_ids: [17, 18, 19]
          normalize_over: token_set
```

Constraints: names unique, weights finite, order deterministic and identical on every rank. verl
compares a digest of the resolved configuration (entries, kwargs, plugin file hash, API version) across
ranks at start-up and refuses to start on a mismatch. An objective with `weight: 0` and
`metrics_only: false` is skipped entirely; use `metrics_only: true` to keep its metrics for ablations.

## Writing an objective

```python
from verl.workers.utils.auxiliary_objectives import ActorObjectiveResult, BaseActorAuxiliaryObjective

class LengthPenalty(BaseActorAuxiliaryObjective):
    required_batch_keys = ("response_mask",)
    required_model_output_keys = ("log_probs",)
    stat_names = ("tokens",)                  # the exact keys prepare_batch returns, on every rank

    def prepare_batch(self, data):
        # local, gradient-free counts over the UNSPLIT mini-batch; verl SUM-reduces them over DP
        return {"tokens": data["response_mask"].sum()}

    def compute(self, *, model_output, data, context):
        from verl.workers.utils.padding import no_padding_2_padding
        lp = no_padding_2_padding(model_output["log_probs"], data)      # (bsz, response_len)
        mask = data["response_mask"].to_padded_tensor(0).bool()
        return ActorObjectiveResult(loss_sum=(-lp * mask).sum(), normalizer="tokens")

def build_objective():
    return LengthPenalty()
```

Two callbacks and three declarations:

| | When | Contract |
| --- | --- | --- |
| `stat_names`, `required_batch_keys`, `required_model_output_keys` | validated at start-up | static declarations. `required_model_output_keys` is checked against what the configured forward emits, so asking for `entropy` without `calculate_entropy` fails at init with the flag to set, not on the first forward. |
| `prepare_batch(data)` | once per mini-batch, before micro-batch splitting | returns exactly `stat_names`, local additive scalars (counts, sums). verl packs every objective's scalars in a static order and SUM-reduces once over the data-parallel group. Never run collectives yourself. |
| `compute(model_output, data, context)` | per micro-batch, after the base loss | returns `ActorObjectiveResult(loss_sum, normalizer, metrics)`. |

`loss_sum` is the **sum** over this micro-batch's applicable elements, unweighted, connected to the
autograd graph. `normalizer` names one of your `stat_names`. verl computes

```
contribution = dp_size * loss_sum / global_stats[normalizer]
```

per micro-batch, which sums across gradient-accumulation steps and averages across ranks to the
global mini-batch mean, the same trick `ppo_loss` uses with `batch_num_tokens`. This assumes the
objective is additive over samples; cross-sample terms (contrastive losses over the whole batch) are
out of scope. If the global normalizer is zero the objective is reported `active=0`, must return a
zero `loss_sum`, and contributes a differentiable zero. Objectives only run on the gradient-enabled
actor update, never during rollout, reference, old-log-prob or validation passes.

`context` carries `name`, `global_stats` (your reduced statistics), `dp_size`, `loss_agg_mode`,
`batch_num_tokens`, `global_batch_size`, and `selected_token_ids` (column order of the projection
below). Everything else is read-only: do not mutate the batch, `model_output` or the base loss, and do
not call `backward`.

### Published metrics

For every objective: `actor/aux/<name>/loss` (unweighted, SUM over micro-batches), `weighted_loss`,
`normalizer`, `active`, plus anything in `ActorObjectiveResult.metrics` (plain numbers become MEAN
metrics; pass a `Metric` to choose otherwise). Keys must not collide.

## Selected-token log-probabilities

Many objectives need the probability the current policy assigns to a few specific tokens at every
position (label letters, yes/no, digits). The full logits are the largest tensor of the update and
never leave the engine. Instead, `actor.selected_token_logprobs.token_ids` makes the engine emit, on
every actor-update forward:

```
model_output["selected_token_logprobs"]   # laid out like log_probs, trailing dim = len(token_ids)
                                          # value = log_softmax(logits / temperature)[token_id]
```

It is computed row-wise from exactly the logits `log_probs` comes from, so it is correct under
remove-padding, packed padding and Ulysses sequence parallelism, and it costs one `(tokens, M)`
float32 tensor. Two things the engine handles for you:

- **In-place cross-entropy backward.** verl's fused cross-entropy writes into the logits buffer during
  backward; the projection's `logsumexp` backward still reads it. When the projection ran, the engine
  keeps the buffer intact (`inplace_backward=False`), at the cost of one extra logits-sized buffer.
- **Fused kernels.** `use_fused_kernels=True` never materializes the logits, so the projection cannot
  run. Configuring both fails at initialization.

## Reference objective: calibrating label tokens

`SelectedTokenCalibrationObjective` (in `verl.workers.utils.auxiliary_objectives`) is a complete
example. At every response position where `data["calibration_target"]` is 0 or 1 it forms the logit
of the positive-token set against the negative set and applies `binary_cross_entropy_with_logits`;
positions marked -1 are ignored. Group-relative advantages (GRPO) only constrain ordering within a
prompt; a term like this anchors absolute probability levels across prompts.

`normalize_over` picks the denominator:

- `token_set` (default): negatives are the other selected ids, so the probability is conditional on the
  label alphabet. A yes/no reranker that scores with `log_softmax(logits[[no, yes]])[yes]` is
  calibrated with `positive_token_ids: [yes]`, `token_ids: [no, yes]`.
- `vocab`: negatives are the whole rest of the vocabulary, i.e. the raw full-vocabulary mass of the
  positive tokens.

```python
from verl.workers.utils.auxiliary_objectives import SelectedTokenCalibrationObjective

def build_objective(positive_token_ids, normalize_over="token_set", target_key="calibration_target"):
    return SelectedTokenCalibrationObjective(positive_token_ids, normalize_over=normalize_over, target_key=target_key)
```

How the `calibration_target` tensor gets into the batch (a dataset field, a reward-side transform) is up
to you.

## Scope and limitations

- Supported: FSDP and FSDP2 actors. Megatron, VeOmni and TorchTitan reject a non-empty list at
  initialization (backend parity, with backend-provided reduction semantics, is a follow-up).
- Objectives are stateless across steps and add no parameters, optimizer groups, checkpoint entries
  or rollout weight-sync changes. Sidecar heads are out of scope.
- No plugin callback receives the logits. A generic logits-processing stage may follow once the
  sequence-parallel layout and fused-kernel interactions are specified; the on-policy distillation
  loss keeps its own processor path, which the composer delegates unchanged.
- Producing new batch fields (for example from reward extra info) is not part of this feature.

## Failure handling

Missing required keys, an objective asking for outputs the forward does not emit, a `prepare_batch`
that returns keys other than its `stat_names`, a detached loss while active, a non-scalar or non-finite
loss, an unknown normalizer, a metric-key collision, an unsupported backend, a fused-kernel conflict
or a configuration that differs across ranks all raise at start-up or with the objective name in the
message. Nothing is skipped with a warning except a `weight: 0` objective without `metrics_only`.
