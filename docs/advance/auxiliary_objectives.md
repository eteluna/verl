# Actor-Side Auxiliary Objectives

Last updated: 09/02/2026.

`actor.auxiliary_objectives` lets you add differentiable terms to the actor update without patching
`ppo_loss` or the training engine:

```
L_total = L_policy (pg + entropy + KL, unchanged)  +  sum_i weight_i * L_i
```

Each objective is a small Python class you point the config at. verl loads it, runs it in a fixed
order, normalizes it over the whole mini-batch, applies the coefficient once, and publishes its
metrics under `actor/aux/<name>/`. With the default empty list nothing changes.

Typical uses: calibration of the probability the policy assigns to a few label tokens, a supervised
term on part of the batch, a consistency or auxiliary-prediction loss. Rewards cannot express these:
a reward only changes advantages, it cannot put a gradient through the current-policy outputs.

## Configuration

```yaml
actor_rollout_ref:
  actor:
    auxiliary_objectives:
      - name: digit_calibration          # unique; metrics land under actor/aux/digit_calibration/
        path: /workspace/objectives.py   # loaded like custom_reward_function.path
        factory: build_objective         # callable in that file, called with **kwargs
        weight: 0.05                     # applied exactly once by verl; 0 still runs (ablation metrics)
        kwargs:
          token_ids: [15, 16, 17, 18, 19]
          positive_token_ids: [17, 18, 19]
```

Constraints: names unique, weights finite, order deterministic and identical on every rank (verl
compares a digest of the configuration across ranks at start-up).

## Writing an objective

```python
from verl.workers.utils.auxiliary_objectives import ActorObjectiveResult, BaseActorAuxiliaryObjective

class LengthPenalty(BaseActorAuxiliaryObjective):
    required_batch_keys = ("response_mask",)
    required_model_output_keys = ("log_probs",)

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

Three callbacks:

| Stage | When | Contract |
| --- | --- | --- |
| `prepare_batch(data)` | once per mini-batch, before micro-batch splitting | returns local additive scalars (counts, sums). verl packs every objective's scalars into one tensor and SUM-reduces once over the data-parallel group. Never run collectives yourself. |
| `process_logits(logits, data, context)` | per micro-batch, inside the forward, only if `uses_logits_processor = True` | reduce the transient `(total_nnz, vocab)` logits to compact `(total_nnz, ...)` tensors. They are re-nested per sequence and appear in `model_output` as `aux/<name>/<key>`. |
| `compute(model_output, data, context)` | per micro-batch, after the base loss | return `ActorObjectiveResult(loss_sum, normalizer, metrics)`. |

`loss_sum` is the **sum** over this micro-batch's applicable elements, unweighted, connected to the
autograd graph. `normalizer` names one of your `prepare_batch` statistics. verl computes

```
contribution = dp_size * loss_sum / global_stats[normalizer]
```

per micro-batch, which sums across gradient-accumulation steps and averages across ranks to the
global mini-batch mean, the same trick `ppo_loss` uses with `batch_num_tokens`. If the global
normalizer is zero the objective is reported `active=0`, must return a zero `loss_sum`, and
contributes a differentiable zero. Objectives only run on the gradient-enabled actor update, never
during rollout, reference, old-log-prob or validation passes.

`context` carries `name` (your configured name, so a processor output is at
`model_output[f"aux/{context.name}/<key>"]`), `global_stats` (your reduced statistics), `dp_size`,
`loss_agg_mode`, `batch_num_tokens`, `global_batch_size`. Everything else is read-only: do not mutate the batch,
`model_output` or the base loss, and do not call `backward`.

### Published metrics

For every objective: `actor/aux/<name>/loss` (unweighted, SUM over micro-batches), `weighted_loss`,
`normalizer`, `active`, plus anything in `ActorObjectiveResult.metrics` (plain numbers become MEAN
metrics; pass a `Metric` to choose otherwise). Keys must not collide.

## Using the logits

`process_logits` sees exactly the logits `log_probs` is computed from: already divided by the
rollout temperature, before any other transform, laid out as `(total_nnz, vocab)` on both the
remove-padding and padded paths (the engine converts). Reduce them immediately; the full logits
never leave the engine and outputs wider than `vocab / 4` per token are rejected.

Two consequences you do not have to handle yourself but should know about:

- **In-place cross-entropy backward.** verl's fused cross-entropy writes into the logits buffer during
  backward. Anything whose backward still needs those logits (a `logsumexp`, for example) would get
  silently wrong gradients. When any processor ran, the engine keeps the buffer intact
  (`inplace_backward=False`); this costs one extra logits-sized buffer during backward.
- **Fused kernels.** `use_fused_kernels=True` never materializes the logits, so a processor cannot
  run. Configuring both fails at initialization.

## Reference objective: calibrating label tokens

`SpecifiedTokenCalibrationObjective` (in `verl.workers.utils.auxiliary_objectives`) is a complete
example. At every response position where `data["calibration_target"]` is 0 or 1 it reads the
current-policy mass on `positive_token_ids` out of `token_ids` and applies binary cross-entropy;
positions marked -1 are ignored. Group-relative advantages (GRPO) only constrain ordering within a
prompt; a term like this anchors absolute probability levels across prompts. How the
`calibration_target` tensor gets into the batch (a dataset field, a reward-side transform) is up to
you.

```python
from verl.workers.utils.auxiliary_objectives import SpecifiedTokenCalibrationObjective

def build_objective(token_ids, positive_token_ids, target_key="calibration_target"):
    return SpecifiedTokenCalibrationObjective(token_ids, positive_token_ids, target_key=target_key)
```

## Scope and limitations

- Supported: FSDP and FSDP2 actors. Megatron, VeOmni and TorchTitan reject a non-empty list at
  initialization (backend parity is a follow-up).
- Objectives are stateless across steps and add no parameters, optimizer groups, checkpoint entries
  or rollout weight-sync changes. Sidecar heads are out of scope.
- The on-policy distillation loss keeps working: the composer wraps it and delegates its
  logits-processor calls unchanged.
- Producing new batch fields (for example from reward extra info) is not part of this feature.

## Failure handling

Missing required keys, a detached loss while active, a non-scalar or non-finite loss, an unknown
normalizer, a metric-key collision, an output with the wrong leading dimension, an unsupported
backend or a fused-kernel conflict all raise with the objective name in the message. Nothing is
skipped with a warning.
