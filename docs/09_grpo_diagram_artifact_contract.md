# GRPO diagrams ↔ empirical artifacts: versioned contract v1

**Implementation:** `src/posttraining_engineer/training/grpo_artifacts.py`. This is a typed schema and theoretical reconstruction, **not yet actual GRPOTrainer instrumentation**. The current toy Lab 01 does not itself yield true LLM rollouts.

## Provenance: don't conflate rollout step and optimizer step

**Generated completion group** (`GroupEvidence`) records `run_id, rollout_step, group_id, task_id, split, rewards, verifier_version`. Reward-group tie diagnostics use these observations.

**Optimization-time token** (`TokenEvidence`) records `run_id, optimizer_step, group_id, completion_id, token_position, completion_mask, old_logprob, current_logprob, advantage, reward`. Those current-policy log probabilities are not necessarily available at rollout generation: obtain them at the optimizer step with explicit policy IDs in `GRPOContext`. Keep rollout and optimization axes separate.

**Context** records dataset revision, verifier version, trainer version, policy and behavior checkpoints, ratio denominator identity, epsilon boundaries, advantage and reduction definitions. An optional reference-policy checkpoint is distinct from the old/behavior policy. Do not use reference KL logprobs as ratio denominators.

**Step metrics** (`StepEvidence`) keep gradient norms and measured KL with definition; heldout success is separately labeled and requires independent evaluation. Missing is `null`, never zero. A multi-turn Harbor rollout is **not** automatically a GRPO token batch.

## Aligned diagrams and diagnostic claims

| Diagram pair | Dummy layer | Empirical input | Exact check |
|---|---|---|---|
| Clipped objective curves | evaluate surrogate at a grid of ratios and +/− advantages | ratio and advantage per eligible optimizer token | compare `raw_objective`, `clipped_branch`, `surrogate_objective` and signs |
| Ratio–advantage suppression map | theoretical active regions | tokens reconstructed from both logprobs, mask and advantage | `active_clip` iff positive advantage & ratio > upper bound OR negative advantage & ratio < lower bound |
| Clip frequency over time | hypothetical sequence (explicitly illustrative) | token rows grouped by **optimizer_step** | fraction active among **valid supervised tokens**, with denominator/missingness |
| Reward ties | synthetic reward groups | independently graded rollout groups | group reward variance/tie rate by **rollout_step** |
| Gradient norm vs surrogate clipping | independent schematic | gradient norms logged before/after optimizer norm clipping | do not infer gradient norm suppression from ratio clipping alone |

## Instrumentation gate before claiming a real plot
1. Pick and pin the exact TRL version, inspect loss implementation, and decide whether ratios use stored old or behavior logprobs.
2. Capture IDs and masks for **the same** generated tokens at their optimizer update. Derive `ratio = exp(current_logprob - old_logprob)`; validate against any trainer logged ratio.
3. Record advantage definition, task group membership, how masked tokens are handled, sequence aggregation, and whether variant weights/KL terms alter the objective. The provided `compute_surrogate` is only the **unweighted standard two-sided clipping surrogate**, not all TRL loss variants.
4. Collect gradient norm/KL values only with their metric definitions and step axes; compute no imaginary values for missing fields.
5. Use matched task/evaluation sets and compute budgets for a clipped-versus-unclipped experiment. A stylized trajectory never proves clipping improves stability or held-out success.

## Learner ownership
Tutor may demonstrate a **fixed, labeled synthetic fixture** and rendering utilities. The learner selects real optimizer-token records, implements analysis aggregates and builds the empirical graphs, validates expected boundaries, and writes the interpretation. A future `GRPOTrainer` adapter must produce records in this schema; the current notebook's placeholder is not an accomplished data bridge.

## Suggested fixture
- Positive advantage +1 and ratio 1.3: **upper active clip** with epsilon 0.2.
- Negative advantage −1 and ratio 0.7: **lower active clip**.
- Negative advantage −1 and ratio 1.3: **outside band but not actively suppressed**.
- Ratio 1.0 and positive advantage +1: **inside, no clip**.

This last contrast prevents a common error: **outside the ratio band does not imply active clipping**.
