# Study 1 — Trajectory Supervision and Self-Context Distillation

**Status: planned Colab-first curriculum; no Study 1 notebook or training result exists yet.**

## Research objective
Learn how selected decisions from coding/ML-engineering agent trajectories can form useful SFT supervision, how to evaluate data sufficiency/leakage, and whether reflections and rehearsal improve actual *held-out task success*. This trains the **task-performing model**, not the future APT diagnostician (Study 2).

## Prerequisites
The first real TRL SFTTrainer lab with loss-mask inspection, trusted train/validation split and reproducible checkpoint evaluation. The initial GRPO clipping toy lab is a separate prerequisite for later GRPO work, not a dependency for trajectory SFT.

## Hands-on Colab sequence
Each lab provides conceptual/mathematical grounding, reproducible setup, plotting style utilities, learner-completed metric/plot calculations, tests, an experiment report, and a measured-versus-hypothesized evidence distinction.

| Lab | Implementation exercise | Student-built diagnostic plots | Saved result |
|---|---|---|---|
| S1-01 | Parse agent trajectory actions/observations into decision-level candidate examples; preserve trace provenance | Selection rate by outcome, length, task family; evidence availability | Validated candidate manifest |
| S1-02 | Build prompt/completion examples; test redaction of privileged guidance and post-outcome information | Masked-token and context-length distributions, leakage audit | Clean split + token mask report |
| S1-03 | TRL SFTTrainer selected-action baseline versus an appropriate demonstration baseline | Train/eval loss and held-out success by task family across checkpoints | Paired checkpoint comparison |
| S1-04 | Construct independently auditable reflection/replay mixtures; control token/compute budgets | Skill retention and slice-by-slice generalization, confidence intervals | Ablation report with multiple seeds |
| S1-05 | Held-out decisions and coding tasks from new repositories; assess transfer and failure explanations | Coverage, calibration/error breakdown, success versus example type | Research conclusions + limitations |

## Scientific safeguards
- Selected decisions must have enough *model-visible* preceding evidence; hidden teacher hints and future results cannot leak into student inputs.
- Train/validation/held-out splits need repository/task-family/template separation where feasible.
- Control supervised tokens and compute, not only number of examples, across data-mixture comparisons.
- Retrospective reflections may contain causal claims; mark unverified interpretations, do not convert them into ground truth labels.
- Diagnostic gains do not follow from training loss alone; measure trusted held-out coding/research-task success.
- This is inspired by Bespoke Labs' research, not a claim that their results have been replicated.

## Links
- https://bespokelabs.ai/blog/teaching-nvidias-nemotron-ml-engineering
- https://huggingface.co/docs/trl/sft_trainer
- https://github.com/huggingface/Repo2RLEnv
