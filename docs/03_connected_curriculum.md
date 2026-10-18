# APT — Connected TRL SFT → GRPO Tutorial Roadmap

**Status: curriculum designed; thirteen Study 1/2 Colab student skeletons and two TRL trainer starters committed on `feat/curriculum-skeletons`. Training runs and integration are NOT yet implemented.** APT's objective is to explain why post-training does not improve trusted held-out coding-task success and choose a discriminating next experiment.

## Required teaching contract
Every chapter must have: (1) prerequisite artifact, (2) math and learning goals, (3) student-written implementation exercises, (4) runnable commands, (5) tests and expected output, (6) saved output for the next lab, (7) failure analysis, (8) open-ended research question without a preselected answer. No disconnected math exercises presented as if they were working trainer instrumentation.

## Connected sequence — diagnostics and training together

**Baseline on main:** Chapter 04 representation, one toy GRPO clipping Colab. **On implementation branch:** learner skeletons for TRL SFT/GRPO, both studies, adapters/probes/tests. **Still planned:** actual training runs, Harbor connectivity, recovery experiments.

| Order | Hands-on milestone | Evidence/diagnostic skill developed in the same milestone |
|---|---|---|
| 1 | [Colab 01 — GRPO clipping toy mechanism](../notebooks/01_grpo_clipping_diagnostics.ipynb) | Implement and plot unclipped vs clipped gradients; distinguish ratio clipping from gradient-norm clipping |
| 2 | Chapter 04 experiment schema + metric ingestion | Check provenance, missingness and optimizer/rollout step axes |
| 3 | Real **TRL SFTTrainer** supervised coding examples | Inspect label masks, train/eval loss, gradient flow and trusted evaluation |
| 4 | [Study 1 — trajectory SFT](07_study1_trajectory_sft.md) | Create selected-action/reflection/replay ablations; test leakage and transfer |
| 5 | GRPO groups, advantage math and TRL **GRPOTrainer** on deterministic single-turn verifiable tasks | Measure tied groups, clipping, KL, Adam updates and reward alignment |
| 6 | Repo2RLEnv + Harbor task/verifier audits | Validate oracle and no-op controls; assess difficulty, reward reliability |
| 7 | Agentic rollouts and validated GRPO integration | Track token masks, stale rollout policy, end-to-end held-out coding success |
| 8 | Deterministic APT probes, fault injection and recovery benchmark | Compare competing explanations and evaluate corrective interventions |
| 9 | [Study 2 — specialized diagnostic model](06_study2_specialized_diagnostic_model.md) | TRL SFT of diagnostic decisions; optional verified RLVR; closed-loop outcome benchmark |

**Student ownership:** plot *calculations and selection* are learner TODOs; notebooks provide publication-quality rendering scaffolds, figure export, checks and interpretation prompts. Each experiment should reuse prior evidence artifacts and document the controls, changed variable, predictions, measured outcomes, and next action.

## Design boundary
TRL SFTTrainer is for supervised demonstrations, not Harbor task instructions with no solutions. A standard one-turn TRL GRPO reward function is not an execution interface for a multi-step Harbor coding agent. Repo2RLEnv generates/verifies tasks; Harbor runs them; APT diagnoses training and evaluation evidence. We must build or validate a rollout integration before claiming multi-turn GRPO works.

## Every experiment writes a lineage
Record model/checkpoint, starting checkpoint and parent run, seed, dataset/task split and revision, reward/verifier version, training configuration, logged metric definitions and axes, held-out evaluation slices, and compute usage. Later labs reuse earlier saved artifacts. Generated task reference answers and private verifier data must never leak into policy prompts.

## Initial priority
Complete/validate the ATIF-like adapter against actual Fleet exports, then execute the TRL SFTTrainer starter with trusted held-out evaluation. The current repo does not yet contain the Fleet/ATIF code mentioned in earlier handoffs: recover it later without rewriting it.

## Sources
- https://huggingface.co/docs/trl/sft_trainer
- https://huggingface.co/docs/trl/grpo_trainer
- https://github.com/huggingface/Repo2RLEnv
- https://huggingface.github.io/Repo2RLEnv/


## Self-contained notebook structure (committed on tutor branch)

All sixteen notebooks have concept-first introduction cells, not just coding worksheets. Each begins with an explicit purpose, core definitions or math, a worked example, and a scoped exercise; each asks the learner to compute and interpret diagnostic plots. The first GRPO clipping lab contains a longer derivation; the other fifteen currently have more compact explanations that should be expanded further as the learner progresses.

**No claim of validated end-to-end Colab execution**: real TRL runs, Harbor integration, actual Study 1/2 ablations, and benchmark results remain student/research milestones.

## Notebook delivery format
**Colab-first:** Each implemented lab is a versioned \`.ipynb\` file in \`notebooks/\` with small markdown explanations, student TODO code cells, checks, research prompts, and an output artifact. Python tests and reusable diagnostic modules remain in \`tests/\` and \`src/\` respectively. Each notebook should link its prerequisite notebook and save a structured experiment report; no claims of successful execution without actual logs.

**First committed Colab:** [GRPO clipping diagnostic lab](../notebooks/01_grpo_clipping_diagnostics.ipynb). This is a controlled mathematical experiment; SFTTrainer and real GRPO training labs remain planned.


## Study 2: Specialized APT diagnostician

After the deterministic probe engine, Repo2RLEnv coding-task evaluation, and controlled fault-injection benchmark are operational, follow the [Study 2 detailed curriculum](06_study2_specialized_diagnostic_model.md). Its eight Colab-first labs cover evidence datasets, decision schema, TRL SFTTrainer, counterfactual probes, replay ablations, optional diagnostic RLVR, closed-loop evaluation, and unseen-domain testing. All eight starter **notebook files now exist on the implementation branch**, but learning exercises, experiment execution, and outcomes are still pending.

## Study 1: trajectory supervision

See [Study 1 detailed curriculum](07_study1_trajectory_sft.md) for five planned Colab labs using TRL SFTTrainer, selected decisions, reflection/replay ablations and held-out evaluation. The five **starter notebook files now exist** on the implementation branch, but their coding exercises, training, and experiments are not completed.


## Evidence-first GRPO teaching pattern

Use three progressively distinct data layers: instructor synthetic diagrams, student-owned mechanism verification, then actual TRL runs diagnosed with their available **public** metrics and independent held-out evaluations. Deep per-token loss-path recording is an **optional research extension** and cannot be assumed from callbacks. See [GRPO diagnostics contract](09_grpo_diagram_artifact_contract.md).
