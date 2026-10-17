# Study 2 — Train and Evaluate a Specialized APT Diagnostic Model

**Status: eight Colab student skeleton files on `feat/curriculum-skeletons`; experiments remain incomplete and unvalidated.**
**Prerequisites:** complete the SFTTrainer and GRPOTrainer mechanism/diagnostic labs, Repo2RLEnv + Harbor verification and rollout evaluation, deterministic diagnostic rules, controlled fault injection, and a held-out recovery benchmark.

## Research question
Can a model trained on *evidence-grounded diagnostic decisions* select better discriminating experiments than deterministic APT rules and a general-purpose LLM, and do its suggestions actually improve trusted held-out coding success?

**Distinct from Study 1:** Study 1 teaches demonstration construction, trajectory selection, and SFT / self-context-distillation style learning. Study 2 trains an APT *diagnostician*, not the coding-task-solving policy. Do not merge their checkpoints or training datasets.

## Connected, Colab-first labs
Every notebook must contain math/conceptual explanation, learner-coded TODOs, plotting scaffolds (**the learner implements metrics and graph construction**), verification tests, an interpretation memo, expected artifact names, and research limits.

| Lab | Learner coding task | Diagnostic graph assignment | Artifact and gate |
|---|---|---|---|
| S2-01 Evidence dataset | Join versioned run logs, evaluation slices, verifier results, injected-fault labels and probe outcomes. Mask privileged causes from model input. | Plot coverage and missingness by failure category, task domain and source run. | Audited train/validation/held-out split manifest; leakage tests. |
| S2-02 Diagnostic decision schema | Define observation, hypotheses, missing evidence, proposed probe, costs, predicted outcomes and confidence/evidence status. | Plot category frequencies and annotator disagreement. | Validated examples with attribution/provenance; no invented certainty. |
| S2-03 Supervised baseline | Use **TRL SFTTrainer** to train on verified diagnosis→probe demonstrations, compare with zero/few-shot and deterministic rules. | Build learning curves, error breakdowns, calibration/reliability plots with supplied plotting skeleton. | Checkpoint + blind held-out results, matched evaluation conditions. |
| S2-04 Counterfactual discrimination | Generate matched cases with same symptom but different causes and outcomes; evaluate if APT asks for different tests. | Plot confusion matrix and conditional probe-selection accuracy. | Counterfactual test suite and error taxonomy. |
| S2-05 Curriculum and replay | Compare selected diagnostic decisions, reflection-like examples and action replay using controlled SFT mixtures; hold training budget constant. | Plot performance/forgetting across diagnostic slices and checkpoint trajectories. | Multiple seeds + uncertainty and ablation report. |
| S2-06 Optional diagnostic RLVR | Only if a reliable verifier can score *diagnostic decisions or probe outcomes* without using hidden future information: design reward and inspect group informativeness, clipping, KL. | Plot reward-group tie rates, verifier agreement, reward vs held-out decision utility. | GRPO prototype with verifier audit; otherwise stay at SFT. |
| S2-07 Closed-loop evaluation | Let model propose a probe, execute within a fixed budget, inspect results, revise hypotheses, test intervention. | Plot diagnostic accuracy, expected vs measured information gain, cost and held-out coding-success recovery against baselines. | Repeated, seed-aware prospective results. |
| S2-08 Generalization and release gate | Evaluate across unseen repository families, model/trainer versions, and fault categories; document failures. | Plot cross-domain capability and confidence intervals, attribution and cost-benefit Pareto. | Reproducible benchmark report and limitations. |

## What constitutes a valid training example?
A single example should contain:
- **Model-visible:** run/checkpoint identifiers (appropriately anonymized), task/evaluation context, sourced observations, definition and provenance of every metric, known confounders, plausible hypotheses, and available probe budget.
- **Supervision target:** evidence-grounded diagnosis, recommended next experiment, what is held fixed/varied, predicted discriminating outcomes, and limitations.
- **Private benchmark fields, never shown in the prompt:** injected fault (when known), observed future probe result, actual recovery outcome, and grading rubric.
- **Audit fields:** human/verifier review, data lineage, dataset split, leakage checks, and whether outcome is measured or proposed.

The diagnosis should be scored on its ability to choose useful experiments, **not** on confident verbal identification of a fault from insufficient observations.

## Experimental comparison
1. Deterministic APT rules with identical observations and probe menu.
2. General-purpose base LLM prompted with the evidence and same tools/budget.
3. SFT-trained APT diagnostic model.
4. Optional SFT→RLVR diagnostic model, only after reward validation.

Primary endpoint: improvement in trusted held-out coding-task success **after** executing the proposed diagnostic probe and intervention, under a fixed experimental budget.

Secondary endpoints: fault identification accuracy when a fault is verifiably known; hypothesis ranking; discriminating-probe selection; recovery success rate; cost per useful probe; calibration; cross-domain transfer; regressions.

Use paired tasks when possible, report variability across training seeds and task families, predefine decision thresholds, and choose checkpoints on validation (not final held-out evaluation). Training accuracy or stylistic fluency is not a success criterion.

## Stop/go criteria
- Do not start supervised diagnostic-model training until validated examples and blind splits exist.
- Do not start RLVR before defining a reliable outcome/reward verifier.
- Do not declare causal root causes solely from gradient curves or observational logs.
- Do not optimize against benchmark hidden answers or leaked oracle solutions.
- Do not claim generalized success from one Repo2RLEnv dataset or from toy diagnostic probes.

## References
- Bespoke Labs, *Teaching NVIDIA's Nemotron ML Engineering*: https://bespokelabs.ai/blog/teaching-nvidias-nemotron-ml-engineering
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- TRL GRPOTrainer: https://huggingface.co/docs/trl/grpo_trainer
- Repo2RLEnv: https://github.com/huggingface/Repo2RLEnv
