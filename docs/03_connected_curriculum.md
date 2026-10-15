# APT — Connected TRL SFT → GRPO Tutorial Roadmap

**Status: curriculum plan, not implemented.** APT's objective is to explain why post-training does not improve trusted held-out coding-task success and choose a discriminating next experiment.

## Required teaching contract
Every chapter must have: (1) prerequisite artifact, (2) math and learning goals, (3) student-written implementation exercises, (4) runnable commands, (5) tests and expected output, (6) saved output for the next lab, (7) failure analysis, (8) open-ended research question without a preselected answer. No disconnected math exercises presented as if they were working trainer instrumentation.

## Connected sequence
| Chapter | Hands-on lab | Input → output | Gate |
|---|---|---|---|
| 04 | Canonical evidence model (already committed) | log observations → validated objects | Unit tests |
| 05 | Ingest toy trainer CSV histories | raw metrics → canonical ExperimentRun | Preserve definitions, time axes, missing values |
| 06 | **TRL SFTTrainer** on small prompt/completion coding examples | demonstrations → checkpoint, loss logs and baseline eval | Inspect completion-only masks; verify task success |
| 07 | SFT failure injection | healthy vs altered data/masks → diagnostic observations | Train/validation divergence |
| 08 | GRPO math with autograd | fixed rewards/logprobs → advantages, ratio, clipping gradients | Predict and reproduce sign changes |
| 09 | **TRL GRPOTrainer** on small deterministic verifiable *single-turn* coding-like tasks | SFT checkpoint → sampled groups, rewards, metrics | Verify all-tie and mixed-group behavior |
| 10 | GRPO optimization instrumentation | GRPO steps → clipping, KL, optimizer deltas | Distinguish raw gradient and actual update |
| 11 | Repo2RLEnv/Harbor task audit | real tasks → inspected oracle/verifier controls | Positive and negative controls |
| 12 | Real coding-agent rollout interface | Harbor task + agent → trajectories + evidence | Confirm multi-turn trainer compatibility |
| 13 | SFT→agentic GRPO | checkpoint + tasks → RL training and trusted held-out eval | Matched reproducible runs |
| 14 | Fault injection and diagnostic probes | controlled fault → blinded diagnosis | Root cause not leaked |
| 15 | Recovery benchmark | diagnosed issue → tested intervention → held-out gain | Compare against heuristics |

## Design boundary
TRL SFTTrainer is for supervised demonstrations, not Harbor task instructions with no solutions. A standard one-turn TRL GRPO reward function is not an execution interface for a multi-step Harbor coding agent. Repo2RLEnv generates/verifies tasks; Harbor runs them; APT diagnoses training and evaluation evidence. We must build or validate a rollout integration before claiming multi-turn GRPO works.

## Every experiment writes a lineage
Record model/checkpoint, starting checkpoint and parent run, seed, dataset/task split and revision, reward/verifier version, training configuration, logged metric definitions and axes, held-out evaluation slices, and compute usage. Later labs reuse earlier saved artifacts. Generated task reference answers and private verifier data must never leak into policy prompts.

## Initial priority
Implement Chapter 05 ingestion, then the first real TRL SFTTrainer coding lab. The current repo does not yet contain the Fleet/ATIF code mentioned in earlier handoffs: recover it later without rewriting it.

## Sources
- https://huggingface.co/docs/trl/sft_trainer
- https://huggingface.co/docs/trl/grpo_trainer
- https://github.com/huggingface/Repo2RLEnv
- https://huggingface.github.io/Repo2RLEnv/
