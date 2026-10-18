# APT — Autonomous Post-Training Engineer

> **North star:** Understand why post-training failed to improve **trusted held-out task success**, identify what evidence is missing, and select the next discriminating experiment or intervention.

APT is a **hands-on research and coding curriculum** that incrementally builds a post-training diagnostic system. The early focus is GRPO/RLVR, with TRL SFT and SFT→GRPO readiness as foundations. The first realistic proving ground will be verifiable coding-agent tasks from **Repo2RLEnv + Harbor**.

## Start here

1. [Connected curriculum and prerequisite map](docs/03_connected_curriculum.md) — what builds on what; each training mechanism is taught alongside its diagnostics.
2. [Colab Lab 01 — GRPO ratio clipping and gradient diagnostics](notebooks/01_grpo_clipping_diagnostics.ipynb) · [Open in Google Colab](https://colab.research.google.com/github/mrazakhan/apt/blob/main/notebooks/01_grpo_clipping_diagnostics.ipynb). This is a **toy PyTorch mechanism lab**, not an actual GRPO model-training run.
3. [Chapter 04 — Experiment representation](docs/04_experiment_representation.md) — canonical run metadata, metric provenance, checkpoint and evaluation records.
4. [Repo2RLEnv / Harbor integration plan](docs/05_repo2rlenv_integration_plan.md) — task, verifier and rollout boundaries.

**Teaching contract:** For each lab, understand the math, implement important code yourself, generate *your own* diagnostic plots from supplied production-quality scaffolds, validate with tests, interpret the evidence, and save an artifact reused in the next lab. Completed solutions must not replace the student's implementation exercises.

## Use the APT tutor directly in VS Code

APT includes a workspace custom tutor agent at [`.github/agents/apt-tutor.agent.md`](.github/agents/apt-tutor.agent.md) and a learner-approved checkpoint at [`tutor/progress.yaml`](tutor/progress.yaml).

1. Open this repository's root folder in VS Code, with a compatible authenticated chat provider such as GitHub Copilot.
2. Open Chat and select **APT Tutor** in the custom-agent dropdown.
3. Say: **"Resume my lab from tutor/progress.yaml. Preserve my TODOs and plotting calculations; ask one question at a time."**
4. At the end ask it to **propose** a progress update; approve changes before they are saved.

See the [tutor setup guide](tutor/README.md). This is an on-demand tutor with durable file-based context, not an always-running background service.

## Self-contained notebook teaching format

All 16 notebooks on `feat/vscode-apt-tutor` now include an introductory learning sequence with: motivation, domain-specific concepts and mathematical anchors, an illustrative worked example distinct from exercises, learner-owned implementation tasks, diagnostic plot assignments, and an interpretation/research memo. **Lab 01 is more extensively explained; later labs currently use shorter teaching companions.** These additions do not constitute executed training or complete standalone runtime integrations.

The existing TODOs and plot computations remain learner-owned. After any checkout, evaluate readability in Colab/VS Code and run the appropriate notebook checks; **notebook execution remains unverified**.

## Research studies

- **[Study 1 — Trajectory SFT and self-context distillation](docs/07_study1_trajectory_sft.md):** Five learner Colab **skeletons now committed**, with training/evaluation TODOs. Experiments and measured outcomes remain pending. It concerns demonstration extraction, reflections, replay and controlled SFT ablations.
- **[Study 2 — Specialized APT diagnostic model](docs/06_study2_specialized_diagnostic_model.md):** Eight learner Colab **skeletons now committed**, following future verified rollouts and recovery evaluation. They are not executed or validated experiments. Uses TRL SFTTrainer for diagnostic decisions and only considers diagnostic GRPO/RLVR after reward verification.

Study 1 trains/assesses a *coding or research-task policy* from curated examples; Study 2 trains/assesses the *diagnostician* that selects evidence-grounded probes. Their policies, supervision, and held-out benchmarks must remain distinct.

## Why these components?

| Component | Responsibility |
|---|---|
| **TRL SFTTrainer** | Train from validated prompt/completion demonstrations and inspect supervised token masks |
| **TRL GRPOTrainer** | Study group-relative advantages, clipping, KL and optimizer behavior on reward-graded completions |
| **Repo2RLEnv** | Generate/package verifiable coding tasks and their controls |
| **Harbor** | Execute and evaluate coding-agent tasks |
| **APT** | Ingest training/evaluation evidence, rank competing hypotheses, design and evaluate diagnostic probes |

**Important:** A standard one-turn GRPO completion reward is **not** a Harbor multi-turn agent-training integration. That integration is a separate validated milestone.

## Current implementation status

**On `feat/curriculum-skeletons` (proposed PR; not merged to main):**
- 2 Colab starter notebooks for TRL `SFTTrainer` and single-turn `GRPOTrainer`, with example configurations, disabled execution, explicit student exercises and diagnostic plotting templates.
- 5 Study 1 and 8 Study 2 learner notebook skeletons. They include mathematical prompts, student TODOs, publication-quality plotting renderers, checklists, and intended saved artifacts.
- Minimal Python contracts for ATIF-like trajectory ingestion, verified Harbor rollout records, deterministic diagnostic probe suggestions, and synthetic paired fault injection.
- Unit tests for the basic contracts. **Not yet run in CI or independently verified.**
- [Implementation status and precise gaps](docs/08_skeleton_implementation_status.md).

**Already on main:** experiment schema, experiment-schema tests, connected curriculum docs, research notes, and Colab 01 on GRPO clipping.

**Still missing / deliberately not claimed:** executed `SFTTrainer` or `GRPOTrainer` jobs, training metrics, verified plot calculations, original Fleet/ATIF baseline migration, actual Harbor SDK integration, multi-turn rollout→policy-gradient updates, validated diagnostic engine/closed-loop recovery and realistic fault-injection benchmark.

**Student ownership:** the coding and diagnostic notebooks are intentionally incomplete. `TODO(student)` and explicit `NotImplementedError` mark the exercises. Start with `notebooks/02_sfttrainer_baseline.ipynb`, then `notebooks/s1_01_trajectory_candidates.ipynb`; GRPO single-turn exploration is `notebooks/03_grpotrainer_verifiable.ipynb`.

## Local package setup

Python 3.11+ for the current minimal APT package; later trainer/runtime labs may need additional hardware and Python requirements. Use:

```bash
uv sync --extra dev
uv run pytest -q
```

TRL is installed in its respective Colab notebooks; the core Python package still intentionally has only minimal runtime dependencies. Harbor is not integrated.

## Project documents and research notes

- [Project vision and diagnosis philosophy](docs/00_project_vision.md)
- [Connected curriculum](docs/03_connected_curriculum.md)
- [Experiment representation](docs/04_experiment_representation.md)
- [Repo2RLEnv integration plan](docs/05_repo2rlenv_integration_plan.md)
- [Trajectory SFT / Study 1](docs/07_study1_trajectory_sft.md)
- [Specialized diagnostician / Study 2](docs/06_study2_specialized_diagnostic_model.md)
- [Research notes](docs/research_notes/README.md)

**Success criterion:** not higher training reward alone, but better **trusted held-out coding-task success after diagnostic experiments and interventions**, compared with appropriate baselines under comparable budget.
