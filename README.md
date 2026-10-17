# APT — Autonomous Post-Training Engineer

> **North star:** Understand why post-training failed to improve **trusted held-out task success**, identify what evidence is missing, and select the next discriminating experiment or intervention.

APT is a **hands-on research and coding curriculum** that incrementally builds a post-training diagnostic system. The early focus is GRPO/RLVR, with TRL SFT and SFT→GRPO readiness as foundations. The first realistic proving ground will be verifiable coding-agent tasks from **Repo2RLEnv + Harbor**.

## Start here

1. [Connected curriculum and prerequisite map](docs/03_connected_curriculum.md) — what builds on what; each training mechanism is taught alongside its diagnostics.
2. [Colab Lab 01 — GRPO ratio clipping and gradient diagnostics](notebooks/01_grpo_clipping_diagnostics.ipynb) · [Open in Google Colab](https://colab.research.google.com/github/mrazakhan/apt/blob/main/notebooks/01_grpo_clipping_diagnostics.ipynb). This is a **toy PyTorch mechanism lab**, not an actual GRPO model-training run.
3. [Chapter 04 — Experiment representation](docs/04_experiment_representation.md) — canonical run metadata, metric provenance, checkpoint and evaluation records.
4. [Repo2RLEnv / Harbor integration plan](docs/05_repo2rlenv_integration_plan.md) — task, verifier and rollout boundaries.

**Teaching contract:** For each lab, understand the math, implement important code yourself, generate *your own* diagnostic plots from supplied production-quality scaffolds, validate with tests, interpret the evidence, and save an artifact reused in the next lab. Completed solutions must not replace the student's implementation exercises.

## Research studies

- **Study 1 — Trajectory SFT and self-context distillation:** planned after the first TRL SFTTrainer lab, before agentic GRPO. This is currently a **roadmap topic**, not a dedicated implemented Colab or full study curriculum. It concerns demonstration extraction, reflections, replay and controlled SFT ablations.
- **[Study 2 — Specialized APT diagnostic model](docs/06_study2_specialized_diagnostic_model.md):** detailed **planned** eight-lab curriculum. Follows verified coding rollouts, deterministic probes, fault injection and recovery evaluation. Uses TRL SFTTrainer for diagnostic decisions and only considers diagnostic GRPO/RLVR after reward verification.

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

**Committed:** experiment representation models, their tests, Chapter 04 tutorial, connected curriculum, Repo2RLEnv integration plan, Study 2 curriculum, Lab 01 student Colab, research notes.

**Not yet implemented:** TRL SFTTrainer/GRPOTrainer training runs, the Study 1 Colab/curriculum, Study 2 Colabs, real Harbor rollout-to-training bridge, the diagnostic engine, fault-injection benchmark, and migration of the previous Fleet/ATIF trajectory ingestion baseline.

Notebook 01 intentionally contains learner TODOs and cannot be run end-to-end without completing them. Tests have been committed, but a passing CI run is **not independently verified**. Avoid interpreting planned experiments as measured outcomes.

## Local package setup

Python 3.11+ for the current minimal APT package; later trainer/runtime labs may need additional hardware and Python requirements. Use:

```bash
uv sync --extra dev
uv run pytest -q
```

Dependencies for the later TRL/Harbor labs are not yet installed by the current minimal `pyproject.toml`.

## Project documents and research notes

- [Project vision and diagnosis philosophy](docs/00_project_vision.md)
- [Connected curriculum](docs/03_connected_curriculum.md)
- [Experiment representation](docs/04_experiment_representation.md)
- [Repo2RLEnv integration plan](docs/05_repo2rlenv_integration_plan.md)
- [Specialized diagnostician / Study 2](docs/06_study2_specialized_diagnostic_model.md)
- [Research notes](docs/research_notes/README.md)

**Success criterion:** not higher training reward alone, but better **trusted held-out coding-task success after diagnostic experiments and interventions**, compared with appropriate baselines under comparable budget.
