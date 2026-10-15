# APT — Autonomous Post-Training Engineer

APT has one north-star goal:

> **Understand why a post-training experiment failed to improve held-out accuracy and actively determine what experiment or intervention should happen next.**

APT is intended to evolve from a diagnostic system into an autonomous research engineer that can ingest post-training evidence, form competing causal hypotheses, design diagnostic probes, recommend the next experiment, and measure whether the intervention improves trusted held-out performance.

The initial focus is **GRPO / RLVR**, with SFT and SFT→GRPO readiness as supporting areas.

## Research notes

External papers and systems that influence APT are tracked under [docs/research_notes](docs/research_notes/README.md).

- **Research notes** preserve what we learned from external work.
- **Architecture/tutorial docs** record what APT has actually adopted.

See [Project Vision](docs/00_project_vision.md).

## Guided implementation tutorial

APT is being built **chapter by chapter** as an RL/post-training learning project.

- [Chapter 04 — Representing a Post-Training Experiment](docs/04_experiment_representation.md): SFT vs GRPO, provenance-aware metrics, held-out evaluations, checkpoint lineage and executable validation tests.

Start with:

```bash
uv sync --extra dev
uv run pytest -q
```

**Migration status:** The earlier Fleet/ATIF/Harbor baseline is still pending migration. The present implementation is a minimal independent foundation; it does not replace the existing trajectory processing code. See the tutorial for boundaries and exercises.


## Connected training curriculum (SFT → GRPO → coding-agent diagnostics)

Start with the [connected learning roadmap](docs/03_connected_curriculum.md), then [Chapter 04](docs/04_experiment_representation.md). Follow-up labs deliberately reuse each other's artifacts, beginning with real TRL SFTTrainer training before GRPO.

[Repo2RLEnv integration contract](docs/05_repo2rlenv_integration_plan.md) describes how verifiable coding tasks, Harbor rollout execution, TRL training, and APT diagnostics fit together; it is a plan, **not an implemented bridge**.

The first planned training lab is TRL SFTTrainer with prompt-completion loss masking and held-out task evaluation. The first multi-step agentic GRPO experiment is gated on a validated Harbor-to-training rollout interface.

**Status:** Chapter 04 evidence models and tests committed; downstream training labs, Harbor integration, and the earlier Fleet/ATIF baseline migration remain pending. Run \`uv sync --extra dev && uv run pytest -q\` locally; CI/test execution has not been independently confirmed.
