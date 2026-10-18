# APT project instructions for VS Code GitHub Copilot

Follow root [AGENTS.md](../AGENTS.md). The APT project is an educational research environment, not a turnkey autonomous coding tool.

Repository map:
- `docs/03_connected_curriculum.md`: connected sequence, prerequisites, evidence gates.
- `docs/07_study1_trajectory_sft.md`: Study 1, learner trains task-solving policy from selected examples.
- `docs/06_study2_specialized_diagnostic_model.md`: Study 2, learner trains a diagnostician; distinct policy and labels.
- `notebooks/`: Colab-first, intentionally incomplete exercises.
- `src/posttraining_engineer/`: shared contracts and diagnostic helpers.
- `tutor/progress.yaml`: persistent *declared*, not inferred, learner progress.

Do not silently fill student TODOs or generate diagnostic metrics on behalf of learner. Give progressive hints, conceptual explanations, validation tests, and plot styling scaffolds. Check provenance and held-out evaluation before discussing conclusions. Ask permission before progress updates and commits.
