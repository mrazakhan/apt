# Coding skeleton implementation map

The `feat/curriculum-skeletons` branch provides **starter contracts**, not validated research outcomes.

| Component | Status | Next student task |
|---|---|---|
| ATIF-like trajectory adapter | Minimal canonical adapter + tests | Map and test actual Fleet/ATIF fields and trajectory edge cases |
| Harbor rollout bridge | Typed verified-rollout boundary | Connect Harbor SDK, sandbox, verifier, IDs and multi-turn action execution |
| Diagnostic engine | Deterministic, transparent baseline | Build causal discrimination probes and observed/proposed metric registry |
| Fault benchmark | Synthetic paired record mutation | Inject training/rollout-level faults, verify detection and recovery |
| TRL SFTTrainer / GRPOTrainer | Colab skeletons | Run seeded GPU jobs, inspect masks, clipping/KL and held-out outcomes |
| Study 1 Colabs | Five guided skeletons | Implement data selection, leakage checks, ablations and plots |
| Study 2 Colabs | Eight guided skeletons | Validate diagnostician labels, counterfactual controls and generalization |

## Non-negotiable boundaries
- No private verifier content, future outcome, or reference solution in model-visible inputs.
- A Harbor verifier-rewarded trajectory is **not** automatically a one-turn TRL GRPO sample.
- Synthetic fault detection does not establish training intervention recovery.
- The reusable plot scaffold handles visuals, but learners write metric calculations, data selection, plot interpretations, and tests.
- The Colabs are intentionally incomplete where TODOs are part of the curriculum.

## Run locally
```bash
uv sync --extra dev
uv run pytest -q
```
See `notebooks/02_sfttrainer_baseline.ipynb` and `notebooks/03_grpotrainer_verifiable.ipynb`.
