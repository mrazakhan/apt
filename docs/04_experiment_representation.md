# 04 — Representing a post-training experiment

> **Research question:** Why did an SFT or GRPO experiment fail to improve trusted held-out success?

## 1. What we are building

A small, validated representation of an experiment that can serve as the common input for future SFT, GRPO, verifier, and infrastructure diagnostics.

**Files in this chapter**

- `src/posttraining_engineer/training/models.py`
- `tests/test_training_models.py`
- `pyproject.toml`

The repository is intentionally being bootstrapped incrementally. The earlier Fleet/ATIF migration has **not** been committed yet. We will integrate it as a trajectory evidence subsystem later, without replacing its working logic.

## 2. Why a data model comes before an LLM diagnostic agent

Imagine two runs with `kl=0.02`. Are they equivalent? Not necessarily.

- One may measure KL between the policy and a fixed SFT reference.
- Another may measure an approximation between the current and *behavior* policy.
- One metric may use optimizer steps on its horizontal axis, the other rollout steps.
- An unknown value is different from a measured zero.

If APT erases these distinctions, even a very capable language model will generate unreliable explanations.

The **north star** remains held-out success, not training reward. GRPO reward, loss, KL, gradient norms and group diversity are intermediate evidence.

## 3. RL concepts: SFT versus GRPO

**SFT** fits a policy to demonstrated target tokens. A conventional token-level cross-entropy objective is

\[
\mathcal{L}_{\mathrm{SFT}}(\theta) = - \mathbb{E}_{(x,y)}\sum_{t \in \mathrm{target}} \log \pi_\theta(y_t\mid x,y_{<t}).
\]

- \(x\): task prompt
- \(y_t\): target token at position \(t\)
- \(\pi_\theta\): model's token distribution
- \(\mathrm{target}\): positions actually supervised by the loss mask

**GRPO** samples a group of completions for a prompt, scores them, then builds a relative learning signal. A common illustrative group-normalized advantage is

\[
A_i = \frac{r_i - \bar r_G}{s_G + \epsilon}.
\]

- \(r_i\): reward for completion \(i\)
- \(\bar r_G\): mean reward in the group
- \(s_G\): group reward standard deviation
- \(\epsilon\): numerical safeguard

Different GRPO implementations vary in normalization, clipping, KL treatment and aggregation. We therefore record the *metric definition*, not just a name.

**Thought experiment:** A group of eight completions receives `[0,0,0,0,0,0,0,0]`. The group cannot distinguish good from bad samples by reward alone. But is the task too hard, the verifier broken, or the model simply unlucky? The numbers do not tell you which. That is an **observation**, not a **root cause**.

## 4. Dependencies

We use:

- Python 3.11 or newer
- `pydantic` for runtime validation and JSON-compatible models
- `pytest` for tests
- `uv` as the environment manager

Install uv following its official documentation: https://docs.astral.sh/uv/

From the repository root:

```bash
uv sync --extra dev
uv run pytest -q
```

We intentionally do **not** add pandas, dashboards, or agent frameworks yet.

## 5. Read the implementation piece by piece

Open `src/posttraining_engineer/training/models.py`.

### TrainingStage

`TrainingStage` distinguishes `sft` and `grpo`. This enables stage-specific diagnostics rather than applying an RL heuristic to SFT.

### StepAxis

`StepAxis` distinguishes optimizer steps from rollout steps. A value at optimizer step 20 must not be compared blindly to a value at rollout step 20.

### MetricPoint

Every metric includes:

| Field | Why it matters |
|---|---|
| `name` | Names the measured signal |
| `value` | Stores the scalar observation |
| `step` and `axis` | Preserve the time/compute coordinate |
| `definition` | Disambiguates KL or reward variants |
| `source` | Makes observations auditable |
| `unit` | Allows meaningful future comparisons |

The `isfinite` validator rejects NaN and infinity. That prevents corrupted logs from quietly entering diagnostic calculations.

### EvaluationResult

Evaluation results are stored **separately** from training metrics. We retain the checkpoint, evaluation slice, metric, sample count and source.

A GRPO training reward of 0.9 is not the same as a trusted held-out success rate of 90%.

### ExperimentRun

A run identifies its training stage, model, starting checkpoint, and optional `parent_run_id`. The parent establishes a link such as SFT run → GRPO run.

It deliberately does not attempt to infer causal readiness from that link.

## 6. Run an example

Create `examples/first_run.py` or execute the following in a Python shell:

```python
from posttraining_engineer.training.models import (
    ExperimentRun, MetricPoint, StepAxis, TrainingStage,
)

run = ExperimentRun(
    run_id="grpo-demo",
    stage=TrainingStage.GRPO,
    model_id="toy-model",
    initial_checkpoint_id="sft-checkpoint",
    metrics=[
        MetricPoint(
            name="group_tie_fraction",
            value=0.91,
            step=10,
            axis=StepAxis.ROLLOUT,
            definition="fraction of rollout groups with identical rewards",
            source="training.csv:10",
        )
    ],
)
print(run.model_dump_json(indent=2))
print("KL observed:", bool(run.observations("kl")))
```

**Expected result:** The JSON preserves the group-tie observation, and `KL observed: False`.

That is **not** the same as reporting KL = 0.

## 7. Test the invariants

Run:

```bash
uv run pytest tests/test_training_models.py -q
```

The tests verify:
- a measured zero reward is preserved
- a missing KL observation stays missing
- non-finite values are rejected
- missing metric definitions are rejected
- negative step indices are rejected
- unexpected fields are rejected
- optimizer and rollout axes can be queried separately

## 8. Common failure modes

**`ModuleNotFoundError`:** Run `uv sync --extra dev` in the repository root, then use `uv run`.

**Validation error for NaN:** This is intentional. Inspect the raw log rather than turning NaN into zero.

**Two values named `kl` look different:** Check their definitions, axes, sources, and trainer versions. Never average mismatched metrics.

**No optimizer or system metrics:** Do not infer they're zero. Add adapters that report missing data explicitly.

## 9. Your research exercises

1. **Interpretation:** A GRPO group has eight rewards, all equal to 1. How does this differ from eight zeros? What is the same about their relative advantages?
2. **Debugging:** A run has improving training reward but declining held-out success. Name three distinct hypotheses and one measurement that could help separate them.
3. **Data integrity:** Why should the API reject NaN but accept a real reward of 0?
4. **Engineering:** Add a test proving that two `MetricPoint` objects can have the same `name` but different `definition` fields without being conflated.

For each exercise, explain *why* your answer follows from the training objective, not just what the answer is.

## 10. What this chapter does **not** claim

- These models do not yet ingest W&B, TensorBoard or trainer-native files.
- There are no automated root-cause diagnoses yet.
- The `config` and `metadata` fields are deliberately small first-pass representations.
- Real GRPO analysis will require group-level and sometimes token-level records, not just scalar history.
- No claim has been made that a comprehensive public SFT → GRPO run history is available.

## 11. What follows

**Chapter 05: Training Metric Ingestion** will introduce source adapters and metric-name mappings. We will learn how a logger's `reward_std` can differ from per-group reward standard deviation, preserve raw provenance, and turn external logs into this canonical representation without inventing missing metrics.

After that, Chapter 06 will compute **observations only**; Chapter 07 will investigate GRPO-specific failure mechanisms.

**Checkpoint question:** Before advancing, can you explain why a metric's mathematical definition and time axis are both part of the data rather than mere documentation?
