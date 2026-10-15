"""Canonical, deliberately small representation of SFT/GRPO experiment evidence.

Metric values are *observations*, not root-cause labels. Missing != zero.
Different KL definitions and step axes must never be silently combined.
"""

from enum import Enum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingStage(str, Enum):
    SFT = "sft"
    GRPO = "grpo"


class StepAxis(str, Enum):
    OPTIMIZER = "optimizer_step"
    ROLLOUT = "rollout_step"


class MetricPoint(BaseModel):
    """One scalar observation, including provenance and precise metric meaning."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: float
    step: int = Field(ge=0)
    axis: StepAxis
    unit: str | None = None
    definition: str = Field(min_length=1)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_value(self) -> "MetricPoint":
        if not isfinite(self.value):
            raise ValueError("metric values must be finite")
        return self


class EvaluationResult(BaseModel):
    """Trusted held-out performance; not interchangeable with training reward."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = Field(min_length=1)
    task_slice: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    value: float
    sample_count: int = Field(gt=0)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_value(self) -> "EvaluationResult":
        if not isfinite(self.value):
            raise ValueError("evaluation values must be finite")
        return self


class ExperimentRun(BaseModel):
    """Evidence from a *single* stage and checkpoint lineage."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    stage: TrainingStage
    model_id: str = Field(min_length=1)
    initial_checkpoint_id: str = Field(min_length=1)
    parent_run_id: str | None = None
    config: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    metrics: list[MetricPoint] = Field(default_factory=list)
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def observations(self, name: str, axis: StepAxis | None = None) -> list[MetricPoint]:
        """Return ordered observations without assuming a step-axis conversion."""
        return sorted(
            (m for m in self.metrics if m.name == name and (axis is None or m.axis == axis)),
            key=lambda m: m.step,
        )
