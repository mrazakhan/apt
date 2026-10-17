"""Minimal ATIF-like trajectory ingestion with provenance and no future leakage.

This is an adapter contract, not a claim to support the full Harbor/ATIF schema.
Map a real provider payload explicitly before deploying.
"""
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class TrajectoryStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    action: str
    observation: str | None = None


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trajectory_id: str
    task_id: str
    repository: str
    source: str
    steps: list[TrajectoryStep]
    verified_success: bool | None = None


def parse_trajectory(record: dict[str, Any]) -> Trajectory:
    """Validate the canonical subset; fail instead of silently dropping fields."""
    return Trajectory.model_validate(record)


def decision_examples(trajectory: Trajectory) -> list[dict[str, Any]]:
    """Each target sees only earlier actions/observations, not future results."""
    results: list[dict[str, Any]] = []
    for pos, step in enumerate(trajectory.steps):
        history = [
            {"action": old.action, "observation": old.observation}
            for old in trajectory.steps[:pos]
        ]
        results.append({
            "trajectory_id": trajectory.trajectory_id,
            "task_id": trajectory.task_id,
            "repository": trajectory.repository,
            "step_index": step.index,
            "history": history,
            "target_action": step.action,
            "source": trajectory.source,
        })
    return results
