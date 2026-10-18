"""Versioned GRPO rollout/optimizer evidence contract for theory-to-data plots.

A token ratio is meaningful only for a *specified* policy pair and loss definition.
Raw rollout generation usually does not contain current-policy log probabilities:
those must be computed at the optimization step. Never invent missing traces.
"""
from __future__ import annotations
from math import exp, isclose, isfinite, log
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "grpo-diagnostics/v1"


class GRPOContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["grpo-diagnostics/v1"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    optimizer_step: int = Field(ge=0)
    rollout_step: int = Field(ge=0)
    policy_checkpoint_id: str = Field(min_length=1)
    behavior_checkpoint_id: str = Field(min_length=1)
    reference_checkpoint_id: str | None = None
    task_dataset_revision: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    trainer_version: str = Field(min_length=1)
    ratio_denominator: Literal["behavior_policy", "stored_old_policy"]
    clipping_variant: Literal["two_sided_standard"] = "two_sided_standard"
    epsilon_low: float = Field(gt=0, lt=1)
    epsilon_high: float = Field(gt=0, lt=1)
    advantage_definition: str = Field(min_length=1)
    reduction_definition: str = Field(min_length=1)


class TokenEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1)
    optimizer_step: int = Field(ge=0)
    group_id: str = Field(min_length=1)
    completion_id: str = Field(min_length=1)
    token_position: int = Field(ge=0)
    completion_mask: bool
    old_logprob: float | None = None
    current_logprob: float | None = None
    advantage: float | None = None
    reward: float | None = None
    # Optional logged value: cross-check against the reconstructed theoretical result.
    recorded_ratio: float | None = None

    @model_validator(mode="after")
    def finite_fields(self):
        for key in ("old_logprob", "current_logprob", "advantage", "reward", "recorded_ratio"):
            v = getattr(self, key)
            if v is not None and not isfinite(v):
                raise ValueError(f"{key} must be finite")
        if self.recorded_ratio is not None and self.recorded_ratio <= 0:
            raise ValueError("recorded_ratio must be positive")
        return self


class GroupEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    rollout_step: int = Field(ge=0)
    group_id: str
    task_id: str
    split: Literal["train", "validation", "heldout"]
    rewards: list[float] = Field(min_length=1)
    verifier_version: str
    @model_validator(mode="after")
    def finite_rewards(self):
        if not all(isfinite(v) for v in self.rewards):
            raise ValueError("group rewards must be finite")
        return self


class StepEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    optimizer_step: int = Field(ge=0)
    gradient_global_norm_before_clipping: float | None = Field(default=None, ge=0)
    gradient_global_norm_after_clipping: float | None = Field(default=None, ge=0)
    measured_kl: float | None = None
    kl_definition: str | None = None
    heldout_success: float | None = Field(default=None, ge=0, le=1)
    heldout_sample_count: int | None = Field(default=None, gt=0)


class TokenSurrogate(BaseModel):
    ratio: float
    raw_objective: float
    clipped_branch: float
    surrogate_objective: float
    active_clip: bool
    clip_side: Literal["upper", "lower", "none"]


def compute_surrogate(row: TokenEvidence, ctx: GRPOContext) -> TokenSurrogate | None:
    """Derived values for *unweighted* two-sided PPO-style max surrogate.

    Ineligible/missing/masked rows are excluded, never filled with zero.
    This is NOT a claim of matching arbitrary TRL implementation-specific losses.
    """
    if row.run_id != ctx.run_id or row.optimizer_step != ctx.optimizer_step:
        raise ValueError("Token and context lineage mismatch")
    if not row.completion_mask:
        return None
    if row.old_logprob is None or row.current_logprob is None or row.advantage is None:
        return None
    ratio = exp(row.current_logprob - row.old_logprob)
    if not isfinite(ratio) or ratio <= 0:
        raise ValueError("invalid probability ratio")
    if row.recorded_ratio is not None and not isclose(ratio, row.recorded_ratio, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("recorded ratio disagrees with log probabilities")
    low, high = 1 - ctx.epsilon_low, 1 + ctx.epsilon_high
    raw = ratio * row.advantage
    clipped = min(max(ratio, low), high) * row.advantage
    active = (row.advantage > 0 and ratio > high) or (row.advantage < 0 and ratio < low)
    side = "upper" if active and row.advantage > 0 else ("lower" if active else "none")
    return TokenSurrogate(ratio=ratio, raw_objective=raw, clipped_branch=clipped,
                          surrogate_objective=min(raw, clipped), active_clip=active, clip_side=side)


def summarize_step(rows: list[TokenEvidence], ctx: GRPOContext) -> dict:
    """Denominator is valid supervised tokens, not all rollout tokens."""
    derived = [compute_surrogate(r, ctx) for r in rows]
    valid = [x for x in derived if x is not None]
    n = len(valid)
    return {
        "run_id": ctx.run_id, "optimizer_step": ctx.optimizer_step,
        "observed_rows": len(rows), "valid_supervised_tokens": n,
        "missing_or_masked_rows": len(rows) - n,
        "active_clip_fraction": sum(v.active_clip for v in valid) / n if n else None,
        "mean_absolute_log_ratio": sum(abs(log(v.ratio)) for v in valid) / n if n else None,
    }
