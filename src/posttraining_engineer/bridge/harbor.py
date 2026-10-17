"""Harbor rollout to training-record validation boundary.

Explicitly does NOT launch Harbor or implement multi-turn GRPO.
This contract prevents conflating task reward with one-turn text reward.
"""
from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerifiedRollout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    policy_checkpoint: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    rollout_source: str = Field(min_length=1)
    actions: list[str] = Field(min_length=1)
    observations: list[str]
    reward: float | None = None
    trusted: bool = False

    @model_validator(mode="after")
    def validate_steps(self) -> "VerifiedRollout":
        if len(self.observations) != len(self.actions):
            raise ValueError("Each action needs one corresponding observation")
        return self


def to_training_candidate(rollout: VerifiedRollout) -> dict:
    """Only trusted, independently evaluated rollouts become candidates."""
    if not rollout.trusted or rollout.reward is None:
        raise ValueError("Cannot train from unverified or unrewarded rollout")
    return rollout.model_dump()


def harbor_runner(*args, **kwargs):
    """Adapter placeholder: supply real Harbor SDK, sandbox and verifier."""
    raise NotImplementedError("Integrate Harbor execution and verifier separately")
