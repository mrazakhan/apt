"""Evidence-driven probes: suggestions, never invented causal conclusions."""
from enum import Enum
from pydantic import BaseModel, Field


class ProbeKind(str, Enum):
    REWARD_AUDIT = "reward_audit"
    SPLIT_LEAKAGE = "split_leakage"
    GROUP_TIES = "group_ties"
    LABEL_MASK = "label_mask"


class ProbeProposal(BaseModel):
    hypothesis: str = Field(min_length=1)
    kind: ProbeKind
    evidence: list[str]
    controlled_variable: str
    predicted_results: dict[str, str]
    estimated_cost: float = Field(ge=0)


def suggest_probes(metrics: dict[str, float | None]) -> list[ProbeProposal]:
    """A transparent, deterministic starter baseline for Study 2."""
    result = []
    if metrics.get("heldout_success") is None:
        result.append(ProbeProposal(
            hypothesis="Training reward may not generalize to trusted task success",
            kind=ProbeKind.REWARD_AUDIT,
            evidence=["Missing heldout_success"],
            controlled_variable="Keep checkpoint fixed; evaluate on trusted held-out tasks",
            predicted_results={"low": "Investigate reward mismatch", "high": "Investigate other causes"},
            estimated_cost=1,
        ))
    if metrics.get("group_tie_fraction") is not None and metrics["group_tie_fraction"] > 0.8:
        result.append(ProbeProposal(
            hypothesis="Many GRPO groups may have uninformative advantages",
            kind=ProbeKind.GROUP_TIES,
            evidence=[f"group_tie_fraction={metrics['group_tie_fraction']}"],
            controlled_variable="Hold model and tasks fixed; inspect reward variance by group",
            predicted_results={"tied": "Review reward discrimination", "mixed": "Check downstream gradient signals"},
            estimated_cost=1,
        ))
    return result
