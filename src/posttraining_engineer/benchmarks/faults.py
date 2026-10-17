"""Controlled synthetic fault-injection benchmark; NOT an end-to-end recovery benchmark."""
from dataclasses import dataclass, replace
from typing import Literal

Fault = Literal["missing_eval", "reward_ties", "stale_checkpoint"]


@dataclass(frozen=True)
class EvidenceCase:
    case_id: str
    checkpoint_id: str
    heldout_success: float | None
    group_tie_fraction: float | None


def inject_fault(case: EvidenceCase, fault: Fault) -> EvidenceCase:
    if fault == "missing_eval":
        return replace(case, heldout_success=None)
    if fault == "reward_ties":
        return replace(case, group_tie_fraction=1.0)
    if fault == "stale_checkpoint":
        return replace(case, checkpoint_id=case.checkpoint_id + "-stale")
    raise ValueError(f"Unrecognized fault: {fault}")


def paired_cases(case: EvidenceCase, faults: list[Fault]) -> list[tuple[Fault, EvidenceCase]]:
    return [(fault, inject_fault(case, fault)) for fault in faults]
