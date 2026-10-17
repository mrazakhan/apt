import pytest

from posttraining_engineer.ingest.atif import Trajectory, TrajectoryStep, decision_examples
from posttraining_engineer.bridge.harbor import VerifiedRollout, to_training_candidate
from posttraining_engineer.diagnostics.probes import ProbeKind, suggest_probes
from posttraining_engineer.benchmarks.faults import EvidenceCase, inject_fault


def test_trajectory_does_not_see_future_observation():
    traj = Trajectory(
        trajectory_id="t1", task_id="task", repository="repo", source="fixture",
        steps=[TrajectoryStep(index=0, action="inspect", observation="secret later"),
               TrajectoryStep(index=1, action="edit", observation="result")],
    )
    examples = decision_examples(traj)
    assert examples[0]["history"] == []
    assert "result" not in str(examples[1]["history"])
    assert examples[1]["history"][0]["action"] == "inspect"


def test_unverified_rollout_is_rejected():
    rollout = VerifiedRollout(task_id="x", trajectory_id="y", policy_checkpoint="ckpt",
        verifier_version="v1", rollout_source="fixture",
        actions=["a"], observations=["o"], reward=1.0, trusted=False)
    with pytest.raises(ValueError):
        to_training_candidate(rollout)


def test_diagnostic_baseline_keeps_missing_distinct_from_zero():
    kinds = {x.kind for x in suggest_probes({"heldout_success": None, "group_tie_fraction": 0.9})}
    assert kinds == {ProbeKind.REWARD_AUDIT, ProbeKind.GROUP_TIES}
    kinds_zero = {x.kind for x in suggest_probes({"heldout_success": 0.0, "group_tie_fraction": 0.1})}
    assert ProbeKind.REWARD_AUDIT not in kinds_zero


def test_faults_are_paired_and_isolated():
    case = EvidenceCase("x", "ckpt", 0.5, 0.1)
    changed = inject_fault(case, "reward_ties")
    assert changed.group_tie_fraction == 1.0
    assert case.group_tie_fraction == 0.1
