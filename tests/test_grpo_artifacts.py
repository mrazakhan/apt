import math
import pytest
from posttraining_engineer.training.grpo_artifacts import (
    GRPOContext, TokenEvidence, GroupEvidence, compute_surrogate, summarize_step,
)


def context():
    return GRPOContext(run_id="demo", optimizer_step=2, rollout_step=1,
        policy_checkpoint_id="new", behavior_checkpoint_id="old",
        task_dataset_revision="fixtures-v1", verifier_version="exact-v1",
        trainer_version="trl-unverified", ratio_denominator="behavior_policy",
        epsilon_low=.2, epsilon_high=.2, advantage_definition="group-centered",
        reduction_definition="unweighted valid token average")


def row(i, ratio, advantage, mask=True):
    return TokenEvidence(run_id="demo", optimizer_step=2, group_id="g",
        completion_id=f"c{i}", token_position=0, completion_mask=mask,
        old_logprob=-3.0, current_logprob=-3.0 + math.log(ratio),
        advantage=advantage, recorded_ratio=ratio)


def test_positive_advantage_upper_clip():
    v = compute_surrogate(row(1, 1.3, 1), context())
    assert v.active_clip and v.clip_side == "upper"
    assert v.raw_objective == pytest.approx(1.3)
    assert v.surrogate_objective == pytest.approx(1.2)


def test_negative_advantage_lower_clip():
    v = compute_surrogate(row(1, .7, -1), context())
    assert v.active_clip and v.clip_side == "lower"
    assert v.surrogate_objective == pytest.approx(-.8)


def test_outside_band_not_always_active():
    v = compute_surrogate(row(1, 1.3, -1), context())
    assert not v.active_clip
    assert v.surrogate_objective == pytest.approx(-1.3)


def test_missingness_and_denominator():
    rows = [row(1, 1.3, 1), row(2, 1.0, 1, mask=False),
        TokenEvidence(run_id="demo", optimizer_step=2, group_id="g",
            completion_id="missing", token_position=0, completion_mask=True)]
    s = summarize_step(rows, context())
    assert s["observed_rows"] == 3
    assert s["valid_supervised_tokens"] == 1
    assert s["active_clip_fraction"] == 1
    assert s["missing_or_masked_rows"] == 2


def test_lineage_mismatch_and_wrong_recorded_ratio_rejected():
    with pytest.raises(ValueError, match="lineage"):
        compute_surrogate(row(1, 1.3, 1).model_copy(update={"optimizer_step": 7}), context())
    with pytest.raises(ValueError, match="disagrees"):
        compute_surrogate(row(1, 1.3, 1).model_copy(update={"recorded_ratio": 1.1}), context())


def test_group_reward_validation():
    with pytest.raises(ValueError):
        GroupEvidence(run_id="demo", rollout_step=1, group_id="g", task_id="t",
            split="train", rewards=[float("nan")], verifier_version="v1")
