import json
import math
from posttraining_engineer.training.grpo_artifacts import GRPOContext, TokenEvidence
from posttraining_engineer.training.trl_adapter import GRPOLogWriter


def test_public_metrics_writer_filters_nonfinite(tmp_path):
    w = GRPOLogWriter(tmp_path, "r")
    w.append_metrics(3, {"reward": 0.5, "kl": float("nan"), "stage": "train"})
    row = json.loads((tmp_path / "step_metrics.jsonl").read_text())
    assert row["optimizer_step"] == 3
    assert row["metrics"] == {"reward": 0.5}


def test_opt_token_writer_records_real_semantics_without_training(tmp_path):
    w = GRPOLogWriter(tmp_path, "r")
    ctx = GRPOContext(run_id="r", optimizer_step=1, rollout_step=0,
        policy_checkpoint_id="new", behavior_checkpoint_id="old",
        task_dataset_revision="d1", verifier_version="v1",
        trainer_version="fixture-only", ratio_denominator="behavior_policy",
        epsilon_low=0.2, epsilon_high=0.2,
        advantage_definition="fixture", reduction_definition="mean valid token")
    token = TokenEvidence(run_id="r", optimizer_step=1, group_id="g", completion_id="c",
        token_position=0, completion_mask=True, old_logprob=-3.,
        current_logprob=-3.+math.log(1.3), advantage=1.)
    s = w.append_optimizer_tokens(ctx, [token])
    record = json.loads((tmp_path / "optimizer_tokens.jsonl").read_text())
    assert s["active_clip_fraction"] == 1
    assert record["derived"]["clip_side"] == "upper"
