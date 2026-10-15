import pytest
from pydantic import ValidationError

from posttraining_engineer.training.models import (
    EvaluationResult,
    ExperimentRun,
    MetricPoint,
    StepAxis,
    TrainingStage,
)


def test_grpo_run_keeps_reward_and_eval_separate():
    run = ExperimentRun(
        run_id="grpo-001",
        stage=TrainingStage.GRPO,
        model_id="example/model",
        initial_checkpoint_id="sft-ckpt-1",
        parent_run_id="sft-001",
        metrics=[
            MetricPoint(
                name="reward_mean",
                value=0.0,
                step=12,
                axis=StepAxis.ROLLOUT,
                definition="mean verified completion reward per rollout",
                source="trainer.csv:row12",
            )
        ],
        evaluations=[
            EvaluationResult(
                checkpoint_id="grpo-ckpt-12",
                task_slice="held_out_coding",
                metric_name="success_rate",
                value=0.25,
                sample_count=100,
                source="eval.json",
            )
        ],
    )
    assert run.observations("reward_mean")[0].value == 0.0
    assert run.observations("kl") == []
    assert run.evaluations[0].value == 0.25


def test_nonfinite_metric_is_rejected():
    with pytest.raises(ValidationError):
        MetricPoint(
            name="kl",
            value=float("nan"),
            step=0,
            axis="optimizer_step",
            definition="reference policy KL",
            source="test",
        )


def test_missing_definition_is_rejected():
    with pytest.raises(ValidationError):
        MetricPoint(name="kl", value=0.1, step=2, axis="optimizer_step", definition="", source="trainer")


def test_negative_step_is_rejected():
    with pytest.raises(ValidationError):
        MetricPoint(name="loss", value=1.0, step=-1, axis="optimizer_step", definition="SFT train loss", source="log")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        ExperimentRun(
            run_id="sft-1", stage="sft", model_id="m", initial_checkpoint_id="base",
            imaginary_metric=123,
        )


def test_axis_specific_query():
    run = ExperimentRun(
        run_id="grpo-2", stage="grpo", model_id="m", initial_checkpoint_id="sft",
        metrics=[
            MetricPoint(name="reward", value=0.4, step=7, axis="rollout_step", definition="mean", source="a"),
            MetricPoint(name="reward", value=0.3, step=2, axis="optimizer_step", definition="mean", source="b"),
        ],
    )
    assert len(run.observations("reward", StepAxis.ROLLOUT)) == 1
    assert run.observations("reward", StepAxis.ROLLOUT)[0].step == 7
