"""TRL GRPO observation adapter: public trainer logs + explicit token batches.

Public TrainerCallback.on_log is non-invasive and captures actual metrics, but
does not expose per-token current-policy logprobs. These require an opt-in,
version-reviewed instrumentation hook within the trainer loss calculation.
"""
from __future__ import annotations

import json
from pathlib import Path
from math import isfinite
from typing import Any

from .grpo_artifacts import GRPOContext, TokenEvidence, compute_surrogate, summarize_step


class GRPOLogWriter:
    """Writes versioned JSONL with no fake missing metrics."""
    def __init__(self, directory: str | Path, run_id: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def append_metrics(self, step: int, logs: dict[str, Any]) -> None:
        metrics = {key: float(value) for key, value in logs.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)
                   and isfinite(float(value))}
        entry = {"record_type": "trl_public_metrics", "run_id": self.run_id,
                 "optimizer_step": int(step), "metrics": metrics}
        with (self.directory / "step_metrics.jsonl").open("a") as out:
            out.write(json.dumps(entry, sort_keys=True) + "\n")

    def append_optimizer_tokens(self, ctx: GRPOContext, rows: list[TokenEvidence]) -> dict:
        """Store already-captured optimizer tokens. No policy forward passes here."""
        if ctx.run_id != self.run_id:
            raise ValueError("Writer/context run mismatch")
        summary = summarize_step(rows, ctx)
        target = self.directory / "optimizer_tokens.jsonl"
        with target.open("a") as out:
            for row in rows:
                value = compute_surrogate(row, ctx)
                out.write(json.dumps({
                    "record_type": "optimizer_token",
                    "context": ctx.model_dump(),
                    "token": row.model_dump(),
                    "derived": value.model_dump() if value else None,
                }, sort_keys=True) + "\n")
        return summary


def make_public_metrics_callback(writer: GRPOLogWriter):
    """Return a normal Transformers callback without hard requiring TRL at import."""
    from transformers import TrainerCallback

    class EvidenceCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs:
                writer.append_metrics(state.global_step, logs)
            return control

    return EvidenceCallback()


def attach_public_metrics(trainer, output_dir: str | Path, run_id: str) -> GRPOLogWriter:
    """Attach callback only; no GRPOTrainer method override or training change."""
    writer = GRPOLogWriter(output_dir, run_id)
    trainer.add_callback(make_public_metrics_callback(writer))
    return writer
