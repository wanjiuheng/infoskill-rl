from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

from infoskill.episode import TrajectoryGroup
from infoskill.evaluation import EvaluationRun


class ZstdJsonlTraceWriter:
    def __init__(self, run_directory: str | Path, *, rank: int = 0) -> None:
        try:
            import zstandard
        except ImportError as error:
            raise RuntimeError("structured trace output requires zstandard") from error
        self._zstandard = zstandard
        self.run_directory = Path(run_directory)
        self.rank = rank
        (self.run_directory / "traces").mkdir(parents=True, exist_ok=True)

    def write_training_update(
        self,
        *,
        global_update: int,
        groups: Sequence[TrajectoryGroup],
        advantages: Sequence[Sequence[float]],
    ) -> Path:
        path = self.run_directory / "traces" / f"train-update-{global_update:06d}-rank-{self.rank:03d}.jsonl.zst"
        records = []
        for group, group_advantages in zip(groups, advantages):
            if len(group.trajectories) != len(group_advantages):
                raise ValueError("trajectory and advantage counts differ")
            records.extend(
                _trajectory_record(trajectory, advantage=advantage, global_update=global_update)
                for trajectory, advantage in zip(group.trajectories, group_advantages)
            )
        self._write(path, records)
        return path

    def write_evaluation(
        self,
        *,
        checkpoint_step: int,
        run: EvaluationRun,
        split: str = "valid_seen",
    ) -> Path:
        label = split.replace("_", "-")
        path = self.run_directory / "traces" / f"{label}-{checkpoint_step:06d}-rank-{self.rank:03d}.jsonl.zst"
        records = [
            _trajectory_record(group.trajectories[0], advantage=0.0, global_update=checkpoint_step)
            for group in run.groups
        ]
        records.extend(
            {
                "schema_version": 1,
                "record_type": "infrastructure_failure",
                **_json_safe(record),  # type: ignore[arg-type]
            }
            for record in run.records
            if record.infrastructure_error is not None
        )
        self._write(path, records)
        return path

    def _write(self, path: Path, records: object) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        compressor = self._zstandard.ZstdCompressor(level=3)
        with temporary.open("wb") as raw:
            with compressor.stream_writer(raw) as stream:
                for record in records:  # type: ignore[union-attr]
                    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    stream.write(line.encode("utf-8"))
        temporary.replace(path)


class MetricLogger:
    def __init__(self, run_directory: str | Path) -> None:
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_directory / "metrics.jsonl"
        self.csv_path = self.run_directory / "metrics.csv"

    def log(self, *, step: int, phase: str, values: Mapping[str, float | int | str | bool | None]) -> None:
        record = {"step": step, "phase": phase, **values}
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        exists = self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("step", "phase", "metric", "value"))
            if not exists:
                writer.writeheader()
            for key, value in values.items():
                writer.writerow({"step": step, "phase": phase, "metric": key, "value": value})


def _trajectory_record(trajectory: object, *, advantage: float, global_update: int) -> dict[str, object]:
    task = trajectory.task  # type: ignore[attr-defined]
    steps = []
    for step in trajectory.steps:  # type: ignore[attr-defined]
        prefix_stats = _tensor_stats(step.conditioned_input.soft_prefix)
        steps.append(
            {
                "canonical_state": _json_safe(step.state_before),
                "candidate_skill_ids": list(step.conditioned_input.candidate_skill_ids),
                "policy_user_message": step.conditioned_input.user_message,
                "soft_prefix_stats": prefix_stats,
                "model_raw_response": step.generation.text,
                "finish_reason": step.generation.finish_reason,
                "response_token_ids": list(step.generation.token_ids),
                "old_token_logprobs": list(step.generation.token_logprobs),
                "prompt_token_count": step.generation.prompt_token_count,
                "action_resolution": _json_safe(step.action),
                "environment_raw_output": {
                    "observation": step.transition.raw_observation,
                    "reward": step.transition.raw_reward,
                    "done": step.transition.raw_done,
                    "won": step.transition.raw_won,
                    "info": _json_safe(step.transition.info),
                    "pre_world_state_checksum": step.transition.pre_world_state_checksum,
                    "post_world_state_checksum": step.transition.post_world_state_checksum,
                },
            }
        )
    return {
        "schema_version": 1,
        "global_update": global_update,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "split": task.split,
        "rollout_id": trajectory.rollout_id,  # type: ignore[attr-defined]
        "won": trajectory.won,  # type: ignore[attr-defined]
        "environment_done": trajectory.environment_done,  # type: ignore[attr-defined]
        "horizon_exhausted": trajectory.horizon_exhausted,  # type: ignore[attr-defined]
        "invalid_action_count": trajectory.invalid_action_count,  # type: ignore[attr-defined]
        "trajectory_reward": trajectory.reward,  # type: ignore[attr-defined]
        "group_advantage": advantage,
        "steps": steps,
    }


def _tensor_stats(value: object | None) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        tensor = value.detach().float()  # type: ignore[attr-defined]
        return {
            "mean": float(tensor.mean().item()),
            "variance": float(tensor.var(unbiased=False).item()),
            "rms": float(tensor.square().mean().sqrt().item()),
            "l2_norm": float(tensor.norm().item()),
            "max_abs": float(tensor.abs().max().item()),
        }
    except (AttributeError, TypeError, RuntimeError):
        return None


def _json_safe(value: object) -> object:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)
