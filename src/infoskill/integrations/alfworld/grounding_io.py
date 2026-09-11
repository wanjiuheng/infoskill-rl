from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from infoskill.domain.state import AgentHistoryEntry, CanonicalAgentState

from .expert_replay import ExpertReplayResult, GroundingSample


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GroundingManifest:
    schema_version: int
    source_split: str
    total_games: int
    successful_games: int
    quarantined_games: int
    success_coverage: float
    over_persist_horizon: int
    over_persist_horizon_rate: float
    task_type_counts: Mapping[str, Mapping[str, int]]
    quarantine_reasons: Mapping[str, int]
    trajectory_lengths: Mapping[str, float | int]
    source_checksums: Mapping[str, str]
    code_revision: str
    expert_name: str
    max_replay_steps: int
    persist_horizon: int
    formal_gate_passed: bool
    formal_gate_failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroundingDataset:
    """Validated train-only expert states grouped by source game."""

    root: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    samples_by_game: Mapping[str, tuple[GroundingSample, ...]]

    @property
    def game_count(self) -> int:
        return len(self.samples_by_game)

    @property
    def sample_count(self) -> int:
        return sum(len(samples) for samples in self.samples_by_game.values())

    @classmethod
    def load(cls, directory: str | Path) -> "GroundingDataset":
        root = Path(directory).expanduser().resolve()
        manifest_path = root / "manifest.json"
        samples_path = root / "grounding_samples.jsonl"
        if not manifest_path.is_file() or not samples_path.is_file():
            raise FileNotFoundError(
                "grounding dataset requires manifest.json and grounding_samples.jsonl"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("grounding manifest must be a JSON object")
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported grounding manifest schema")
        if manifest.get("source_split") != "train":
            raise ValueError("grounding dataset must be train-only")
        if manifest.get("formal_gate_passed") is not True:
            failures = manifest.get("formal_gate_failures", [])
            raise ValueError(f"grounding formal gate did not pass: {failures}")

        grouped: dict[str, list[GroundingSample]] = defaultdict(list)
        with samples_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    sample = _decode_grounding_sample(payload)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid grounding sample at line {line_number}: {error}"
                    ) from error
                if sample.state.split != "train":
                    raise ValueError("grounding dataset must be train-only")
                if sample.expert_action not in sample.state.admissible_commands:
                    raise ValueError(
                        "grounding expert action must be an exact admissible command"
                    )
                if not sample.state.candidate_skill_ids:
                    raise ValueError("grounding state requires candidate skill IDs")
                grouped[sample.state.task_id].append(sample)
        if not grouped:
            raise ValueError("grounding dataset contains no successful samples")
        expected_games = manifest.get("successful_games")
        if isinstance(expected_games, int) and expected_games != len(grouped):
            raise ValueError(
                "grounding successful game count does not match the sample file"
            )
        return cls(
            root=root,
            manifest=manifest,
            manifest_sha256=sha256_file(manifest_path),
            samples_by_game={
                task_id: tuple(samples)
                for task_id, samples in sorted(grouped.items())
            },
        )

    def sample_games(self, *, game_count: int, seed: int) -> tuple[GroundingSample, ...]:
        """Choose distinct games, then one uniformly random state per game."""

        if game_count <= 0:
            raise ValueError("grounding game_count must be positive")
        if seed < 0:
            raise ValueError("grounding seed must be non-negative")
        if game_count > self.game_count:
            raise ValueError(
                f"requested {game_count} grounding games from only {self.game_count}"
            )
        generator = random.Random(seed)
        task_ids = generator.sample(sorted(self.samples_by_game), game_count)
        return tuple(
            generator.choice(self.samples_by_game[task_id])
            for task_id in task_ids
        )


def build_grounding_manifest(
    *,
    results: Sequence[tuple[str, ExpertReplayResult]],
    source_checksums: Mapping[str, str],
    code_revision: str,
    max_replay_steps: int,
    persist_horizon: int,
    minimum_success_coverage: float = 0.99,
    maximum_over_horizon_rate: float = 0.01,
) -> GroundingManifest:
    if not results:
        raise ValueError("cannot build a grounding manifest without replay results")
    success = [result for _, result in results if result.succeeded]
    lengths = [result.total_steps for _, result in results]
    over_horizon = sum(result.total_steps > persist_horizon for _, result in results)
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    for task_type, result in results:
        type_counts[task_type]["total"] += 1
        if result.succeeded:
            type_counts[task_type]["successful"] += 1
        else:
            type_counts[task_type]["quarantined"] += 1
            reasons[result.quarantine_reason or "unknown"] += 1

    coverage = len(success) / len(results)
    over_rate = over_horizon / len(results)
    failures: list[str] = []
    if coverage < minimum_success_coverage:
        failures.append("success_coverage_below_threshold")
    if over_rate > maximum_over_horizon_rate:
        failures.append("over_horizon_rate_above_threshold")
    ordered = sorted(lengths)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return GroundingManifest(
        schema_version=1,
        source_split="train",
        total_games=len(results),
        successful_games=len(success),
        quarantined_games=len(results) - len(success),
        success_coverage=coverage,
        over_persist_horizon=over_horizon,
        over_persist_horizon_rate=over_rate,
        task_type_counts={key: dict(value) for key, value in sorted(type_counts.items())},
        quarantine_reasons=dict(sorted(reasons.items())),
        trajectory_lengths={
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": median,
        },
        source_checksums=dict(sorted(source_checksums.items())),
        code_revision=code_revision,
        expert_name="ALFWorld HandCodedTWAgent (direct, strict admissibility)",
        max_replay_steps=max_replay_steps,
        persist_horizon=persist_horizon,
        formal_gate_passed=not failures,
        formal_gate_failures=tuple(failures),
    )


def write_grounding_artifacts(
    *,
    output_directory: str | Path,
    results: Iterable[tuple[str, ExpertReplayResult]],
    manifest: GroundingManifest,
) -> None:
    """Atomically persist successful samples, quarantines, and the audit manifest."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    samples_path = destination / "grounding_samples.jsonl"
    quarantine_path = destination / "quarantine.jsonl"
    manifest_path = destination / "manifest.json"
    sample_lines: list[str] = []
    quarantine_lines: list[str] = []
    for task_type, result in results:
        if result.succeeded:
            for sample in result.samples:
                sample_lines.append(
                    json.dumps(
                        _grounding_sample_payload(task_type, result.task_id, sample),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        else:
            quarantine_payload = {
                "task_id": result.task_id,
                "task_type": task_type,
                "total_steps": result.total_steps,
                "reason": result.quarantine_reason,
            }
            if result.exception_type is not None:
                quarantine_payload["exception_stage"] = result.exception_stage
                quarantine_payload["exception_type"] = result.exception_type
                quarantine_payload["exception_message"] = result.exception_message
            quarantine_lines.append(
                json.dumps(
                    quarantine_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    _atomic_write(samples_path, "\n".join(sample_lines) + ("\n" if sample_lines else ""))
    _atomic_write(quarantine_path, "\n".join(quarantine_lines) + ("\n" if quarantine_lines else ""))
    _atomic_write(
        manifest_path,
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def grounding_result_payload(
    task_type: str,
    result: ExpertReplayResult,
) -> dict[str, object]:
    """Serialize one complete replay result for an internal worker boundary."""

    return {
        "task_type": task_type,
        "task_id": result.task_id,
        "succeeded": result.succeeded,
        "samples": [
            _grounding_sample_payload(task_type, result.task_id, sample)
            for sample in result.samples
        ],
        "total_steps": result.total_steps,
        "quarantine_reason": result.quarantine_reason,
        "exception_stage": result.exception_stage,
        "exception_type": result.exception_type,
        "exception_message": result.exception_message,
    }


def read_grounding_results(
    path: str | Path,
) -> list[tuple[str, ExpertReplayResult]]:
    results: list[tuple[str, ExpertReplayResult]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("result row must be an object")
                task_type = str(payload["task_type"])
                task_id = str(payload["task_id"])
                sample_payloads = payload["samples"]
                if not isinstance(sample_payloads, list):
                    raise TypeError("result samples must be a list")
                samples = tuple(
                    _decode_grounding_sample(sample) for sample in sample_payloads
                )
                if any(sample.state.task_id != task_id for sample in samples):
                    raise ValueError("result sample task_id mismatch")
                result = ExpertReplayResult(
                    task_id=task_id,
                    succeeded=bool(payload["succeeded"]),
                    samples=samples,
                    total_steps=int(payload["total_steps"]),
                    quarantine_reason=payload.get("quarantine_reason"),
                    exception_stage=payload.get("exception_stage"),
                    exception_type=payload.get("exception_type"),
                    exception_message=payload.get("exception_message"),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid grounding result at line {line_number}: {error}"
                ) from error
            results.append((task_type, result))
    return results


def _grounding_sample_payload(
    task_type: str,
    task_id: str,
    sample: GroundingSample,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "state": asdict(sample.state),
        "expert_action": sample.expert_action,
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _decode_grounding_sample(payload: object) -> GroundingSample:
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise TypeError("grounding row requires a state object")
    state_payload = dict(payload["state"])
    history_payload = state_payload.get("history", [])
    if not isinstance(history_payload, list):
        raise TypeError("grounding state history must be a list")
    state_payload["history"] = tuple(
        AgentHistoryEntry(**entry) for entry in history_payload
    )
    for field in ("admissible_commands", "candidate_skill_ids"):
        value = state_payload.get(field, [])
        if not isinstance(value, list):
            raise TypeError(f"grounding state {field} must be a list")
        state_payload[field] = tuple(value)
    state = CanonicalAgentState(**state_payload)
    if payload.get("task_id") != state.task_id:
        raise ValueError("grounding row task_id does not match state.task_id")
    if payload.get("task_type") != state.task_type:
        raise ValueError("grounding row task_type does not match state.task_type")
    action = payload.get("expert_action")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("grounding row requires a non-empty expert_action")
    return GroundingSample(state=state, expert_action=action)
