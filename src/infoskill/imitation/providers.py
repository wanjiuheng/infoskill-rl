from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DemonstrationStep:
    prompt: str
    action: str


@dataclass(frozen=True, slots=True)
class DemonstrationTrajectory:
    trajectory_id: str
    environment: str
    steps: tuple[DemonstrationStep, ...]


class DemonstrationProvider(Protocol):
    def trajectories(self) -> tuple[DemonstrationTrajectory, ...]: ...


class JsonlDemonstrationProvider:
    """Strict adapter for environment-native WebShop/Search demonstrations.

    This deliberately does not pretend that those environments expose the
    ALFWorld planner.  Each environment owns generation of its successful
    trajectories; this class only validates and normalizes their handoff.
    """

    def __init__(self, path: str | Path, *, environment: str) -> None:
        if environment not in {"webshop", "search"}:
            raise ValueError("environment must be webshop or search")
        self.path = Path(path)
        self.environment = environment

    def trajectories(self) -> tuple[DemonstrationTrajectory, ...]:
        result: list[DemonstrationTrajectory] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("environment") != self.environment:
                    raise ValueError(
                        f"demonstration environment mismatch at line {line_number}"
                    )
                if payload.get("success") is not True:
                    raise ValueError(
                        f"demonstration must be successful at line {line_number}"
                    )
                steps_payload = payload.get("steps")
                if not isinstance(steps_payload, list) or not steps_payload:
                    raise ValueError(
                        f"demonstration requires non-empty steps at line {line_number}"
                    )
                steps = tuple(
                    DemonstrationStep(
                        prompt=_nonempty(step, "prompt", line_number),
                        action=_nonempty(step, "action", line_number),
                    )
                    for step in steps_payload
                )
                trajectory_id = str(payload.get("trajectory_id", "")).strip()
                if not trajectory_id:
                    raise ValueError(
                        f"demonstration requires trajectory_id at line {line_number}"
                    )
                result.append(
                    DemonstrationTrajectory(
                        trajectory_id=trajectory_id,
                        environment=self.environment,
                        steps=steps,
                    )
                )
        if not result:
            raise ValueError("demonstration provider returned no trajectories")
        ids = [item.trajectory_id for item in result]
        if len(ids) != len(set(ids)):
            raise ValueError("demonstration trajectory IDs must be unique")
        return tuple(result)


class WebShopDemonstrationProvider(JsonlDemonstrationProvider):
    """Consume successful demonstrations emitted by the WebShop adapter."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, environment="webshop")


class SearchDemonstrationProvider(JsonlDemonstrationProvider):
    """Consume successful demonstrations emitted by the Search adapter."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, environment="search")


def _nonempty(payload: object, field: str, line_number: int) -> str:
    if not isinstance(payload, dict):
        raise ValueError(f"demonstration step must be an object at line {line_number}")
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"demonstration step requires {field} at line {line_number}"
        )
    return value.strip()
