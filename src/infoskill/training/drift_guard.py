from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol


class MetricUpdate(Protocol):
    global_update: int
    values: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DriftGuardObservation:
    global_update: int
    ppo_kl: float
    invalid_action_rate: float
    breached: bool
    consecutive_breaches: int
    triggered: bool
    trigger_update: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class TrainingDriftGuard:
    """Latch a safe pause after sustained simultaneous policy drift."""

    def __init__(
        self,
        *,
        ppo_kl_threshold: float,
        invalid_action_rate_threshold: float,
        consecutive_updates: int,
    ) -> None:
        for name, value in (
            ("ppo_kl_threshold", ppo_kl_threshold),
            ("invalid_action_rate_threshold", invalid_action_rate_threshold),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if consecutive_updates <= 0:
            raise ValueError("consecutive_updates must be positive")
        self.ppo_kl_threshold = ppo_kl_threshold
        self.invalid_action_rate_threshold = invalid_action_rate_threshold
        self.consecutive_updates = consecutive_updates
        self._consecutive_breaches = 0
        self._trigger_update: int | None = None
        self._last_observation: DriftGuardObservation | None = None

    @property
    def triggered(self) -> bool:
        return self._trigger_update is not None

    @property
    def trigger_update(self) -> int | None:
        return self._trigger_update

    def observe(self, update: MetricUpdate) -> DriftGuardObservation:
        ppo_kl = self._metric(update.values, "actor/ppo_kl")
        invalid_action_rate = self._metric(
            update.values,
            "rollout/invalid_action_rate",
        )
        breached = (
            ppo_kl > self.ppo_kl_threshold
            and invalid_action_rate > self.invalid_action_rate_threshold
        )
        if not self.triggered:
            self._consecutive_breaches = (
                self._consecutive_breaches + 1 if breached else 0
            )
            if self._consecutive_breaches >= self.consecutive_updates:
                self._trigger_update = update.global_update
        observation = DriftGuardObservation(
            global_update=update.global_update,
            ppo_kl=ppo_kl,
            invalid_action_rate=invalid_action_rate,
            breached=breached,
            consecutive_breaches=self._consecutive_breaches,
            triggered=self.triggered,
            trigger_update=self.trigger_update,
        )
        self._last_observation = observation
        return observation

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "enabled": True,
            "ppo_kl_threshold": self.ppo_kl_threshold,
            "invalid_action_rate_threshold": (
                self.invalid_action_rate_threshold
            ),
            "consecutive_updates": self.consecutive_updates,
            "triggered": self.triggered,
            "trigger_update": self.trigger_update,
            "consecutive_breaches": self._consecutive_breaches,
            "last_observation": (
                self._last_observation.as_dict()
                if self._last_observation is not None
                else None
            ),
        }

    @staticmethod
    def _metric(values: Mapping[str, float], name: str) -> float:
        if name not in values:
            raise RuntimeError(
                f"training drift guard requires metric {name!r}"
            )
        value = float(values[name])
        if not math.isfinite(value):
            raise RuntimeError(
                f"training drift guard metric {name!r} must be finite"
            )
        return value
