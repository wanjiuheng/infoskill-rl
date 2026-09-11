from __future__ import annotations

from typing import Mapping


class PlannerPayloadExpert:
    """Expose ALFWorld's current planner command through the expert protocol."""

    def reset(self, gamefile: str) -> None:
        del gamefile

    def observe(self, feedback: str) -> None:
        del feedback

    def act(
        self,
        game_state: Mapping[str, object],
        reward: float,
        done: bool,
        last_action: str,
    ) -> str:
        del reward, done, last_action
        plan = game_state.get("extra.expert_plan")
        if isinstance(plan, str):
            commands = (plan,)
        else:
            try:
                commands = tuple(plan)  # type: ignore[arg-type]
            except TypeError as error:
                raise RuntimeError(
                    "ALFWorld planner did not provide a command list"
                ) from error
        if not commands or not isinstance(commands[0], str) or not commands[0].strip():
            raise RuntimeError("ALFWorld planner returned an empty command list")
        return commands[0]
