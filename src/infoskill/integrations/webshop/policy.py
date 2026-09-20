from __future__ import annotations

from collections.abc import Sequence


def render_webshop_policy_message(
    *,
    task_description: str,
    current_observation: str,
    available_actions: Sequence[str],
    history: Sequence[tuple[str, str]] = (),
    history_limit: int = 2,
) -> str:
    """Render the registered InfoSkill WebShop prompt contract.

    Keeping this renderer independent from Gym and Ray lets warm-start and the
    future online adapter share one interface without importing SkillRL internals.
    """

    task = _required_text(task_description, "task_description")
    observation = _required_text(current_observation, "current_observation")
    actions = tuple(_required_text(item, "available_action") for item in available_actions)
    if not actions:
        raise ValueError("WebShop policy prompt requires available actions")
    if history_limit < 0:
        raise ValueError("history_limit must be non-negative")
    rendered_actions = "\n".join(f"'{action}'," for action in actions)
    if not history or history_limit == 0:
        return (
            "You are an expert autonomous agent operating in the WebShop "
            "e‑commerce environment.\n"
            f"Your task is to: {task}.\n"
            f"Your current observation is: {observation}.\n"
            "Your admissible actions of the current situation are: \n[\n"
            f"{rendered_actions}\n].\n\n"
            "Now it's your turn to take one action for the current step.\n"
            "You should first reason step-by-step about the current situation, "
            "then think carefully which admissible action best advances the "
            "shopping goal. This reasoning process MUST be enclosed within "
            "<think> </think> tags. \n"
            "Once you've finished your reasoning, you should choose an "
            "admissible action for current step and present it within "
            "<action> </action> tags."
        )
    recent = tuple(history[-history_limit:])
    start_index = len(history) - len(recent) + 1
    history_lines = []
    for offset, (past_observation, past_action) in enumerate(recent):
        step = start_index + offset
        history_lines.append(
            f"[Observation {step}: '{_required_text(past_observation, 'history observation')}', "
            f"Action {step}: '{_required_text(past_action, 'history action')}']"
        )
    action_history = "\n".join(history_lines)
    return (
        "You are an expert autonomous agent operating in the WebShop "
        "e‑commerce environment.\n"
        f"Your task is to: {task}.\n"
        f"Prior to this step, you have already taken {len(history)} step(s). "
        f"Below are the most recent {len(recent)} observations and the "
        f"corresponding actions you took: {action_history}\n"
        f"You are now at step {len(history) + 1} and your current observation "
        f"is: {observation}.\n"
        "Your admissible actions of the current situation are:\n[\n"
        f"{rendered_actions}\n].\n\n"
        "Now it's your turn to take one action for the current step.\n"
        "You should first reason step-by-step about the current situation, "
        "then think carefully which admissible action best advances the "
        "shopping goal. This reasoning process MUST be enclosed within "
        "<think> </think> tags.\n"
        "Once you've finished your reasoning, you should choose an admissible "
        "action for current step and present it within <action> </action> tags."
    )


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"WebShop {field} must be non-empty text")
    return value.strip()
