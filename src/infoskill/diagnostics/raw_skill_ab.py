from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence

from infoskill.episode import TaskSpec, TrajectoryGroup
from infoskill.integrations.alfworld import ALFWORLD_TASK_TYPES


@dataclass(frozen=True, slots=True)
class RawSkillAbVariant:
    name: str
    retrieval_mode: Literal["embedding", "template"] | None
    prompt_format: Literal[
        "full",
        "skillrl",
        "skillrl_rl_exact",
        "skillrl_sft_exact",
        "skillrl_sft_no_skills",
        "unified_no_skill",
    ]
    policy_mode: Literal["no_skill", "raw_skill_prompt"] = "raw_skill_prompt"
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.do_sample and self.temperature <= 0:
            raise ValueError("sampled diagnostic variants require temperature > 0")
        if not self.do_sample and self.temperature != 0:
            raise ValueError("deterministic diagnostic variants require temperature=0")
        if not 0 < self.top_p <= 1:
            raise ValueError("diagnostic variant top_p must be in (0, 1]")

    @property
    def trace_slug(self) -> str:
        """Return a filesystem-safe label without changing the public name."""

        slug = re.sub(r"[^A-Za-z0-9-]+", "-", self.name).strip("-")
        if not slug:
            raise ValueError("diagnostic variant name has no trace-safe characters")
        return slug


RAW_SKILL_AB_VARIANTS = (
    RawSkillAbVariant("embedding-skillrl", "embedding", "skillrl"),
    RawSkillAbVariant("template-full", "template", "full"),
    RawSkillAbVariant("template-skillrl", "template", "skillrl"),
)

# Kept out of RAW_SKILL_AB_VARIANTS so the already-completed three-cell matrix
# remains the default. This reference-prompt probe is selected explicitly and
# does not redefine the registered raw_skill_prompt control.
SKILLRL_RL_EXACT_VARIANT = RawSkillAbVariant(
    "skillrl-rl-exact",
    "template",
    "skillrl_rl_exact",
)

SKILLRL_SFT_EXACT_VARIANT = RawSkillAbVariant(
    "skillrl-sft-exact",
    "template",
    "skillrl_sft_exact",
)

# A three-cell causal diagnostic for the released SFT policy.  These variants
# intentionally remain outside RAW_SKILL_AB_VARIANTS so the registered default
# matrix and formal deterministic evaluation are unchanged.
SKILLRL_SFT_CAUSAL_VARIANTS = (
    RawSkillAbVariant(
        "skillrl-sft-shell-no-skills-deterministic",
        None,
        "skillrl_sft_no_skills",
    ),
    RawSkillAbVariant(
        "no-skill-sampled-t0.4",
        None,
        "unified_no_skill",
        policy_mode="no_skill",
        do_sample=True,
        temperature=0.4,
    ),
    RawSkillAbVariant(
        "skillrl-sft-exact-sampled-t0.4",
        "template",
        "skillrl_sft_exact",
        do_sample=True,
        temperature=0.4,
    ),
)


def resolve_raw_skill_diagnostic_variants(
    names: Sequence[str] | None,
) -> tuple[RawSkillAbVariant, ...]:
    if names is None:
        return RAW_SKILL_AB_VARIANTS
    available = {
        item.name: item
        for item in (
            *RAW_SKILL_AB_VARIANTS,
            SKILLRL_RL_EXACT_VARIANT,
            SKILLRL_SFT_EXACT_VARIANT,
            *SKILLRL_SFT_CAUSAL_VARIANTS,
        )
    }
    unknown = tuple(name for name in names if name not in available)
    if unknown:
        raise ValueError(
            "unknown raw-skill diagnostic variant(s): " + ", ".join(unknown)
        )
    if len(set(names)) != len(names):
        raise ValueError("raw-skill diagnostic variants must be unique")
    selected = tuple(available[name] for name in names)
    if len({item.trace_slug for item in selected}) != len(selected):
        raise ValueError("raw-skill diagnostic trace labels must be unique")
    return selected


def select_stratified_tasks(
    tasks: Sequence[TaskSpec],
    *,
    tasks_per_type: int = 2,
) -> tuple[TaskSpec, ...]:
    """Choose a stable, balanced diagnostic subset from registered valid_seen."""

    if tasks_per_type <= 0:
        raise ValueError("tasks_per_type must be positive")
    selected: list[TaskSpec] = []
    for task_type in ALFWORLD_TASK_TYPES:
        candidates = sorted(
            (item for item in tasks if item.task_type == task_type),
            key=lambda item: item.task_id,
        )
        if len(candidates) < tasks_per_type:
            raise ValueError(
                f"raw skill diagnostic requires {tasks_per_type} tasks for "
                f"{task_type}; found {len(candidates)}"
            )
        selected.extend(candidates[:tasks_per_type])
    return tuple(selected)


def summarize_probe_groups(groups: Sequence[TrajectoryGroup]) -> dict[str, object]:
    """Summarize diagnostic trajectories without claiming formal evaluation."""

    trajectories = tuple(group.trajectories[0] for group in groups)
    total_steps = sum(len(item.steps) for item in trajectories)
    successes = Counter(
        item.task.task_type for item in trajectories if item.won
    )
    denominators = Counter(item.task.task_type for item in trajectories)
    per_type = {
        task_type: successes[task_type] / denominators[task_type]
        for task_type in ALFWORLD_TASK_TYPES
        if denominators[task_type]
    }
    repeats = 0
    transitions = 0
    all_look = 0
    no_object_interaction = 0
    interaction_verbs = {
        "take",
        "put",
        "open",
        "close",
        "use",
        "heat",
        "cool",
        "clean",
        "move",
        "examine",
    }
    for trajectory in trajectories:
        actions = tuple(step.action.candidate or "" for step in trajectory.steps)
        repeats += sum(left == right for left, right in zip(actions, actions[1:]))
        transitions += max(0, len(actions) - 1)
        verbs = tuple(action.split(" ", 1)[0] if action else "" for action in actions)
        all_look += int(bool(verbs) and all(verb == "look" for verb in verbs))
        no_object_interaction += int(
            not any(verb in interaction_verbs for verb in verbs)
        )
    task_count = len(trajectories)
    return {
        "diagnostic_only": True,
        "reportable_as_valid_seen": False,
        "task_count": task_count,
        "success_count": sum(successes.values()),
        "overall_success": (
            sum(successes.values()) / task_count if task_count else None
        ),
        "macro_success": (
            sum(per_type.values()) / len(per_type) if per_type else None
        ),
        "per_task_type_success": per_type,
        "total_steps": total_steps,
        "mean_steps": total_steps / task_count if task_count else None,
        "invalid_action_rate": (
            sum(item.invalid_action_count for item in trajectories) / total_steps
            if total_steps
            else None
        ),
        "consecutive_repeat_rate": repeats / transitions if transitions else None,
        "all_look_trajectory_count": all_look,
        "no_object_interaction_trajectory_count": no_object_interaction,
    }
