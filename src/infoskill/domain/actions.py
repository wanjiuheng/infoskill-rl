from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence


INVALID_ACTION_SENTINEL = "__invalid_action__"

_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_THINK_TAG = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_LIST_PREFIX = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_ROLE_PREFIX = re.compile(r"^(?:action|assistant)\s*:\s*", re.IGNORECASE)
_UNCLOSED_ACTION_PREFIX = re.compile(r"^<action>\s*", re.IGNORECASE)

ExtractionMethod = Literal["action_tag", "last_line", "none"]


@dataclass(frozen=True, slots=True)
class ActionResolution:
    candidate: str | None
    resolved_action: str | None
    executed_action: str
    extraction_method: ExtractionMethod
    is_executable: bool
    had_action_tag: bool
    had_think_tag: bool
    format_compliant: bool
    failure_reason: str | None


def _normalize(command: str) -> str:
    return " ".join(command.strip().lower().split())


def _match_environment_command(candidate: str, commands: Sequence[str]) -> str | None:
    normalized = _normalize(candidate)
    matches = [command for command in commands if _normalize(command) == normalized]
    return matches[0] if len(matches) == 1 else None


def _clean_last_line(line: str) -> str:
    candidate = _LIST_PREFIX.sub("", line.strip(), count=1).strip()
    if len(candidate) >= 6 and candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:-3].strip()
    candidate = _ROLE_PREFIX.sub("", candidate, count=1).strip()
    candidate = _UNCLOSED_ACTION_PREFIX.sub("", candidate, count=1).strip()
    if len(candidate) >= 2 and candidate.startswith("`") and candidate.endswith("`"):
        candidate = candidate[1:-1].strip()
    return candidate


def resolve_action(model_output: str, admissible_commands: Sequence[str]) -> ActionResolution:
    """Resolve one model response without correcting its semantic action."""

    tagged = _ACTION_TAG.findall(model_output)
    had_think_tag = _THINK_TAG.search(model_output) is not None
    if tagged:
        normalized_tags = {_normalize(candidate) for candidate in tagged}
        if len(normalized_tags) != 1:
            return ActionResolution(
                candidate=None,
                resolved_action=None,
                executed_action=INVALID_ACTION_SENTINEL,
                extraction_method="action_tag",
                is_executable=False,
                had_action_tag=True,
                had_think_tag=had_think_tag,
                format_compliant=False,
                failure_reason="conflicting_action_tags",
            )
        candidate = tagged[0].strip()
        resolved = _match_environment_command(candidate, admissible_commands)
        return ActionResolution(
            candidate=candidate,
            resolved_action=resolved,
            executed_action=resolved or INVALID_ACTION_SENTINEL,
            extraction_method="action_tag",
            is_executable=resolved is not None,
            had_action_tag=True,
            had_think_tag=had_think_tag,
            format_compliant=had_think_tag,
            failure_reason=None if resolved is not None else "not_admissible",
        )

    nonempty_lines = [line for line in model_output.splitlines() if line.strip()]
    candidate = _clean_last_line(nonempty_lines[-1]) if nonempty_lines else ""
    resolved = _match_environment_command(candidate, admissible_commands) if candidate else None
    return ActionResolution(
        candidate=candidate or None,
        resolved_action=resolved,
        executed_action=resolved or INVALID_ACTION_SENTINEL,
        extraction_method="last_line" if candidate else "none",
        is_executable=resolved is not None,
        had_action_tag=False,
        had_think_tag=had_think_tag,
        format_compliant=False,
        failure_reason=None if resolved is not None else ("not_admissible" if candidate else "unresolved"),
    )
