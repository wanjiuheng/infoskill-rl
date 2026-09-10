"""Skill-conditioning interface and registered control modes."""

from .contracts import (
    ConditionedPolicyInput,
    ConditioningContext,
    ConditioningRequest,
    InfoSkillConditioningResult,
    InfoSkillConditioningRuntime,
    InfoSkillConditioningWorkItem,
    InfoSkillReplayTrace,
    SkillConditioner,
)
from .no_skill import NoSkillConditioner
from .runtime_info_skill import RuntimeInfoSkillConditioner
from .raw_skill import (
    EpisodeRetriever,
    RawSkillPromptConditioner,
    SkillRlGrpoPromptConditioner,
    SkillRlSftPromptConditioner,
    SkillRlSftNoSkillsPromptConditioner,
    format_raw_skill_block,
)

try:
    from .info_skill import InfoSkillConditioner
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    InfoSkillConditioner = None  # type: ignore[assignment,misc]

__all__ = [
    "ConditionedPolicyInput",
    "ConditioningContext",
    "ConditioningRequest",
    "EpisodeRetriever",
    "NoSkillConditioner",
    "InfoSkillConditioner",
    "InfoSkillConditioningResult",
    "InfoSkillConditioningRuntime",
    "InfoSkillConditioningWorkItem",
    "InfoSkillReplayTrace",
    "RawSkillPromptConditioner",
    "RuntimeInfoSkillConditioner",
    "SkillRlGrpoPromptConditioner",
    "SkillRlSftPromptConditioner",
    "SkillRlSftNoSkillsPromptConditioner",
    "SkillConditioner",
    "format_raw_skill_block",
]
