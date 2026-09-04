"""Skill-conditioning interface and registered control modes."""

from .contracts import ConditionedPolicyInput, ConditioningContext, SkillConditioner
from .no_skill import NoSkillConditioner
from .raw_skill import EpisodeRetriever, RawSkillPromptConditioner, format_raw_skill_block

try:
    from .info_skill import InfoSkillConditioner, InfoSkillTrace
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    InfoSkillConditioner = None  # type: ignore[assignment,misc]
    InfoSkillTrace = None  # type: ignore[assignment,misc]

__all__ = [
    "ConditionedPolicyInput",
    "ConditioningContext",
    "EpisodeRetriever",
    "NoSkillConditioner",
    "InfoSkillConditioner",
    "InfoSkillTrace",
    "RawSkillPromptConditioner",
    "SkillConditioner",
    "format_raw_skill_block",
]
