from __future__ import annotations

import sys
from pathlib import Path


def load_handcoded_expert(
    *,
    alfworld_source: str | Path,
    max_steps: int = 200,
) -> object:
    """Load ALFWorld's expert directly, bypassing AlfredExpert's fallback plan."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    source = str(Path(alfworld_source).expanduser().resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from alfworld.agents.expert.handcoded_expert_tw import HandCodedTWAgent
    except ImportError as error:
        raise RuntimeError(
            "ALFWorld handcoded expert dependencies are missing; install the locked server environment"
        ) from error
    return HandCodedTWAgent(max_steps=max_steps)
