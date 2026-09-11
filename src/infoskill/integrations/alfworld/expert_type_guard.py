from __future__ import annotations

import sys
from functools import wraps
from pathlib import Path


_SUPPORTED_EXPERT_TYPES = frozenset({"handcoded", "planner"})
_GUARD_MARKER = "__infoskill_alfworld_expert_type_guard__"
_REQUESTED_MARKER = "__infoskill_requested_expert_type__"
_CORRECTED_MARKER = "__infoskill_positional_binding_corrected__"


def install_alfworld_expert_type_guard(expert_class: type) -> bool:
    """Correct ALFWorld's legacy positional wrapper construction in-process.

    Pinned ALFWorld calls ``AlfredExpert(expert_type)`` even though the first
    constructor parameter is ``env``.  The guard is deliberately narrow: it
    only rebinds one of ALFWorld's two known expert-type strings when the
    explicit ``expert_type`` parameter still has its legacy default.
    """

    original = getattr(expert_class, "__init__", None)
    if not callable(original):
        raise RuntimeError("ALFWorld AlfredExpert has no callable constructor")
    if bool(getattr(original, _GUARD_MARKER, False)):
        return False

    @wraps(original)
    def guarded(
        instance: object,
        env: object = None,
        expert_type: str = "handcoded",
    ) -> None:
        corrected = (
            isinstance(env, str)
            and env in _SUPPORTED_EXPERT_TYPES
            and expert_type == "handcoded"
        )
        requested = env if corrected else expert_type
        if corrected:
            env = None
            expert_type = str(requested)
        original(instance, env=env, expert_type=expert_type)
        setattr(instance, _REQUESTED_MARKER, requested)
        setattr(instance, _CORRECTED_MARKER, corrected)

    setattr(guarded, _GUARD_MARKER, True)
    setattr(expert_class, "__init__", guarded)
    return True


def verify_alfworld_expert_type_binding(
    expert_class: type,
    *,
    requested_expert_type: str,
    module_path: str,
) -> dict[str, object]:
    """Exercise the exact positional call used by pinned ALFWorld and fail closed."""

    if requested_expert_type not in _SUPPORTED_EXPERT_TYPES:
        raise ValueError("requested expert type must be handcoded or planner")
    constructor = getattr(expert_class, "__init__", None)
    guard_active = bool(getattr(constructor, _GUARD_MARKER, False))
    probe = expert_class(requested_expert_type)
    effective = getattr(probe, "expert_type", None)
    corrected = bool(getattr(probe, _CORRECTED_MARKER, False))
    if effective != requested_expert_type:
        raise RuntimeError(
            "ALFWorld effective expert type does not match the requested type: "
            f"requested={requested_expert_type!r}, effective={effective!r}"
        )
    if not guard_active or not corrected:
        raise RuntimeError(
            "ALFWorld expert-type compatibility guard did not intercept the "
            "legacy positional constructor call"
        )
    return {
        "requested_expert_type": requested_expert_type,
        "effective_expert_type": effective,
        "compatibility_guard_active": guard_active,
        "positional_binding_corrected": corrected,
        "alfworld_module_path": module_path,
    }


def prepare_alfworld_expert_type_binding(
    alfworld_source: str | Path,
    *,
    requested_expert_type: str,
) -> dict[str, object]:
    """Load the pinned module, install the guard, and verify actual binding."""

    source = Path(alfworld_source).expanduser().resolve()
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from alfworld.agents.environment import alfred_tw_env
    except ImportError as error:
        raise RuntimeError(
            "ALFWorld runtime dependencies are missing; install the locked "
            "server environment"
        ) from error

    module_path = Path(alfred_tw_env.__file__).resolve()
    try:
        module_path.relative_to(source)
    except ValueError as error:
        raise RuntimeError(
            "loaded ALFWorld module is outside the configured source tree: "
            f"configured={source}, loaded={module_path}"
        ) from error

    expert_class = alfred_tw_env.AlfredExpert
    install_alfworld_expert_type_guard(expert_class)
    report = verify_alfworld_expert_type_binding(
        expert_class,
        requested_expert_type=requested_expert_type,
        module_path=str(module_path),
    )
    report["module_within_configured_source"] = True
    return report
