from __future__ import annotations

from functools import wraps
from threading import RLock
from typing import Callable


_PARSER_LOCK = RLock()
_GUARD_MARKER = "__infoskill_textworld_parser_guard__"


def install_textworld_parser_guard(parser_module: object | None = None) -> bool:
    """Serialize calls into TextWorld's process-global mutable Tatsu parser.

    TextWorld 1.7 keeps generated parser instances in module globals. Their
    parse stacks are mutated by ``_parse_and_convert`` during both reset and
    step, so concurrent ALFWorld environments must not enter the same parser
    at the same time. Only parsing is guarded; the rest of each independent
    environment step remains concurrent.

    Returns True when this call installs the guard and False when it was
    already installed.
    """

    if parser_module is None:
        from textworld.envs.pddl import textgen as parser_module

    parse = getattr(parser_module, "_parse_and_convert", None)
    if not callable(parse):
        raise RuntimeError("TextWorld parser module has no _parse_and_convert callable")
    if bool(getattr(parse, _GUARD_MARKER, False)):
        return False

    original: Callable[..., object] = parse

    @wraps(original)
    def guarded_parse(*args: object, **kwargs: object) -> object:
        with _PARSER_LOCK:
            return original(*args, **kwargs)

    setattr(guarded_parse, _GUARD_MARKER, True)
    setattr(parser_module, "_parse_and_convert", guarded_parse)
    return True


def install_textworld_parser_guards(
    *,
    textgen: object | None = None,
    logic: object | None = None,
) -> tuple[bool, bool]:
    """Guard both mutable parsers reached by ALFWorld PDDL steps.

    Grammar expansion uses ``textworld.envs.pddl.textgen``, while rule
    conditions reached from that expansion use ``textworld.logic``. Both
    modules keep a process-global Tatsu parser and both wrappers intentionally
    share ``_PARSER_LOCK`` so nested calls are safe.
    """

    if textgen is None:
        from textworld.envs.pddl import textgen
    if logic is None:
        import textworld.logic as logic
    return (
        install_textworld_parser_guard(textgen),
        install_textworld_parser_guard(logic),
    )
