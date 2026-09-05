from __future__ import annotations

from functools import wraps
from threading import RLock
from typing import Callable


_PARSER_LOCK = RLock()
_GUARD_MARKER = "__infoskill_textworld_parser_guard__"


def install_textworld_parser_guard(textgen_module: object | None = None) -> bool:
    """Serialize calls into TextWorld's process-global mutable Tatsu parser.

    TextWorld 1.7 keeps one generated parser instance in
    ``textworld.envs.pddl.textgen._PARSER``. Its parse stacks are mutated by
    ``_parse_and_convert`` during both reset and step, so concurrent ALFWorld
    environments must not enter that parser at the same time. Only parsing is
    guarded; the rest of each independent environment step remains concurrent.

    Returns True when this call installs the guard and False when it was
    already installed.
    """

    if textgen_module is None:
        from textworld.envs.pddl import textgen as textgen_module

    parse = getattr(textgen_module, "_parse_and_convert", None)
    if not callable(parse):
        raise RuntimeError("TextWorld textgen._parse_and_convert is unavailable")
    if bool(getattr(parse, _GUARD_MARKER, False)):
        return False

    original: Callable[..., object] = parse

    @wraps(original)
    def guarded_parse(*args: object, **kwargs: object) -> object:
        with _PARSER_LOCK:
            return original(*args, **kwargs)

    setattr(guarded_parse, _GUARD_MARKER, True)
    setattr(textgen_module, "_parse_and_convert", guarded_parse)
    return True
