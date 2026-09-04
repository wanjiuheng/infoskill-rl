from __future__ import annotations

from typing import Any


def require_vllm_084_cachetools_compatibility(
    cachetools_module: Any | None = None,
) -> None:
    """Fail before GPU initialization when cachetools removed vLLM's private API."""

    if cachetools_module is None:
        import cachetools as cachetools_module

    cache = cachetools_module.LRUCache(maxsize=1)
    if not hasattr(cache, "_LRUCache__update"):
        version = getattr(cachetools_module, "__version__", "unknown")
        raise RuntimeError(
            "vLLM 0.8.4 LoRA requires cachetools==5.5.2; "
            f"installed cachetools={version} removed _LRUCache__update"
        )
