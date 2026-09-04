from __future__ import annotations

import unittest

from infoskill.integrations.verl.compatibility import (
    require_vllm_084_cachetools_compatibility,
)


class _CompatibleCache:
    def __init__(self, maxsize: int) -> None:
        del maxsize
        self._LRUCache__update = lambda key: key


class _IncompatibleCache:
    def __init__(self, maxsize: int) -> None:
        del maxsize
        self._LRUCache__touch = lambda key: key


class VerlCompatibilityTests(unittest.TestCase):
    def test_accepts_cachetools_private_api_used_by_vllm_084(self) -> None:
        module = type(
            "Cachetools",
            (),
            {"LRUCache": _CompatibleCache, "__version__": "5.5.2"},
        )

        require_vllm_084_cachetools_compatibility(module)

    def test_rejects_new_cachetools_before_loading_models(self) -> None:
        module = type(
            "Cachetools",
            (),
            {"LRUCache": _IncompatibleCache, "__version__": "6.0.0"},
        )

        with self.assertRaisesRegex(RuntimeError, "cachetools==5.5.2"):
            require_vllm_084_cachetools_compatibility(module)


if __name__ == "__main__":
    unittest.main()
