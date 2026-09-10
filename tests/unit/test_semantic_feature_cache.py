from __future__ import annotations

import unittest
from types import MappingProxyType

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environment
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class SemanticFeatureCacheTests(unittest.TestCase):
    def test_fixed_library_warmup_encodes_each_skill_only_once(self) -> None:
        from infoskill.semantic import FeatureBatch, SemanticFeatureCache
        from infoskill.skills import SkillRecord

        encoder = _Encoder(FeatureBatch)
        records = (
            SkillRecord(
                "skill-1",
                "general",
                None,
                "Inspect",
                "inspect first",
                MappingProxyType({}),
            ),
            SkillRecord(
                "skill-2",
                "task_specific",
                "clean",
                "Clean",
                "clean before placing",
                MappingProxyType({}),
            ),
        )
        cache = SemanticFeatureCache(encoder)  # type: ignore[arg-type]

        cache.warm_skills(records)
        cache.warm_skills(records)
        cache.skill_batch(records[:1], batch_size=2)

        self.assertEqual(
            encoder.calls,
            [("inspect first", "clean before placing")],
        )


if torch is not None:

    class _Encoder:
        device = torch.device("cpu")

        def __init__(self, feature_batch) -> None:
            self._feature_batch = feature_batch
            self.calls = []

        def encode_tokens(self, texts, *, max_length=512):
            del max_length
            self.calls.append(tuple(texts))
            rows = len(texts)
            return self._feature_batch(
                torch.ones((rows, 2, 3)),
                torch.ones((rows, 2), dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
