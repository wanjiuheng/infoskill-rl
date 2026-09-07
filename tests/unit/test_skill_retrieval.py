from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.skills import (
    EmbeddingRetriever,
    FixedSkillLibrary,
    PrecomputedEmbeddingRetriever,
    TemplateRetriever,
)


class _Encoder:
    vectors = {
        "General A. alpha. always": (1.0, 0.0),
        "General B. beta. sometimes": (0.0, 1.0),
        "Clean A. wash. dirty": (0.2, 0.8),
        "Heat A. warm. cold": (0.9, 0.1),
        "heat the apple": (1.0, 0.0),
        "clean the apple": (0.0, 1.0),
    }

    def encode(self, texts):
        return [self.vectors[text] for text in texts]


class SkillRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = FixedSkillLibrary.load(Path(__file__).parents[1] / "fixtures" / "skills.json")

    def test_embedding_ranks_task_skills_across_all_categories(self) -> None:
        result = EmbeddingRetriever(
            self.library,
            _Encoder(),
            general_top_k=1,
            task_top_k=1,
            mistake_count=1,
        ).retrieve("heat the apple")

        self.assertEqual(result.skill_ids, ("gen_a", "heat_a", "err_a"))
        self.assertEqual(result.mode, "embedding")

    def test_template_uses_goal_category_without_an_embedding_model(self) -> None:
        result = TemplateRetriever(self.library).retrieve("clean an apple and put it away")

        self.assertIn("clean_a", result.skill_ids)
        self.assertNotIn("heat_a", result.skill_ids)

    def test_precomputed_embedding_retrieval_batches_and_freezes_queries(self) -> None:
        class RecordingEncoder(_Encoder):
            def __init__(self) -> None:
                self.calls = []

            def encode(self, texts):
                self.calls.append(tuple(texts))
                return super().encode(texts)

        encoder = RecordingEncoder()
        retriever = PrecomputedEmbeddingRetriever(
            self.library,
            encoder,
            queries=("heat the apple", "clean the apple", "heat the apple"),
            general_top_k=1,
            task_top_k=1,
            mistake_count=1,
        )

        self.assertEqual(len(encoder.calls), 2)
        self.assertEqual(
            encoder.calls[1],
            ("heat the apple", "clean the apple"),
        )
        self.assertEqual(
            retriever.retrieve("heat the apple").skill_ids,
            ("gen_a", "heat_a", "err_a"),
        )
        with self.assertRaisesRegex(KeyError, "not precomputed"):
            retriever.retrieve("unknown goal")


if __name__ == "__main__":
    unittest.main()
