from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.skills import EmbeddingRetriever, FixedSkillLibrary, TemplateRetriever


class _Encoder:
    vectors = {
        "General A. alpha. always": (1.0, 0.0),
        "General B. beta. sometimes": (0.0, 1.0),
        "Clean A. wash. dirty": (0.2, 0.8),
        "Heat A. warm. cold": (0.9, 0.1),
        "heat the apple": (1.0, 0.0),
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


if __name__ == "__main__":
    unittest.main()
