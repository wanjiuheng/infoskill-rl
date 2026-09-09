from __future__ import annotations

import unittest

from infoskill.config import ExperimentConfig, SkillMode


class ExperimentConfigTests(unittest.TestCase):
    def test_formal_infoskill_defaults_match_registered_protocol(self) -> None:
        config = ExperimentConfig.formal(mode=SkillMode.INFO_SKILL)

        self.assertEqual(config.episode.max_steps, 30)
        self.assertEqual(config.episode.history_length, 2)
        self.assertEqual(config.batch.trajectories_per_update, 64)
        self.assertEqual(config.batch.max_tokens_per_gpu, 12_288)
        self.assertEqual(config.generation.max_prompt_tokens, 4096)
        self.assertEqual(config.generation.max_response_tokens, 256)
        self.assertFalse(config.generation.eval_do_sample)
        self.assertEqual(config.evaluation.total_tasks, 140)
        self.assertEqual(
            config.evaluation.manifest_sha256,
            "935b09a7e79f0fe37010081e4cf60ff33d29e301bf2e49822a492c0c02eeacd3",
        )
        config.validate()


if __name__ == "__main__":
    unittest.main()
