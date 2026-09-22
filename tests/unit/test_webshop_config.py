from __future__ import annotations

import unittest
from pathlib import Path

from infoskill.app_config import AppConfig
from infoskill.builders import _load_skill_library_provenance
from infoskill.skills.library import FixedSkillLibrary


class WebShopConfigTests(unittest.TestCase):
    def test_webshop_config_and_skill_provenance_are_registered(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = AppConfig.load(root / "configs/webshop_qwen25_7b.yaml")
        self.assertEqual(config.environment, "webshop")
        self.assertIsNone(config.paths.alfworld_data)
        self.assertEqual(Path(config.paths.webshop_data).parts[-2:], ("data", "webshop"))
        self.assertTrue(
            Path(config.paths.webshop_human_demonstrations).name
            == "il_trajs_finalized_images.jsonl"
        )
        self.assertTrue(
            Path(config.paths.webshop_task_manifest).name
            == "webshop-paper128-manifest.json"
        )
        self.assertEqual(config.max_steps, 15)
        self.assertEqual(config.eval_batch_size, 64)

        skill_bank = root.parent / "SkillRL/memory_data/webshop/claude_style_skills.json"
        library = FixedSkillLibrary.load(skill_bank)
        provenance = _load_skill_library_provenance(
            library,
            manifest_path=str(root / "configs/webshop_skill_bank_manifest.json"),
            environment="webshop",
        )
        self.assertEqual(provenance["environment"], "webshop")
        self.assertEqual(provenance["trajectory_count"], 200)


if __name__ == "__main__":
    unittest.main()
