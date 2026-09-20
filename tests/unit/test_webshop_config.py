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
        self.assertEqual(
            config.paths.webshop_data,
            "/root/autodl-tmp/wjh/data/webshop/data",
        )
        self.assertTrue(
            config.paths.webshop_human_demonstrations.startswith(
                "/root/autodl-tmp/wjh/data/webshop/"
            )
        )

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
