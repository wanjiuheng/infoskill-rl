from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from infoskill.integrations.webshop import audit_official_human_archive


class WebShopHumanArchiveTests(unittest.TestCase):
    def test_audit_reports_structure_without_emitting_goal_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "all_trajs.zip"
            goal = {
                "instruction_text": "private red shoes",
                "attributes": ["red"],
            }
            rows = [
                {"page": "index", "url": "http://example/", "goal": goal},
                {
                    "page": "search_results",
                    "url": "http://example/search",
                    "goal": goal,
                    "content": {"keywords": ["red", "shoes"], "page": 1},
                },
                {
                    "page": "done",
                    "url": "http://example/done",
                    "goal": goal,
                    "content": {"asin": "A1", "options": {}, "price": 20.0},
                    "reward": 1.0,
                },
            ]
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("all_trajs/", "")
                bundle.writestr("__MACOSX/._all_trajs", "metadata")
                bundle.writestr(
                    "all_trajs/20220514_0_fixed_100_1906.jsonl",
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                )

            report = audit_official_human_archive(archive)

            self.assertTrue(report["archive_readable"])
            self.assertEqual(report["zip_entry_count"], 3)
            self.assertEqual(report["trajectory_file_count"], 1)
            self.assertEqual(report["ignored_entry_count"], 2)
            self.assertEqual(report["files_with_done"], 1)
            self.assertEqual(report["terminal_rewards"]["perfect_count"], 1)
            self.assertEqual(report["fixed_goal_indices"]["min"], 1906)
            self.assertNotIn("private red shoes", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
