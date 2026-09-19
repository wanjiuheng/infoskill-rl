from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from infoskill.imitation.providers import (
    SearchDemonstrationProvider,
    WebShopDemonstrationProvider,
)


class DemonstrationProviderTests(unittest.TestCase):
    def test_webshop_and_search_require_successful_environment_native_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "demos.jsonl"
            path.write_text(
                json.dumps({
                    "trajectory_id": "x",
                    "environment": "webshop",
                    "success": True,
                    "steps": [{"prompt": "state", "action": "click[item]"}],
                }) + "\n",
                encoding="utf-8",
            )
            provider = WebShopDemonstrationProvider(path)
            self.assertEqual(provider.trajectories()[0].environment, "webshop")

            with self.assertRaisesRegex(ValueError, "environment"):
                SearchDemonstrationProvider(path).trajectories()


if __name__ == "__main__":
    unittest.main()
