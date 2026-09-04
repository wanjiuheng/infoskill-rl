from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", line)
        if match:
            names.add(match.group(0).lower().replace("_", "-"))
    return names


class ServerRequirementsTests(unittest.TestCase):
    def test_covers_dependencies_skipped_by_editable_verl_install(self) -> None:
        actual = _requirement_names(PROJECT_ROOT / "requirements-server.txt")
        required = {
            "datasets",
            "flash-attn",
            "pybind11",
            "pylatexenc",
            "qwen-vl-utils",
            "torchdata",
            "wandb",
        }

        self.assertEqual(required - actual, set())

    def test_pins_cachetools_before_vllm_084_private_lru_api_changed(self) -> None:
        requirements = (PROJECT_ROOT / "requirements-server.txt").read_text(
            encoding="utf-8"
        )

        self.assertRegex(requirements, r"(?m)^cachetools==5\.5\.2$")


if __name__ == "__main__":
    unittest.main()
