from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_doctor_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/webshop_asset_doctor.py"
    spec = importlib.util.spec_from_file_location("webshop_asset_doctor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebShopAssetDoctorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doctor = _load_doctor_module()

    def test_java_11_is_the_only_accepted_major_version(self) -> None:
        self.assertTrue(self.doctor._is_java_11('openjdk version "11.0.25"'))
        self.assertTrue(self.doctor._is_java_11("openjdk 11.0.25 2024-10-15"))
        self.assertFalse(self.doctor._is_java_11('openjdk version "1.8.0_432"'))
        self.assertFalse(self.doctor._is_java_11('openjdk version "17.0.13"'))

    def test_missing_webshop_source_fails_closed(self) -> None:
        missing = Path(__file__).resolve().parent / "does-not-exist"
        status = self.doctor._web_agent_site_status(missing)
        self.assertFalse(status["source_present"])
        self.assertFalse(status["importable"])


if __name__ == "__main__":
    unittest.main()
