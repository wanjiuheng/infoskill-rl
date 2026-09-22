from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


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

    def test_existing_parent_supports_an_uncreated_data_root(self) -> None:
        root = Path(__file__).resolve().parents[2]
        missing = root / "not-created" / "webshop" / "raw"
        self.assertEqual(self.doctor._existing_parent(missing), root)

    def test_index_manifest_requires_a_complete_full_corpus(self) -> None:
        path = Path(__file__).resolve().parent / "not-created-index-manifest.json"
        self.assertFalse(self.doctor._index_manifest_status(path)["complete"])
        report = {
            "schema_version": 1,
            "status": "complete",
            "document_count": 100_000,
            "documents_sha256": "a" * 64,
            "probe_hits": 1,
            "index_file_count": 1,
        }
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(
            Path, "read_text", return_value=json.dumps(report)
        ):
            self.assertFalse(self.doctor._index_manifest_status(path)["complete"])
        report["document_count"] = 100_001
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(
            Path, "read_text", return_value=json.dumps(report)
        ):
            self.assertTrue(self.doctor._index_manifest_status(path)["complete"])


if __name__ == "__main__":
    unittest.main()
