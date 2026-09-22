import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.integrations.webshop.paper_protocol import (
    build_paper128_manifest,
    load_paper128_manifest,
    validate_paper128_manifest,
    validation_session_indices,
    write_paper128_manifest,
)


FIXTURE_ROOT = Path("tests/fixtures/webshop-paper128")
SOURCE_FILES = {
    "products": FIXTURE_ROOT / "items_shuffle_1000.json",
    "attributes": FIXTURE_ROOT / "items_ins_v2_1000.json",
    "human_instructions": FIXTURE_ROOT / "items_human_ins.json",
}


def _goals(worker_seed: int):
    return [
        {
            "asin": f"asin-{worker_seed}-{index}",
            "instruction_text": f"goal {worker_seed} {index}",
            "attributes": ["durable"],
            "price_upper": 40.0,
            "goal_options": {"color": "blue"},
        }
        for index in range(500)
    ]


class WebShopPaperProtocolTests(unittest.TestCase):
    def test_first_public_validation_draw_is_stable(self) -> None:
        indices = validation_session_indices()
        self.assertEqual(len(indices), 128)
        self.assertEqual(
            indices[:10],
            (319, 207, 22, 420, 352, 491, 60, 264, 454, 21),
        )
        self.assertEqual(indices[-5:], (122, 265, 226, 64, 117))
        self.assertEqual(len(set(indices)), 128)

    def test_later_validation_call_advances_same_rng(self) -> None:
        first = validation_session_indices(validation_call_index=0)
        second = validation_session_indices(validation_call_index=1)
        self.assertNotEqual(first, second)
        self.assertEqual(
            second,
            validation_session_indices(validation_call_index=1),
        )

    def test_manifest_freezes_resolved_per_worker_goals(self) -> None:
        manifest = build_paper128_manifest(
            goal_factory=_goals,
            source_files=SOURCE_FILES,
            environment_commit="a" * 40,
        )

        self.assertEqual(len(manifest["tasks"]), 128)
        first = manifest["tasks"][0]
        self.assertEqual(first["worker_seed"], 1000)
        self.assertEqual(first["session_index"], 319)
        self.assertEqual(first["goal"]["asin"], "asin-1000-319")
        validate_paper128_manifest(manifest)

    def test_manifest_rejects_tampering_and_overwrite(self) -> None:
        manifest = build_paper128_manifest(
            goal_factory=_goals,
            source_files=SOURCE_FILES,
            environment_commit="b" * 40,
        )

        output = Path("paper128-do-not-create.json")
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(manifest)),
        ):
            write_paper128_manifest(output, manifest)

        with (
            patch.object(Path, "resolve", return_value=output),
            patch.object(Path, "read_text", return_value=json.dumps(manifest)),
        ):
            loaded = load_paper128_manifest(output, environment_commit="b" * 40)
        self.assertEqual(loaded["manifest_sha256"], manifest["manifest_sha256"])

        with (
            patch.object(Path, "resolve", return_value=output),
            patch.object(Path, "read_text", return_value=json.dumps(manifest)),
        ):
            with self.assertRaisesRegex(ValueError, "environment commit"):
                load_paper128_manifest(output, environment_commit="different")

        tampered = copy.deepcopy(manifest)
        tampered["tasks"][0]["goal"]["instruction_text"] = "changed"
        with self.assertRaisesRegex(ValueError, "goal checksum"):
            validate_paper128_manifest(tampered)

        different = build_paper128_manifest(
            goal_factory=_goals,
            source_files=SOURCE_FILES,
            environment_commit="c" * 40,
        )
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(manifest)),
        ):
            with self.assertRaises(FileExistsError):
                write_paper128_manifest(output, different)


if __name__ == "__main__":
    unittest.main()
