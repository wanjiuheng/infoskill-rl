from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_vllm_patch import _sha256_git_blob


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_ROOT = PROJECT_ROOT / "third_party" / "patches" / "vllm-0.8.4"


class VllmPatchBundleTests(unittest.TestCase):
    def test_manifest_pins_exact_upstream_and_patch(self) -> None:
        manifest = json.loads((PATCH_ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["upstream_version"], "0.8.4")
        self.assertEqual(
            manifest["upstream_commit"],
            "dc1b4a6f1300003ae27f033afbdff5e2683721ce",
        )
        self.assertEqual(manifest["patches"], ["0001-infoskill-hybrid-prefix.patch"])
        self.assertEqual(
            manifest["upstream_file_sha256"]["vllm/inputs/data.py"],
            "2a9a7f611cb79a709636403cdc1f48b20758c8c4dc3caa295adfa943e83bc237",
        )
        patch_path = PATCH_ROOT / manifest["patches"][0]
        self.assertEqual(
            hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            manifest["patch_sha256"][patch_path.name],
        )

    def test_patch_contains_the_complete_v1_transport_chain(self) -> None:
        patch = (PATCH_ROOT / "0001-infoskill-hybrid-prefix.patch").read_text(encoding="utf-8")

        required_paths = {
            "vllm/inputs/data.py",
            "vllm/inputs/preprocess.py",
            "vllm/v1/engine/__init__.py",
            "vllm/v1/engine/processor.py",
            "vllm/v1/request.py",
            "vllm/v1/core/sched/output.py",
            "vllm/v1/worker/gpu_input_batch.py",
            "vllm/v1/worker/gpu_model_runner.py",
        }
        for path in required_paths:
            self.assertIn(path, patch)
        self.assertIn("infoskill_prefix_embeds", patch)
        self.assertIn("infoskill_prefix_mask", patch)
        self.assertIn("INFOSKILL_HYBRID_PREFIX_API = 1", patch)
        self.assertIn("enforce_eager=True", patch)

    def test_build_validates_wheel_before_installing_it(self) -> None:
        script = (PROJECT_ROOT / "scripts" / "build_patched_vllm.sh").read_text(
            encoding="utf-8"
        )

        verify_position = script.index("prepare_vllm_wheel.py\" verify")
        install_position = script.index("pip install --force-reinstall")
        self.assertLess(verify_position, install_position)
        self.assertNotIn("export VLLM_USE_PRECOMPILED=1", script)

    def test_verification_hashes_the_pinned_git_blob_not_checkout_line_endings(
        self,
    ) -> None:
        canonical = b"first line\nsecond line\n"
        with patch(
            "scripts.verify_vllm_patch._git_bytes",
            return_value=canonical,
        ) as git_bytes:
            actual = _sha256_git_blob(Path("unused"), "fixed-commit", "tracked.py")

        self.assertEqual(actual, hashlib.sha256(canonical).hexdigest())
        git_bytes.assert_called_once_with(
            Path("unused"), "show", "fixed-commit:tracked.py"
        )


if __name__ == "__main__":
    unittest.main()
