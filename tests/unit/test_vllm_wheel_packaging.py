from __future__ import annotations

import unittest

from scripts.prepare_vllm_wheel import inspect_wheel_entries, repackage_entries


class VllmWheelPackagingTests(unittest.TestCase):
    def test_python_only_wheel_is_rejected_before_install(self) -> None:
        entries = {
            "vllm/inputs/data.py": b"INFOSKILL_HYBRID_PREFIX_API = 1\n",
            "vllm-0.8.4+infoskill1.dist-info/METADATA": (
                b"Name: vllm\nVersion: 0.8.4+infoskill1\n"
            ),
        }

        with self.assertRaisesRegex(RuntimeError, "native extension"):
            inspect_wheel_entries(entries)

    def test_repackage_retains_native_extension_and_overlays_patched_python(
        self,
    ) -> None:
        base = {
            "vllm/_C.abi3.so": b"compiled extension",
            "vllm/_version.py": b"__version__ = '0.8.4'\n",
            "vllm/inputs/data.py": b"upstream\n",
            "vllm-0.8.4.dist-info/METADATA": (
                b"Metadata-Version: 2.1\nName: vllm\nVersion: 0.8.4\n"
            ),
            "vllm-0.8.4.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
            "vllm-0.8.4.dist-info/RECORD": b"old record\n",
        }
        patched = {
            "vllm/inputs/data.py": b"INFOSKILL_HYBRID_PREFIX_API = 1\n"
        }

        result = repackage_entries(base, patched, "0.8.4+infoskill1")

        self.assertEqual(result["vllm/_C.abi3.so"], b"compiled extension")
        self.assertEqual(result["vllm/inputs/data.py"], patched["vllm/inputs/data.py"])
        self.assertIn(
            "vllm-0.8.4+infoskill1.dist-info/METADATA",
            result,
        )
        self.assertNotIn("vllm-0.8.4.dist-info/METADATA", result)
        self.assertNotIn("vllm-0.8.4.dist-info/RECORD", result)
        self.assertIn(b"Version: 0.8.4+infoskill1\n", result[
            "vllm-0.8.4+infoskill1.dist-info/METADATA"
        ])
        self.assertIn(b"0.8.4+infoskill1", result["vllm/_version.py"])
        inspect_wheel_entries(result)


if __name__ == "__main__":
    unittest.main()
