from __future__ import annotations

import contextlib
import io
import json
import unittest

from infoskill import runtime_doctor


class _StructSamplingParams:
    __struct_fields__ = ("temperature", "seed", "stop", "include_stop_str_in_output")

    def __init__(
        self,
        temperature: float = 1.0,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        detokenize: bool = False,
        include_stop_str_in_output: bool = False,
    ) -> None:
        self.temperature = temperature
        self.seed = seed
        self.stop = stop
        self.detokenize = detokenize
        self.include_stop_str_in_output = include_stop_str_in_output


class RuntimeDoctorTests(unittest.TestCase):
    def test_action_stop_probe_checks_runtime_sampling_constructor(self) -> None:
        self.assertTrue(runtime_doctor._action_stop_roundtrip(_StructSamplingParams))

    def test_sampling_fields_support_msgspec_structs(self) -> None:
        fields = runtime_doctor._sampling_param_fields(_StructSamplingParams)

        self.assertIn("seed", fields)
        self.assertIn("stop", fields)
        self.assertIn("include_stop_str_in_output", fields)

    def test_main_emits_json_even_when_collection_prints_logs(self) -> None:
        original = runtime_doctor.collect_report

        def noisy_report() -> dict[str, object]:
            print("third-party import log")
            return {"ok": True}

        runtime_doctor.collect_report = noisy_report
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                result = runtime_doctor.main()
        finally:
            runtime_doctor.collect_report = original

        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["captured_import_logs"], ["third-party import log"])

    def test_hybrid_prefix_capability_requires_marker_and_both_fields(self) -> None:
        self.assertFalse(runtime_doctor._has_hybrid_prefix_api(None, ()))
        self.assertFalse(
            runtime_doctor._has_hybrid_prefix_api(1, ("infoskill_prefix_embeds",))
        )
        self.assertTrue(
            runtime_doctor._has_hybrid_prefix_api(
                1,
                ("infoskill_prefix_embeds", "infoskill_prefix_mask"),
            )
        )


if __name__ == "__main__":
    unittest.main()
