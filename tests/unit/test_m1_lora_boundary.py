import unittest
from unittest.mock import patch

from infoskill.m1_lora_boundary import (
    VllmBoundaryCapture,
    build_boundary_isolation_report,
    compare_boundary_rounds,
)


def row(*, history="same", raw=1.0, processed=1.0, token=7, logprob=-0.2):
    return {
        "history_sha256": history,
        "raw": {"top_values": [raw], "logsumexp": raw + 0.2},
        "processed": {"top_values": [processed], "logsumexp": processed + 0.2},
        "sampled_token_id": token,
        "returned_logprob": logprob,
    }


class M1LoraBoundaryTests(unittest.TestCase):
    def test_raw_logit_is_first_changed_boundary(self):
        report = compare_boundary_rounds(((row(),), (row(raw=1.1),)))
        self.assertEqual(report["first_changed_boundary"], "raw_logits")
        self.assertEqual(report["comparable_rows"], 1)

    def test_processed_logit_is_first_changed_boundary(self):
        report = compare_boundary_rounds(((row(),), (row(processed=1.1),)))
        self.assertEqual(report["first_changed_boundary"], "processed_logits")

    def test_sampler_or_logprob_boundary_is_distinguished(self):
        token_report = compare_boundary_rounds(((row(),), (row(token=8),)))
        self.assertEqual(token_report["first_changed_boundary"], "sampled_token")
        logprob_report = compare_boundary_rounds(((row(),), (row(logprob=-0.3),)))
        self.assertEqual(logprob_report["first_changed_boundary"], "returned_logprob")

    def test_different_history_is_not_compared_after_a_branch(self):
        report = compare_boundary_rounds(((row(),), (row(history="branch", raw=9.0),)))
        self.assertEqual(report["comparable_rows"], 0)
        self.assertEqual(report["first_changed_boundary"], "insufficient_common_history")

    def test_scoped_hooks_restore_all_original_methods(self):
        class Tensor:
            def detach(self): return self
            def cpu(self): return self
            def reshape(self, _size): return self
            def tolist(self): return [7]

        class Output:
            sampled_token_ids = Tensor()
            logprobs_tensors = None

        class Sampler:
            def sample(self, logits, metadata):
                return Tensor()

        class Model:
            def __init__(self): self.sampler = Sampler()
            def compute_logits(self, hidden, metadata): return "raw logits"
            def sample(self, *, logits, sampling_metadata):
                self.sampler.sample(logits, sampling_metadata)
                return Output()

        class Request:
            prompt_token_ids = [1, 2]
            output_token_ids = []

        class InputBatch:
            req_ids = ["a"]

        class Runner:
            def __init__(self):
                self.model = Model()
                self.input_batch = InputBatch()
                self.requests = {"a": Request()}

        runner = Runner()
        capture = VllmBoundaryCapture(runner, rank=0)
        with patch("infoskill.m1_lora_boundary._logit_summary", return_value=[{
            "top_values": [1.0], "logsumexp": 1.2, "top_token_ids": [7]
        }]):
            capture.install()
            runner.model.sample(
                logits=runner.model.compute_logits(None, None),
                sampling_metadata=None,
            )
            rows = capture.take()
            self.assertEqual(rows[0]["sampled_token_id"], 7)
            self.assertEqual(rows[0]["rank"], 0)
            capture.remove()
        self.assertEqual(runner.model.compute_logits(None, None), "raw logits")
        self.assertIsInstance(runner.model.sampler.sample(None, None), Tensor)

    def test_base_control_drift_prevents_lora_attribution(self):
        from types import SimpleNamespace
        samples = [
            SimpleNamespace(label=label, cells=[SimpleNamespace(
                name="token-full3",
                boundary_rounds=((row(),), (row(raw=1.1),)),
            )])
            for label in (
                "checkpoint-eager", "base-eager", "checkpoint-graph", "base-graph"
            )
        ]
        report = build_boundary_isolation_report(samples)
        self.assertEqual(
            report["classification"], "base_control_drift_blocks_lora_attribution"
        )


if __name__ == "__main__":
    unittest.main()
