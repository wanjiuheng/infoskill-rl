from __future__ import annotations

import unittest

from infoskill.integrations.verl.generation_boundary import (
    trim_vllm_padding_sentinel,
)


class VerlGenerationBoundaryTests(unittest.TestCase):
    def test_terminal_padding_with_sentinel_logprob_is_removed(self) -> None:
        tokens, logprobs = trim_vllm_padding_sentinel(
            token_ids=(10, 20, 151643),
            token_logprobs=(-0.2, -0.3, -1.0),
            pad_token_id=151643,
        )

        self.assertEqual(tokens, (10, 20))
        self.assertEqual(logprobs, (-0.2, -0.3))

    def test_real_generated_special_token_is_preserved(self) -> None:
        tokens, logprobs = trim_vllm_padding_sentinel(
            token_ids=(10, 151643),
            token_logprobs=(-0.2, -0.75),
            pad_token_id=151643,
        )

        self.assertEqual(tokens, (10, 151643))
        self.assertEqual(logprobs, (-0.2, -0.75))

    def test_minus_one_logprob_on_ordinary_token_is_preserved(self) -> None:
        tokens, logprobs = trim_vllm_padding_sentinel(
            token_ids=(10, 20),
            token_logprobs=(-0.2, -1.0),
            pad_token_id=151643,
        )

        self.assertEqual(tokens, (10, 20))
        self.assertEqual(logprobs, (-0.2, -1.0))


if __name__ == "__main__":
    unittest.main()
