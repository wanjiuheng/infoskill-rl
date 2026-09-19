from __future__ import annotations

import unittest

from infoskill.imitation.train import (
    _evaluation_strategy_kwargs,
    encode_training_example,
)


class _Tokenizer:
    eos_token_id = 99

    def __init__(self) -> None:
        self.chat_calls: list[object] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.chat_calls.append((messages, tokenize, add_generation_prompt))
        return [10, 11, 12]

    def __call__(self, text, **kwargs):
        return {"input_ids": [20, 21]}


class ImitationTrainingTests(unittest.TestCase):
    def test_evaluation_strategy_matches_installed_transformers_signature(self) -> None:
        class NewTrainingArguments:
            def __init__(self, output_dir, *, eval_strategy):
                pass

        class LegacyTrainingArguments:
            def __init__(self, output_dir, *, evaluation_strategy):
                pass

        self.assertEqual(
            _evaluation_strategy_kwargs(NewTrainingArguments),
            {"eval_strategy": "steps"},
        )
        self.assertEqual(
            _evaluation_strategy_kwargs(LegacyTrainingArguments),
            {"evaluation_strategy": "steps"},
        )

    def test_online_chat_prompt_is_masked_and_response_has_loss(self) -> None:
        tokenizer = _Tokenizer()
        encoded = encode_training_example(
            tokenizer,
            prompt="state",
            response="<action>go</action>",
            max_length=16,
        )

        self.assertEqual(encoded["input_ids"], [10, 11, 12, 20, 21, 99])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 20, 21, 99])
        self.assertEqual(
            tokenizer.chat_calls,
            [
                (
                    [{"role": "user", "content": "state"}],
                    True,
                    True,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
