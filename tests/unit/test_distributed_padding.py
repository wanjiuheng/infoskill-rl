from __future__ import annotations

import unittest

from infoskill.distributed import pad_batch_to_divisor


class _FakeBatch:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, item):
        return _FakeBatch(self.values[item])

    @classmethod
    def concat(cls, batches: list["_FakeBatch"]) -> "_FakeBatch":
        return cls([item for batch in batches for item in batch.values])


class DistributedPaddingTests(unittest.TestCase):
    def test_41_environment_steps_are_padded_for_four_workers(self) -> None:
        original = _FakeBatch(list(range(41)))

        padded, padding_count = pad_batch_to_divisor(original, 4)

        self.assertEqual(len(original), 41)
        self.assertEqual(len(padded), 44)
        self.assertEqual(padding_count, 3)
        self.assertEqual(padded.values[-3:], [0, 1, 2])

    def test_batch_smaller_than_divisor_repeats_until_divisible(self) -> None:
        padded, padding_count = pad_batch_to_divisor(_FakeBatch([7]), 4)

        self.assertEqual(padded.values, [7, 7, 7, 7])
        self.assertEqual(padding_count, 3)

    def test_already_divisible_batch_is_returned_unchanged(self) -> None:
        original = _FakeBatch([1, 2, 3, 4])

        padded, padding_count = pad_batch_to_divisor(original, 4)

        self.assertIs(padded, original)
        self.assertEqual(padding_count, 0)


if __name__ == "__main__":
    unittest.main()
