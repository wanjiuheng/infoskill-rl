from __future__ import annotations

import unittest

from infoskill.distributed import pad_batch_to_divisor, policy_rank_balanced_order


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

    def test_rank_balance_preserves_each_synchronized_minibatch(self) -> None:
        token_counts = [10, 8, 6, 4, 1, 3, 5, 7]

        def balanced_pairs(lengths, partitions, equal_size):
            self.assertEqual(len(lengths), 4)
            self.assertEqual(partitions, 2)
            self.assertTrue(equal_size)
            return [[0, 2], [1, 3]]

        order = policy_rank_balanced_order(
            token_counts,
            world_size=2,
            global_minibatch_size=4,
            partitioner=balanced_pairs,
        )

        self.assertEqual(order, (0, 4, 2, 6, 1, 5, 3, 7))
        self.assertEqual(
            sum(token_counts[index] for index in order[:4]),
            sum(token_counts[index] for index in order[4:]),
        )
        self.assertEqual(set(order[0:2] + order[4:6]), {0, 1, 4, 5})
        self.assertEqual(set(order[2:4] + order[6:8]), {2, 3, 6, 7})

    def test_rank_balance_matches_verl_floor_for_non_divisible_minibatch(self) -> None:
        calls = []

        def one_per_rank(lengths, partitions, equal_size):
            calls.append(tuple(lengths))
            self.assertEqual(partitions, 2)
            self.assertTrue(equal_size)
            return [[0], [1]]

        order = policy_rank_balanced_order(
            [1, 2, 3, 4],
            world_size=2,
            global_minibatch_size=3,
            partitioner=one_per_rank,
        )

        self.assertEqual(order, (0, 1, 2, 3))
        self.assertEqual(calls, [(1, 3), (2, 4)])


if __name__ == "__main__":
    unittest.main()
