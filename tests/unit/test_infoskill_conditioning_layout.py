from __future__ import annotations

import unittest


class InfoSkillConditioningLayoutTests(unittest.TestCase):
    def test_worker_rows_are_conditioned_one_at_a_time(self) -> None:
        from infoskill.conditioning.distributed_layout import (
            _condition_rows_individually,
        )

        calls: list[tuple[str, ...]] = []

        def condition(rows: tuple[str, ...]) -> tuple[str, ...]:
            calls.append(rows)
            return tuple(f"conditioned-{row}" for row in rows)

        results = _condition_rows_individually(
            ("first", "second"),
            condition,
        )

        self.assertEqual(calls, [("first",), ("second",)])
        self.assertEqual(results, ("conditioned-first", "conditioned-second"))

    def test_grouped_rpc_replicates_the_logical_sequence_on_every_rank(self) -> None:
        from infoskill.conditioning.distributed_layout import (
            _rank_replicated_conditioning_rows,
        )

        rows = _rank_replicated_conditioning_rows(("first", "second"), 3)

        self.assertEqual(
            rows,
            ("first", "second", "first", "second", "first", "second"),
        )

    def test_grouped_rpc_selects_the_rank_zero_copy(self) -> None:
        from infoskill.conditioning.distributed_layout import (
            _select_rank_zero_conditioning_rows,
        )

        selected = _select_rank_zero_conditioning_rows(
            ("rank0-first", "rank0-second", "rank1-first", "rank1-second"),
            logical_size=2,
            world_size=2,
        )

        self.assertEqual(selected, ("rank0-first", "rank0-second"))


if __name__ == "__main__":
    unittest.main()
