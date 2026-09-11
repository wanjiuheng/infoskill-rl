from __future__ import annotations

import unittest
from dataclasses import dataclass

from infoskill.conditioning import InfoSkillReplayTrace
from infoskill.training.auxiliary_work import (
    OfflineAuxiliaryExample,
    OnlineAuxiliaryExample,
    partition_auxiliary_work,
)


@dataclass(frozen=True)
class _Sample:
    name: str


class AuxiliaryWorkTests(unittest.TestCase):
    def test_partition_preserves_real_examples_and_equalizes_calls(self) -> None:
        trace = InfoSkillReplayTrace(1, None, None, None, None, None, None)
        online = tuple(
            OnlineAuxiliaryExample(
                replay_trace=trace,
                candidate_skill_ids=("skill",),
                fidelity_target=float(index),
                trajectory_index=index // 2,
                trajectory_weight=0.5,
            )
            for index in range(10)
        )
        offline = tuple(
            OfflineAuxiliaryExample(  # type: ignore[arg-type]
                sample=_Sample(str(index)),
                epsilon_seed=index,
            )
            for index in range(7)
        )

        work = partition_auxiliary_work(
            online=online,
            offline=offline,
            world_size=3,
            micro_batch_size=2,
        )

        self.assertEqual([len(item.online) for item in work], [4, 4, 4])
        self.assertEqual([len(item.offline) for item in work], [4, 4, 4])
        self.assertEqual(
            sum(row.eligible for item in work for row in item.online),
            len(online),
        )
        self.assertEqual(
            sum(row.eligible for item in work for row in item.offline),
            len(offline),
        )
        self.assertTrue(all(item.global_trajectory_count == 5 for item in work))
        self.assertTrue(all(item.global_offline_count == 7 for item in work))

    def test_partition_is_deterministic(self) -> None:
        trace = InfoSkillReplayTrace(1, None, None, None, None, None, None)
        online = (
            OnlineAuxiliaryExample(trace, ("skill",), 1.0, 0, 1.0),
        )
        offline = (
            OfflineAuxiliaryExample(  # type: ignore[arg-type]
                sample=_Sample("sample"),
                epsilon_seed=7,
            ),
        )

        first = partition_auxiliary_work(
            online=online,
            offline=offline,
            world_size=4,
            micro_batch_size=2,
        )
        second = partition_auxiliary_work(
            online=online,
            offline=offline,
            world_size=4,
            micro_batch_size=2,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
