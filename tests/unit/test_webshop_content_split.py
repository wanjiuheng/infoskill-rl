from __future__ import annotations

import unittest

from infoskill.imitation.dataset import content_grouped_validation_ids
from infoskill.imitation.providers import DemonstrationStep, DemonstrationTrajectory


def _trajectory(
    trajectory_id: str,
    *steps: tuple[str, str],
) -> DemonstrationTrajectory:
    return DemonstrationTrajectory(
        trajectory_id=trajectory_id,
        environment="webshop",
        steps=tuple(
            DemonstrationStep(prompt=prompt, action=action)
            for prompt, action in steps
        ),
    )


class WebShopContentGroupedSplitTests(unittest.TestCase):
    def test_trajectories_sharing_row_content_stay_in_the_same_split(self) -> None:
        shared = ("same prompt", "click[same]")
        trajectories = (
            _trajectory("a", shared, ("a only", "click[a]")),
            _trajectory("b", shared, ("b only", "click[b]")),
            _trajectory("c", ("c only", "click[c]")),
            _trajectory("d", ("d only", "click[d]")),
        )

        validation = content_grouped_validation_ids(
            trajectories,
            validation_fraction=0.25,
            split_seed=0,
        )

        self.assertEqual(len(validation), 1)
        self.assertEqual("a" in validation, "b" in validation)

    def test_exact_duplicate_trajectories_are_kept_together(self) -> None:
        steps = (("same prompt", "click[same]"),)
        trajectories = (
            _trajectory("a", *steps),
            _trajectory("b", *steps),
            _trajectory("c", ("c only", "click[c]")),
        )

        validation = content_grouped_validation_ids(
            trajectories,
            validation_fraction=1 / 3,
            split_seed=0,
        )

        self.assertEqual("a" in validation, "b" in validation)


if __name__ == "__main__":
    unittest.main()
