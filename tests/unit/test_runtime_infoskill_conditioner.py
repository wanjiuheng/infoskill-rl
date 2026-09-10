from __future__ import annotations

import unittest

from infoskill.conditioning import (
    ConditioningRequest,
    InfoSkillConditioningResult,
    InfoSkillConditioningWorkItem,
    InfoSkillReplayTrace,
    RuntimeInfoSkillConditioner,
)
from infoskill.domain.state import CanonicalAgentState, render_state_views
from infoskill.skills import RetrievalResult, SkillRecord
from infoskill.skills.retrieval import RetrievedSkill


class RuntimeInfoSkillConditionerTests(unittest.TestCase):
    def test_work_item_rejects_unknown_latent_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "latent mode"):
            InfoSkillConditioningWorkItem(
                compression_view="state",
                candidate_skill_ids=("skill-1",),
                latent_seed=0,
                latent_mode="unknown",  # type: ignore[arg-type]
            )

    def test_worker_runtime_owns_prefix_generation_for_fixed_episode_skills(self) -> None:
        skill = SkillRecord(
            skill_id="general-1",
            kind="general",
            category=None,
            title="Inspect first",
            text="inspect before acting",
            fields={},
        )
        retriever = _Retriever(
            RetrievalResult(
                mode="embedding",
                query="put an object somewhere",
                skills=(RetrievedSkill(skill, 0.9),),
            )
        )
        trace = InfoSkillReplayTrace(
            latent_seed=17,
            state_summary="summary",
            state_tokens="tokens",
            posterior_mu="mu",
            posterior_logvar="logvar",
            latent="latent",
            epsilon="epsilon",
        )
        runtime = _Runtime(
            outputs=(InfoSkillConditioningResult("prefix", trace),)
        )
        state = CanonicalAgentState(
            task_id="task-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="put the object in the cabinet.",
            step_index=0,
            observation="You are in a room.",
            history=(),
            admissible_commands=("look",),
        )
        request = ConditioningRequest(
            state=state,
            views=render_state_views(state),
            rollout_id=2,
            global_update=3,
            latent_seed=17,
        )
        conditioner = RuntimeInfoSkillConditioner(
            retriever=retriever,
            runtime=runtime,
            latent_mode="sample",
            query_by_task_id={"task-1": "put an object somewhere"},
        )

        context = conditioner.prepare_group(state)
        result = conditioner.condition_batch((request,), context)

        self.assertEqual(context.candidate_skill_ids, ("general-1",))
        self.assertEqual(runtime.call[1], ("general-1",))
        self.assertEqual(runtime.call[2], "sample")
        self.assertEqual(result[0].user_message, request.views.policy_view)
        self.assertEqual(result[0].candidate_skill_ids, ("general-1",))
        self.assertEqual(result[0].soft_prefix, "prefix")
        self.assertIs(result[0].conditioning_trace, trace)

    def test_rejects_runtime_output_cardinality_mismatch(self) -> None:
        skill = SkillRecord(
            skill_id="general-1",
            kind="general",
            category=None,
            title="Inspect first",
            text="inspect before acting",
            fields={},
        )
        retriever = _Retriever(
            RetrievalResult(
                "embedding",
                "goal",
                (RetrievedSkill(skill, 0.9),),
            )
        )
        runtime = _Runtime(outputs=())
        state = CanonicalAgentState(
            task_id="task-1",
            split="train",
            task_type="pick_and_place_simple",
            goal="goal",
            step_index=0,
            observation="room",
            history=(),
            admissible_commands=("look",),
        )
        request = ConditioningRequest(
            state=state,
            views=render_state_views(state),
            rollout_id=0,
            global_update=0,
            latent_seed=0,
        )
        conditioner = RuntimeInfoSkillConditioner(
            retriever=retriever,
            runtime=runtime,
        )

        with self.assertRaisesRegex(RuntimeError, "one output per request"):
            conditioner.condition_batch(
                (request,),
                conditioner.prepare_group(state),
            )


class _Retriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result

    def retrieve(self, query: str) -> RetrievalResult:
        if query != self.result.query:
            raise AssertionError(query)
        return self.result


class _Runtime:
    def __init__(self, *, outputs: tuple[InfoSkillConditioningResult, ...]) -> None:
        self.outputs = outputs
        self.call = None

    def condition_infoskill(self, requests, candidate_skill_ids, *, latent_mode):
        self.call = (requests, candidate_skill_ids, latent_mode)
        return self.outputs


if __name__ == "__main__":
    unittest.main()
