from __future__ import annotations

import unittest

from infoskill.conditioning import (
    ConditioningContext,
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

    def test_groups_share_one_runtime_call_and_keep_per_task_candidates(self) -> None:
        first = _request("task-1", latent_seed=11)
        second = _request("task-2", latent_seed=12)
        trace = InfoSkillReplayTrace(
            latent_seed=11,
            state_summary="summary",
            state_tokens="tokens",
            posterior_mu="mu",
            posterior_logvar="logvar",
            latent="latent",
            epsilon="epsilon",
        )
        runtime = _Runtime(
            outputs=(
                InfoSkillConditioningResult("prefix-1", trace),
                InfoSkillConditioningResult("prefix-2", trace),
            )
        )
        conditioner = RuntimeInfoSkillConditioner(
            retriever=_Retriever(
                RetrievalResult("embedding", "goal", ())
            ),
            runtime=runtime,
            latent_mode="mean",
            group_across_tasks=True,
        )
        contexts = (
            ConditioningContext(
                candidate_skill_ids=("skill-1",),
                retrieval=RetrievalResult("embedding", "goal-1", ()),
            ),
            ConditioningContext(
                candidate_skill_ids=("skill-2", "skill-3"),
                retrieval=RetrievalResult("embedding", "goal-2", ()),
            ),
        )

        grouped = conditioner.condition_groups(
            ((first,), (second,)),
            contexts,
        )

        self.assertEqual(runtime.grouped_calls, 1)
        self.assertEqual(
            runtime.call[1],
            (("skill-1",), ("skill-2", "skill-3")),
        )
        self.assertEqual(
            tuple(item[0].candidate_skill_ids for item in grouped),
            (("skill-1",), ("skill-2", "skill-3")),
        )

    def test_grouping_is_opt_in_and_default_preserves_per_task_calls(self) -> None:
        first = _request("task-1", latent_seed=11)
        second = _request("task-2", latent_seed=12)
        trace = InfoSkillReplayTrace(
            latent_seed=11,
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
        conditioner = RuntimeInfoSkillConditioner(
            retriever=_Retriever(RetrievalResult("embedding", "goal", ())),
            runtime=runtime,
            latent_mode="mean",
        )
        contexts = (
            ConditioningContext(
                candidate_skill_ids=("skill-1",),
                retrieval=RetrievalResult("embedding", "goal-1", ()),
            ),
            ConditioningContext(
                candidate_skill_ids=("skill-2",),
                retrieval=RetrievalResult("embedding", "goal-2", ()),
            ),
        )

        conditioner.condition_groups(((first,), (second,)), contexts)

        self.assertEqual(runtime.grouped_calls, 0)
        self.assertEqual(runtime.regular_calls, 2)


def _request(task_id: str, *, latent_seed: int) -> ConditioningRequest:
    state = CanonicalAgentState(
        task_id=task_id,
        split="train",
        task_type="pick_and_place_simple",
        goal="goal",
        step_index=0,
        observation="room",
        history=(),
        admissible_commands=("look",),
    )
    return ConditioningRequest(
        state=state,
        views=render_state_views(state),
        rollout_id=0,
        global_update=0,
        latent_seed=latent_seed,
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
        self.grouped_calls = 0
        self.regular_calls = 0

    def condition_infoskill(self, requests, candidate_skill_ids, *, latent_mode):
        self.regular_calls += 1
        self.call = (requests, candidate_skill_ids, latent_mode)
        return self.outputs

    def condition_infoskill_grouped(
        self,
        requests,
        candidate_skill_ids_by_request,
        *,
        latent_mode,
    ):
        self.grouped_calls += 1
        self.call = (requests, candidate_skill_ids_by_request, latent_mode)
        return self.outputs


if __name__ == "__main__":
    unittest.main()
