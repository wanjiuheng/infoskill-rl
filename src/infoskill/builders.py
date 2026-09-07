from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from infoskill.app_config import AppConfig
from infoskill.conditioning import (
    NoSkillConditioner,
    RawSkillPromptConditioner,
    SkillConditioner,
    format_raw_skill_block,
)
from infoskill.config import SkillMode
from infoskill.integrations.alfworld import AlfworldEnvironmentFactory
from infoskill.rollout import GenerationParameters, TransformersBackend
from infoskill.skills import (
    EmbeddingRetriever,
    FixedSkillLibrary,
    PrecomputedEmbeddingRetriever,
    SentenceTransformerEncoder,
    TemplateRetriever,
)


@dataclass(frozen=True, slots=True)
class RawSkillSetup:
    conditioner: SkillConditioner
    library: FixedSkillLibrary
    provenance: Mapping[str, object]
    skill_blocks: tuple[str, ...]

    def require_checkpoint_compatibility(
        self,
        checkpoint_provenance: Mapping[str, object],
    ) -> None:
        recorded = checkpoint_provenance.get("skill_conditioning")
        compatibility_keys = (
            "retrieval_schema_version",
            "retrieval_mode",
            "prompt_format",
            "skill_library_provenance_id",
            "skill_bank_sha256",
            "skill_count",
            "general_top_k",
            "task_top_k",
            "mistake_count",
            "history_length",
            "policy_prompt_schema_version",
        )
        if not isinstance(recorded, Mapping) or any(
            recorded.get(key) != self.provenance[key]
            for key in compatibility_keys
        ):
            raise RuntimeError(
                "portable checkpoint skill conditioning differs from evaluation"
            )
        current_semantic = self.provenance.get("semantic_model_identity")
        recorded_semantic = recorded.get("semantic_model_identity")
        for identity in (current_semantic, recorded_semantic):
            if identity is not None and not isinstance(identity, Mapping):
                raise RuntimeError(
                    "portable checkpoint semantic model identity is invalid"
                )
        if isinstance(current_semantic, Mapping) and (
            not isinstance(recorded_semantic, Mapping)
            or recorded_semantic.get("algorithm")
            != current_semantic.get("algorithm")
            or recorded_semantic.get("sha256") != current_semantic.get("sha256")
        ):
            raise RuntimeError(
                "portable checkpoint semantic model differs from evaluation"
            )
        current_plan = self.provenance.get("retrieval_plan")
        recorded_plan = recorded.get("retrieval_plan")
        if isinstance(current_plan, Mapping) and (
            not isinstance(recorded_plan, Mapping)
            or any(
                recorded_plan.get(task_id) != entry
                for task_id, entry in current_plan.items()
            )
        ):
            raise RuntimeError(
                "portable checkpoint raw skill retrieval plan differs from evaluation"
            )


def audit_raw_skill_prompt_budget(
    setup: RawSkillSetup,
    *,
    tokenizer: object,
    max_prompt_tokens: int,
) -> Mapping[str, int | bool]:
    """Audit complete raw blocks against the final prompt limit without truncation."""

    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    unique_blocks = tuple(dict.fromkeys(setup.skill_blocks))
    token_counts = tuple(
        len(tokenizer.encode(block, add_special_tokens=False))  # type: ignore[attr-defined]
        for block in unique_blocks
    )
    maximum = max(token_counts)
    return {
        "raw_skill_block_count": len(unique_blocks),
        "raw_skill_block_tokens_min": min(token_counts),
        "raw_skill_block_tokens_max": maximum,
        "raw_skill_prompt_token_limit": max_prompt_tokens,
        "raw_skill_block_fits_prompt_limit": maximum < max_prompt_tokens,
    }


def audit_raw_skill_prompt_budget_for_model(
    setup: RawSkillSetup,
    *,
    model_path: str,
    max_prompt_tokens: int,
) -> Mapping[str, int | bool]:
    """Load only the policy tokenizer and audit all distinct raw skill blocks."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("raw skill prompt audit requires transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return audit_raw_skill_prompt_budget(
        setup,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
    )


def build_raw_skill_setup(
    config: AppConfig,
    *,
    retrieval_queries: Mapping[str, str] | Sequence[str],
    embedding_device: str = "cuda:0",
    retrieval_mode: Literal["embedding", "template"] | None = None,
    prompt_format: Literal["full", "skillrl"] = "full",
) -> RawSkillSetup:
    """Build one immutable raw-skill conditioner for a complete run corpus."""

    effective_retrieval_mode = retrieval_mode or config.retrieval_mode
    if effective_retrieval_mode not in {"embedding", "template"}:
        raise ValueError("raw skill retrieval mode must be embedding or template")

    query_by_task_id = (
        dict(retrieval_queries)
        if isinstance(retrieval_queries, Mapping)
        else None
    )
    query_values = (
        tuple(query_by_task_id.values())
        if query_by_task_id is not None
        else tuple(retrieval_queries)
    )
    queries = tuple(dict.fromkeys(query_values))
    if not queries:
        raise ValueError("raw-skill setup requires at least one retrieval query")
    library = FixedSkillLibrary.load(config.paths.skill_bank)
    skill_library_provenance = _load_skill_library_provenance(
        library,
        manifest_path=config.paths.skill_bank_manifest,
    )
    semantic_model_identity = (
        _semantic_model_identity(config.paths.semantic_model)
        if effective_retrieval_mode == "embedding"
        else None
    )
    if effective_retrieval_mode == "embedding":
        encoder = SentenceTransformerEncoder(
            config.paths.semantic_model,
            device=embedding_device,
            show_progress_bar=True,
        )
        try:
            retriever = PrecomputedEmbeddingRetriever(
                library,
                encoder,
                queries=queries,
                general_top_k=config.general_top_k,
                task_top_k=config.task_top_k,
                mistake_count=config.mistake_count,
            )
        finally:
            encoder.close()
    else:
        retriever = TemplateRetriever(
            library,
            general_count=config.general_top_k,
            task_count=config.task_top_k,
            mistake_count=config.mistake_count,
        )
    retrieval_results = tuple(retriever.retrieve(query) for query in queries)
    result_by_query = {result.query: result for result in retrieval_results}
    retrieval_plan = [
        {
            "query": result.query,
            "skill_ids": list(result.skill_ids),
        }
        for result in retrieval_results
    ]
    retrieval_plan_sha256 = hashlib.sha256(
        json.dumps(
            retrieval_plan,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    retrieval_plan_by_task = (
        {
            task_id: {
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "skill_ids": list(result_by_query[query].skill_ids),
            }
            for task_id, query in query_by_task_id.items()
        }
        if query_by_task_id is not None
        else None
    )
    return RawSkillSetup(
        conditioner=RawSkillPromptConditioner(
            retriever,
            history_length=config.history_length,
            query_by_task_id=query_by_task_id,
            prompt_format=prompt_format,
        ),
        library=library,
        provenance={
            "retrieval_schema_version": 1,
            "retrieval_mode": effective_retrieval_mode,
            "prompt_format": prompt_format,
            "semantic_model_identity": semantic_model_identity,
            "skill_library_provenance_id": skill_library_provenance[
                "provenance_id"
            ],
            "skill_library_provenance": skill_library_provenance,
            "retrieval_query_count": len(queries),
            "retrieval_plan_sha256": retrieval_plan_sha256,
            "retrieval_plan": retrieval_plan_by_task,
            "skill_bank_sha256": library.source_sha256,
            "skill_count": (
                len(library.general)
                + len(library.task_specific)
                + len(library.mistakes)
            ),
            "general_top_k": config.general_top_k,
            "task_top_k": config.task_top_k,
            "mistake_count": config.mistake_count,
            "history_length": config.history_length,
            "policy_prompt_schema_version": 1,
        },
        skill_blocks=tuple(
            dict.fromkeys(
                format_raw_skill_block(result, style=prompt_format)
                for result in retrieval_results
            )
        ),
    )


def _load_skill_library_provenance(
    library: FixedSkillLibrary,
    *,
    manifest_path: str | None,
) -> dict[str, object]:
    resolved_manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "alfworld_skill_bank_manifest.json"
        )
    )
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("skill library provenance manifest must be an object")
    if payload.get("skill_bank_sha256_algorithm") != "canonical-json-sha256-v1":
        raise RuntimeError(
            "skill library provenance manifest uses an unsupported checksum algorithm"
        )
    if payload.get("skill_bank_sha256") != library.source_sha256:
        raise RuntimeError(
            "skill library provenance manifest does not match the configured bank"
        )
    if payload.get("source_split") != "train" or payload.get(
        "trajectory_count"
    ) != 223:
        raise RuntimeError(
            "ALFWorld skill library provenance must identify 223 train trajectories"
        )
    metadata = dict(library.metadata)
    if metadata.get("total_memories_analyzed") != 223:
        raise RuntimeError(
            "skill bank metadata does not identify 223 analyzed trajectories"
        )
    normalized = {**payload, "embedded_metadata": metadata}
    identity_payload = dict(normalized)
    identity_payload.pop("provenance_id", None)
    provenance_id = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    declared = payload.get("provenance_id")
    if declared not in {None, provenance_id}:
        raise RuntimeError("skill library provenance ID is invalid")
    normalized["provenance_id"] = provenance_id
    return normalized


def _semantic_model_identity(model_path: str) -> dict[str, object]:
    from infoskill.persistence.model_identity import fingerprint_policy_model

    identity = fingerprint_policy_model(model_path)
    return {
        "path": identity.path,
        "revision": Path(identity.path).name,
        "algorithm": identity.algorithm,
        "sha256": identity.sha256,
        "file_count": identity.file_count,
        "total_bytes": identity.total_bytes,
    }


def build_transformers_evaluation(
    config: AppConfig,
    *,
    mode: SkillMode,
    environment_backend: str = "native_batch",
    conditioner: SkillConditioner | None = None,
):
    if TransformersBackend is None:
        raise RuntimeError("Transformers evaluation requires torch and transformers")
    backend = TransformersBackend.from_pretrained(
        config.paths.policy_model,
        adapter_path=config.paths.policy_adapter,
        device="cuda:0",
        max_prompt_tokens=config.max_prompt_tokens,
    )
    factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=config.paths.alfworld_source,
        config_path=config.paths.alfworld_config,
        data_root=config.paths.alfworld_data,
        max_steps=config.max_steps,
    )
    if mode is SkillMode.NO_SKILL:
        if conditioner is not None:
            raise ValueError("no_skill evaluation does not accept a conditioner")
        conditioner = NoSkillConditioner()
    else:
        if mode is SkillMode.RAW_SKILL_PROMPT:
            if conditioner is None:
                raise ValueError(
                    "raw_skill_prompt evaluation requires its precomputed "
                    "conditioner"
                )
        else:
            if conditioner is not None:
                raise ValueError(
                    "infoskill evaluation constructs its checkpoint conditioner"
                )
            library = FixedSkillLibrary.load(config.paths.skill_bank)
            conditioner = _build_infoskill_conditioner(config, backend, library)
    from infoskill.episode import TrajectoryCollector

    return TrajectoryCollector(
        environment_factory=factory,
        conditioner=conditioner,
        rollout_backend=backend,
        max_steps=config.max_steps,
        history_limit=config.history_length,
        invalid_action_penalty=0.01,
        generation_parameters=GenerationParameters(
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=config.max_response_tokens,
        ),
        environment_workers=1,
        environment_backend=environment_backend,  # type: ignore[arg-type]
    )


def build_verl_no_skill_evaluation(
    config: AppConfig,
    *,
    backend: object,
    environment_backend: str = "native_batch",
):
    """Build the same deterministic no-skill collector used during M0 training."""

    return build_verl_policy_evaluation(
        config,
        mode=SkillMode.NO_SKILL,
        backend=backend,
        environment_backend=environment_backend,
    )


def build_verl_policy_evaluation(
    config: AppConfig,
    *,
    mode: SkillMode,
    backend: object,
    conditioner: SkillConditioner | None = None,
    environment_backend: str = "native_batch",
):
    """Build deterministic no-skill/raw-skill evaluation on the VERL backend."""

    if mode is SkillMode.NO_SKILL:
        if conditioner is not None:
            raise ValueError("no_skill evaluation does not accept a conditioner")
        conditioner = NoSkillConditioner()
    elif mode is SkillMode.RAW_SKILL_PROMPT:
        if conditioner is None:
            raise ValueError("raw_skill_prompt evaluation requires its conditioner")
    else:
        raise ValueError("VERL policy evaluation supports no_skill or raw_skill_prompt")

    factory = AlfworldEnvironmentFactory.from_paths(
        alfworld_source=config.paths.alfworld_source,
        config_path=config.paths.alfworld_config,
        data_root=config.paths.alfworld_data,
        max_steps=config.max_steps,
    )
    from infoskill.episode import TrajectoryCollector

    return TrajectoryCollector(
        environment_factory=factory,
        conditioner=conditioner,
        rollout_backend=backend,  # type: ignore[arg-type]
        max_steps=config.max_steps,
        history_limit=config.history_length,
        invalid_action_penalty=0.01,
        generation_parameters=GenerationParameters(
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=config.max_response_tokens,
        ),
        environment_workers=1,
        environment_backend=environment_backend,  # type: ignore[arg-type]
    )


def _build_retriever(config: AppConfig, library: FixedSkillLibrary, encoder: object):
    if config.retrieval_mode == "embedding":
        return EmbeddingRetriever(
            library,
            encoder,  # type: ignore[arg-type]
            general_top_k=config.general_top_k,
            task_top_k=config.task_top_k,
            mistake_count=config.mistake_count,
        )
    return TemplateRetriever(
        library,
        general_count=config.general_top_k,
        task_count=config.task_top_k,
        mistake_count=config.mistake_count,
    )


def _build_infoskill_conditioner(
    config: AppConfig, backend: object, library: FixedSkillLibrary
):
    if not config.paths.infoskill_checkpoint:
        raise ValueError("infoskill mode requires paths.infoskill_checkpoint")
    import torch

    from infoskill.conditioning import InfoSkillConditioner
    from infoskill.models import InfoSkillCompressor, LatentProjector
    from infoskill.semantic import FrozenSemanticEncoder, SemanticFeatureCache

    semantic = FrozenSemanticEncoder.from_pretrained(config.paths.semantic_model, device="cuda:0")
    retriever = _build_retriever(config, library, semantic)
    compressor = InfoSkillCompressor(semantic.hidden_size).to(semantic.device)
    policy_config = backend.model.config  # type: ignore[attr-defined]
    policy_width = int(getattr(policy_config, "hidden_size"))
    projector = LatentProjector(
        latent_dim=32, policy_hidden_size=policy_width, prefix_length=5
    ).to(semantic.device)
    payload = torch.load(config.paths.infoskill_checkpoint, map_location=semantic.device, weights_only=True)
    compressor.load_state_dict(payload["compressor"])
    projector.load_state_dict(payload["projector"])
    compressor.eval()
    projector.eval()
    return InfoSkillConditioner(
        retriever=retriever,  # type: ignore[arg-type]
        semantic_encoder=semantic,
        feature_cache=SemanticFeatureCache(semantic),
        compressor=compressor,
        projector=projector,
        latent_mode="mean",
    )
