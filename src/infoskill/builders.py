from __future__ import annotations

from infoskill.app_config import AppConfig
from infoskill.conditioning import NoSkillConditioner, RawSkillPromptConditioner
from infoskill.config import SkillMode
from infoskill.integrations.alfworld import AlfworldEnvironmentFactory
from infoskill.rollout import GenerationParameters, TransformersBackend
from infoskill.skills import EmbeddingRetriever, FixedSkillLibrary, SentenceTransformerEncoder, TemplateRetriever


def build_transformers_evaluation(config: AppConfig, *, mode: SkillMode):
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
        conditioner = NoSkillConditioner()
    else:
        library = FixedSkillLibrary.load(config.paths.skill_bank)
        if mode is SkillMode.RAW_SKILL_PROMPT:
            retriever = _build_retriever(
                config,
                library,
                SentenceTransformerEncoder(config.paths.semantic_model, device="cuda:0"),
            )
            conditioner = RawSkillPromptConditioner(
                retriever, history_length=config.history_length
            )
        else:
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
