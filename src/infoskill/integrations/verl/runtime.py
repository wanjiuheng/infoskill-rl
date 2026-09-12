from __future__ import annotations

import hashlib
import logging
import math
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import ray

from infoskill.conditioning import (
    ConditioningRequest,
    InfoSkillConditioningResult,
    InfoSkillConditioningWorkItem,
)
from infoskill.conditioning.distributed_layout import (
    _rank_replicated_conditioning_rows,
    _select_rank_zero_conditioning_rows,
)
from infoskill.config import DEFAULT_POLICY_MAX_TOKENS_PER_GPU
from infoskill.distributed import pad_batch_to_divisor, policy_rank_balanced_order
from infoskill.episode import TrajectoryGroup
from infoskill.learning import require_logprob_alignment, summarize_logprob_alignment
from infoskill.rollout import GenerationRequest, GenerationResult

from .codec import VerlBatchCodec
from .compatibility import require_vllm_084_cachetools_compatibility
from .hybrid_rollout import vllm_action_stop_settings
from .memory_metrics import (
    summarize_cuda_memory_snapshots,
    summarize_rank_token_load,
)
from .portable_load import load_portable_state_after_base_sync


@dataclass(frozen=True, slots=True)
class VerlRuntimeConfig:
    skillrl_source: str
    model_path: str
    num_gpus: int
    num_cpus: int = 96
    max_prompt_tokens: int = 4096
    max_response_tokens: int = 256
    total_training_steps: int = 445
    lora_rank: int = 16
    lora_alpha: int = 32
    actor_learning_rate: float = 1e-6
    action_minibatch_size: int = 256
    policy_max_tokens_per_gpu: int = DEFAULT_POLICY_MAX_TOKENS_PER_GPU
    rollout_max_batched_tokens: int = 16_384
    gpu_memory_utilization: float = 0.50
    allow_unkeyed_vllm_sampling: bool = False
    require_hybrid_prefix: bool = False
    soft_prefix_length: int = 5
    master_seed: int = 0
    persistent_rollout_session: bool = True
    verbose_runtime_logs: bool = False
    cuda_memory_poll_interval_ms: int = 0
    balance_policy_tokens_across_ranks: bool = True
    enable_infoskill_modules: bool = False
    semantic_model_path: str | None = None
    skill_bank_path: str | None = None
    infoskill_latent_dim: int = 32
    infoskill_initialization_seed: int = 0
    infoskill_projector_learning_rate: float = 1e-4
    infoskill_projector_weight_decay: float = 0.01
    infoskill_policy_warmup_ratio: float = 0.03
    enable_infoskill_auxiliary: bool = False
    grounding_data_path: str | None = None
    infoskill_history_length: int = 2
    infoskill_auxiliary_micro_batch_size: int = 8
    infoskill_offline_games_per_update: int = 256
    infoskill_auxiliary_learning_rate: float = 1e-4
    infoskill_auxiliary_weight_decay: float = 0.01
    infoskill_auxiliary_warmup_ratio: float = 0.03
    infoskill_fidelity_weight: float = 1.0
    infoskill_rate_weight: float = 0.001
    infoskill_grounding_weight: float = 0.1
    infoskill_auxiliary_max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.soft_prefix_length <= 0:
            raise ValueError("soft prefix length must be positive")
        if self.enable_infoskill_auxiliary and not self.enable_infoskill_modules:
            raise ValueError(
                "INFO-SKILL auxiliary training requires INFO-SKILL modules"
            )
        if not self.enable_infoskill_modules:
            return
        if not self.require_hybrid_prefix:
            raise ValueError(
                "INFO-SKILL runtime modules require hybrid-prefix rollout"
            )
        if not self.semantic_model_path or not self.semantic_model_path.strip():
            raise ValueError("INFO-SKILL runtime requires a semantic model path")
        if not self.skill_bank_path or not self.skill_bank_path.strip():
            raise ValueError("INFO-SKILL runtime requires a skill bank path")
        if self.infoskill_latent_dim <= 0:
            raise ValueError("INFO-SKILL latent dimension must be positive")
        if self.infoskill_initialization_seed < 0:
            raise ValueError("INFO-SKILL initialization seed must be non-negative")
        if self.infoskill_projector_learning_rate <= 0:
            raise ValueError("INFO-SKILL projector learning rate must be positive")
        if self.infoskill_projector_weight_decay < 0:
            raise ValueError("INFO-SKILL projector weight decay must be non-negative")
        if not 0 <= self.infoskill_policy_warmup_ratio <= 1:
            raise ValueError("INFO-SKILL policy warmup ratio must be in [0, 1]")
        if not self.enable_infoskill_auxiliary:
            return
        if not self.grounding_data_path or not self.grounding_data_path.strip():
            raise ValueError("INFO-SKILL auxiliary training requires grounding data")
        if self.infoskill_history_length < 0:
            raise ValueError("INFO-SKILL history length must be non-negative")
        if min(
            self.infoskill_auxiliary_micro_batch_size,
            self.infoskill_offline_games_per_update,
        ) <= 0:
            raise ValueError("INFO-SKILL auxiliary batch sizes must be positive")
        if self.infoskill_auxiliary_learning_rate <= 0:
            raise ValueError("INFO-SKILL auxiliary learning rate must be positive")
        if self.infoskill_auxiliary_weight_decay < 0:
            raise ValueError("INFO-SKILL auxiliary weight decay must be non-negative")
        if not 0 <= self.infoskill_auxiliary_warmup_ratio <= 1:
            raise ValueError("INFO-SKILL auxiliary warmup ratio must be in [0, 1]")
        if min(
            self.infoskill_fidelity_weight,
            self.infoskill_rate_weight,
            self.infoskill_grounding_weight,
        ) < 0:
            raise ValueError("INFO-SKILL auxiliary weights must be non-negative")
        if self.infoskill_auxiliary_max_grad_norm <= 0:
            raise ValueError("INFO-SKILL auxiliary grad norm must be positive")


class VerlRuntime:
    """Own only worker initialization, generation, logprobs, and LoRA updates."""

    def __init__(
        self,
        *,
        worker_group: object,
        codec: VerlBatchCodec,
        config: VerlRuntimeConfig,
    ) -> None:
        self.worker_group = worker_group
        self.codec = codec
        self.config = config
        self._completed_updates = 0
        self._generation_calls = 0
        self._generation_requests = 0
        self._generation_seconds = 0.0
        self._generation_worker_seconds = 0.0
        self._rollout_session_active = False
        self._grounding_dataset = None
        if config.enable_infoskill_auxiliary:
            from infoskill.integrations.alfworld import GroundingDataset

            self._grounding_dataset = GroundingDataset.load(
                config.grounding_data_path
            )

    @classmethod
    def start(cls, config: VerlRuntimeConfig) -> "VerlRuntime":
        if config.num_gpus <= 0:
            raise ValueError("VERL runtime requires at least one GPU")
        require_vllm_084_cachetools_compatibility()
        if config.require_hybrid_prefix:
            _require_hybrid_prefix_runtime()
        source = str(Path(config.skillrl_source).expanduser().resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from omegaconf import OmegaConf
        from verl.single_controller.ray import (
            RayClassWithInitArgs,
            RayResourcePool,
            RayWorkerGroup,
            create_colocated_worker_cls_fused,
        )
        from verl.utils import hf_tokenizer
        from .worker import PortableActorRolloutRefWorker

        runtime_config = _actor_config(config)
        started_ray = not ray.is_initialized()
        if started_ray:
            ray.init(
                num_cpus=config.num_cpus,
                num_gpus=config.num_gpus,
                ignore_reinit_error=True,
                log_to_driver=config.verbose_runtime_logs,
                logging_level=(
                    logging.INFO if config.verbose_runtime_logs else logging.ERROR
                ),
            )
        try:
            pool = RayResourcePool(
                process_on_nodes=[config.num_gpus],
                use_gpu=True,
                max_colocate_count=1,
                name_prefix="infoskill",
            )
            actor = RayClassWithInitArgs(
                cls=ray.remote(PortableActorRolloutRefWorker),
                config=runtime_config.actor_rollout_ref,
                role="actor_rollout",
            )
            classes = {"actor_rollout": actor}
            colocated = create_colocated_worker_cls_fused(class_dict=classes)
            group = RayWorkerGroup(
                resource_pool=pool,
                ray_cls_with_init=colocated,
                device_name="cuda",
            )
            worker_group = group.spawn(prefix_set=classes.keys())["actor_rollout"]
            worker_group.init_model()
            tokenizer = hf_tokenizer(config.model_path, trust_remote_code=True)
            codec = VerlBatchCodec(
                tokenizer,
                max_prompt_tokens=config.max_prompt_tokens,
                max_response_tokens=config.max_response_tokens,
                max_soft_prefix_length=config.soft_prefix_length,
            )
            return cls(worker_group=worker_group, codec=codec, config=config)
        except Exception:
            if started_ray and ray.is_initialized():
                ray.shutdown()
            raise

    def generate(self, requests: tuple[GenerationRequest, ...]) -> tuple[GenerationResult, ...]:
        if not requests:
            return ()
        generation_started = time.perf_counter()
        if (
            any(request.soft_prefix is not None for request in requests)
            and not self.config.require_hybrid_prefix
        ):
            raise RuntimeError(
                "soft-prefix requests require require_hybrid_prefix=True and patched vLLM"
            )
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        data = self.codec.generation_dataproto(requests)
        padded, pad_size = pad_dataproto_to_divisor(data, self.worker_group.world_size)
        worker_started = time.perf_counter()
        if self._rollout_session_active:
            output = self.worker_group.generate_sequences_in_infoskill_session(padded)
        else:
            output = self.worker_group.generate_sequences(padded)
        worker_seconds = time.perf_counter() - worker_started
        output = unpad_dataproto(output, pad_size=pad_size)
        decoded = self.codec.decode_generation(requests, output)
        self._generation_calls += 1
        self._generation_requests += len(requests)
        self._generation_worker_seconds += worker_seconds
        self._generation_seconds += time.perf_counter() - generation_started
        return decoded

    def condition_infoskill(
        self,
        requests: tuple[ConditioningRequest, ...],
        candidate_skill_ids: tuple[str, ...],
        *,
        latent_mode: Literal["sample", "mean"],
    ) -> tuple[InfoSkillConditioningResult, ...]:
        """Condition states on the same workers that own trainable M1 modules."""

        if not self.config.enable_infoskill_modules:
            raise RuntimeError("INFO-SKILL runtime modules are not enabled")
        if not requests:
            return ()
        import numpy as np
        import torch
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        items = tuple(
            InfoSkillConditioningWorkItem(
                compression_view=request.views.compression_view,
                candidate_skill_ids=candidate_skill_ids,
                latent_seed=request.latent_seed,
                latent_mode=latent_mode,
            )
            for request in requests
        )
        data = DataProto.from_dict(
            tensors={"infoskill_row_id": torch.arange(len(items))},
            non_tensors={
                "infoskill_work_item": np.asarray(items, dtype=object),
            },
        )
        padded, pad_size = pad_dataproto_to_divisor(
            data,
            self.worker_group.world_size,
        )
        output = self.worker_group.condition_infoskill(padded)
        output = unpad_dataproto(output, pad_size=pad_size)
        row_ids = tuple(
            int(value)
            for value in output.batch["infoskill_row_id"].tolist()
        )
        if row_ids != tuple(range(len(items))):
            raise RuntimeError("INFO-SKILL worker conditioning changed row order")
        results = tuple(
            output.non_tensor_batch["infoskill_conditioning_result"].tolist()
        )
        if len(results) != len(items) or not all(
            isinstance(result, InfoSkillConditioningResult)
            for result in results
        ):
            raise RuntimeError("INFO-SKILL worker returned an invalid conditioning batch")
        return results

    def condition_infoskill_grouped(
        self,
        requests: tuple[ConditioningRequest, ...],
        candidate_skill_ids_by_request: tuple[tuple[str, ...], ...],
        *,
        latent_mode: Literal["sample", "mean"],
    ) -> tuple[InfoSkillConditioningResult, ...]:
        """Condition requests with per-episode candidates in one worker RPC."""

        if not self.config.enable_infoskill_modules:
            raise RuntimeError("INFO-SKILL runtime modules are not enabled")
        if not requests:
            if candidate_skill_ids_by_request:
                raise ValueError("empty requests cannot have candidate skills")
            return ()
        if len(requests) != len(candidate_skill_ids_by_request):
            raise ValueError("conditioning requests and candidate skills must align")
        import numpy as np
        import torch
        from verl import DataProto

        logical_items = tuple(
            InfoSkillConditioningWorkItem(
                compression_view=request.views.compression_view,
                candidate_skill_ids=candidate_skill_ids,
                latent_seed=request.latent_seed,
                latent_mode=latent_mode,
            )
            for request, candidate_skill_ids in zip(
                requests,
                candidate_skill_ids_by_request,
            )
        )
        items = _rank_replicated_conditioning_rows(
            logical_items,
            self.worker_group.world_size,
        )
        data = DataProto.from_dict(
            tensors={"infoskill_row_id": torch.arange(len(items))},
            non_tensors={
                "infoskill_work_item": np.asarray(items, dtype=object),
            },
        )
        output = self.worker_group.condition_infoskill_serial(data)
        row_ids = tuple(
            int(value)
            for value in output.batch["infoskill_row_id"].tolist()
        )
        if row_ids != tuple(range(len(items))):
            raise RuntimeError("INFO-SKILL worker conditioning changed row order")
        physical_results = tuple(
            output.non_tensor_batch["infoskill_conditioning_result"].tolist()
        )
        if len(physical_results) != len(items) or not all(
            isinstance(result, InfoSkillConditioningResult)
            for result in physical_results
        ):
            raise RuntimeError("INFO-SKILL worker returned an invalid conditioning batch")
        return _select_rank_zero_conditioning_rows(
            physical_results,
            logical_size=len(logical_items),
            world_size=self.worker_group.world_size,
        )

    @contextmanager
    def rollout_session(self):
        if not self.config.persistent_rollout_session:
            yield
            return
        if self._rollout_session_active:
            yield
            return
        self.worker_group.begin_infoskill_rollout_session()
        self._rollout_session_active = True
        try:
            yield
        finally:
            try:
                self.worker_group.end_infoskill_rollout_session()
            finally:
                self._rollout_session_active = False

    def rollout_memory_metrics(self) -> dict[str, float]:
        """Return the completed rollout session's per-rank physical peak."""

        if self._rollout_session_active:
            raise RuntimeError("rollout memory requires a completed rollout session")
        return summarize_cuda_memory_snapshots(
            self.worker_group.infoskill_rollout_memory_snapshot()
        )

    def update_policy(
        self,
        groups: tuple[TrajectoryGroup, ...],
        policy_advantages: tuple[tuple[float, ...], ...],
        *,
        global_update: int,
    ) -> Mapping[str, float]:
        policy_update_started = time.perf_counter()
        stage_started = policy_update_started
        data = self.codec.training_dataproto(groups, policy_advantages)
        real_sample_count = len(data)
        data, padding_count = pad_batch_to_divisor(
            data,
            self.worker_group.world_size,
        )
        token_counts = [
            int(value)
            for value in data.batch["attention_mask"].sum(dim=-1).tolist()
        ]
        token_load_metrics: dict[str, float] = {}
        real_positions: list[int] | None = None
        if self.config.balance_policy_tokens_across_ranks:
            import torch

            from verl.utils.seqlen_balancing import (
                get_seqlen_balanced_partitions,
            )

            token_load_metrics.update(
                summarize_rank_token_load(
                    token_counts,
                    self.worker_group.world_size,
                    prefix="perf/tokens_before_balance",
                )
            )
            order = policy_rank_balanced_order(
                token_counts,
                world_size=self.worker_group.world_size,
                global_minibatch_size=self.config.action_minibatch_size,
                partitioner=get_seqlen_balanced_partitions,
            )
            data.reorder(torch.tensor(order, dtype=torch.long))
            token_counts = [token_counts[index] for index in order]
            real_positions = [
                position
                for position, original_index in enumerate(order)
                if original_index < real_sample_count
            ]
        token_load_metrics.update(
            summarize_rank_token_load(
                token_counts,
                self.worker_group.world_size,
            )
        )
        training_codec_seconds = time.perf_counter() - stage_started
        if self.config.cuda_memory_poll_interval_ms > 0:
            self.worker_group.begin_infoskill_policy_memory_measurement()
        stage_started = time.perf_counter()
        old = self.worker_group.compute_log_prob(data)
        old_logprob_seconds = time.perf_counter() - stage_started
        alignment_metrics: dict[str, float] = {}
        if global_update == 0:
            real_data = (
                data[real_positions]
                if real_positions is not None
                else data[:real_sample_count]
            )
            real_old = (
                old[real_positions]
                if real_positions is not None
                else old[:real_sample_count]
            )
            response_width = int(real_data.batch["responses"].shape[-1])
            response_mask = real_data.batch["attention_mask"][:, -response_width:].bool()
            alignment = summarize_logprob_alignment(
                rollout=real_data.batch["rollout_log_probs"].tolist(),
                recomputed=real_old.batch["old_log_probs"].tolist(),
                mask=response_mask.tolist(),
                token_ids=real_data.batch["responses"].tolist(),
            )
            require_logprob_alignment(alignment)
            alignment_metrics = {
                f"rollout_recompute/{key}": float(value)
                for key, value in alignment.items()
            }
        data = data.union(old)
        if self.config.enable_infoskill_modules:
            reference_logprob_seconds = None
        else:
            stage_started = time.perf_counter()
            reference = self.worker_group.compute_ref_log_prob(data)
            reference_logprob_seconds = time.perf_counter() - stage_started
            data = data.union(reference)
        stage_started = time.perf_counter()
        result = self.worker_group.update_actor(data)
        actor_update_seconds = time.perf_counter() - stage_started
        cuda_memory_metrics = (
            summarize_cuda_memory_snapshots(
                self.worker_group.infoskill_cuda_memory_snapshot()
            )
            if (
                self.config.persistent_rollout_session
                or self.config.cuda_memory_poll_interval_ms > 0
            )
            else {}
        )
        self._completed_updates = global_update + 1
        metrics = _reduce_metrics(result.meta_info.get("metrics", {}))
        if reference_logprob_seconds is None:
            reference_logprob_seconds = metrics.get(
                "perf/reference_logprob_seconds", 0.0
            )
        metrics.update(
            {
                "runtime/training_sample_count": float(real_sample_count),
                "runtime/training_padding_count": float(padding_count),
                "runtime/training_padded_sample_count": float(len(data)),
                "runtime/effective_action_minibatch_size": float(
                    _effective_global_minibatch_size(
                        self.config.action_minibatch_size,
                        self.worker_group.world_size,
                    )
                ),
                "perf/rollout_generation_calls": float(self._generation_calls),
                "perf/rollout_generation_requests": float(self._generation_requests),
                "perf/rollout_generation_seconds": self._generation_seconds,
                "perf/rollout_generation_worker_seconds": self._generation_worker_seconds,
                "perf/training_codec_seconds": training_codec_seconds,
                "perf/old_logprob_seconds": old_logprob_seconds,
                "perf/reference_logprob_seconds": reference_logprob_seconds,
                "perf/actor_update_seconds": actor_update_seconds,
                "perf/runtime_policy_update_seconds": time.perf_counter()
                - policy_update_started,
            }
        )
        metrics.update(alignment_metrics)
        metrics.update(cuda_memory_metrics)
        metrics.update(token_load_metrics)
        self._generation_calls = 0
        self._generation_requests = 0
        self._generation_seconds = 0.0
        self._generation_worker_seconds = 0.0
        return metrics

    def update_auxiliary(
        self,
        groups: tuple[TrajectoryGroup, ...],
        fidelity_targets: tuple[tuple[float, ...], ...],
        *,
        global_update: int,
    ) -> Mapping[str, float]:
        if not self.config.enable_infoskill_auxiliary:
            raise RuntimeError("INFO-SKILL auxiliary runtime is not enabled")
        if self._grounding_dataset is None:
            raise RuntimeError("INFO-SKILL grounding dataset is not loaded")
        import numpy as np
        import torch
        from verl import DataProto

        from infoskill.training.auxiliary_work import (
            build_offline_auxiliary_examples,
            collect_online_auxiliary_examples,
            partition_auxiliary_work,
        )

        sample_seed = _named_seed(
            self.config.master_seed,
            "offline-grounding-sample",
            global_update,
        )
        samples = self._grounding_dataset.sample_games(
            game_count=self.config.infoskill_offline_games_per_update,
            seed=sample_seed,
        )
        epsilon_seeds = tuple(
            _named_seed(
                self.config.master_seed,
                "offline-grounding-epsilon",
                global_update,
                index,
                sample.state.task_id,
                sample.state.step_index,
            )
            for index, sample in enumerate(samples)
        )
        work = partition_auxiliary_work(
            online=collect_online_auxiliary_examples(groups, fidelity_targets),
            offline=build_offline_auxiliary_examples(samples, epsilon_seeds),
            world_size=self.worker_group.world_size,
            micro_batch_size=self.config.infoskill_auxiliary_micro_batch_size,
        )
        data = DataProto.from_dict(
            tensors={
                "infoskill_auxiliary_rank": torch.arange(len(work)),
            },
            non_tensors={
                "infoskill_auxiliary_work": np.asarray(work, dtype=object),
            },
        )
        result = self.worker_group.update_infoskill_auxiliary(data)
        ranks = tuple(
            int(value)
            for value in result.batch["infoskill_auxiliary_rank"].tolist()
        )
        if ranks != tuple(range(self.worker_group.world_size)):
            raise RuntimeError("INFO-SKILL auxiliary workers changed rank order")
        reports = tuple(
            result.non_tensor_batch["infoskill_auxiliary_metrics"].tolist()
        )
        return _require_equal_auxiliary_metrics(reports)

    def synchronize_rollout_weights(self) -> None:
        # VERL's hybrid FSDP/vLLM sharding manager synchronizes on the next generation context.
        return None

    def save_portable_state(self, directory: Path) -> Mapping[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        self.worker_group.save_portable_checkpoint(
            str(directory / "actor"), self._completed_updates
        )
        return {
            "format": (
                "infoskill-full-rank0-portable-v1"
                if self.config.enable_infoskill_modules
                else "infoskill-rank0-portable-v1"
            ),
            "portable": True,
            "base_weights_included": False,
            "infoskill_modules_included": self.config.enable_infoskill_modules,
        }

    def load_portable_state(
        self, directory: Path
    ) -> tuple[Mapping[str, object], ...]:
        return load_portable_state_after_base_sync(
            worker_group=self.worker_group,
            actor_directory=directory / "actor",
        )

    def compare_portable_actor_state(
        self, directory: Path
    ) -> tuple[Mapping[str, object], ...]:
        snapshots = self.worker_group.compare_infoskill_portable_actor(
            str(directory / "actor")
        )
        return tuple(snapshots)

    def vllm_lora_snapshot(self) -> tuple[Mapping[str, object], ...]:
        if not self._rollout_session_active:
            raise RuntimeError("vLLM LoRA snapshot requires an active rollout session")
        return tuple(self.worker_group.infoskill_vllm_lora_snapshot())

    def reset_rollout_prefix_cache(self) -> None:
        if not self._rollout_session_active:
            raise RuntimeError("prefix-cache reset requires an active rollout session")
        self.worker_group.reset_infoskill_rollout_prefix_cache()

    def close(self) -> None:
        if self._rollout_session_active:
            try:
                self.worker_group.end_infoskill_rollout_session()
            finally:
                self._rollout_session_active = False
        if ray.is_initialized():
            ray.shutdown()


def _actor_config(settings: VerlRuntimeConfig):
    from omegaconf import OmegaConf, open_dict

    source = Path(settings.skillrl_source) / "verl" / "trainer" / "config" / "ppo_trainer.yaml"
    config = OmegaConf.load(source)
    prefix_budget = settings.soft_prefix_length if settings.require_hybrid_prefix else 0
    config.data.max_prompt_length = settings.max_prompt_tokens + prefix_budget
    config.data.max_response_length = settings.max_response_tokens
    actor_ref = config.actor_rollout_ref
    actor_ref.model.path = settings.model_path
    actor_ref.model.lora_rank = settings.lora_rank
    actor_ref.model.lora_alpha = settings.lora_alpha
    actor_ref.model.target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    actor_ref.model.enable_gradient_checkpointing = True
    actor_ref.model.use_remove_padding = True
    actor_ref.model.trust_remote_code = True
    with open_dict(actor_ref.model):
        actor_ref.model.initialization_seed = settings.master_seed
        actor_ref.model.infoskill_cuda_memory_poll_interval_ms = (
            settings.cuda_memory_poll_interval_ms
        )
        actor_ref.model.infoskill_modules_enabled = (
            settings.enable_infoskill_modules
        )
        actor_ref.model.infoskill_semantic_model_path = settings.semantic_model_path
        actor_ref.model.infoskill_skill_bank_path = settings.skill_bank_path
        actor_ref.model.infoskill_latent_dim = settings.infoskill_latent_dim
        actor_ref.model.infoskill_prefix_length = settings.soft_prefix_length
        actor_ref.model.infoskill_initialization_seed = (
            settings.infoskill_initialization_seed
        )
        actor_ref.model.infoskill_total_policy_steps = settings.total_training_steps
        actor_ref.model.infoskill_projector_learning_rate = (
            settings.infoskill_projector_learning_rate
        )
        actor_ref.model.infoskill_projector_weight_decay = (
            settings.infoskill_projector_weight_decay
        )
        actor_ref.model.infoskill_policy_warmup_ratio = (
            settings.infoskill_policy_warmup_ratio
        )
        actor_ref.model.infoskill_history_length = settings.infoskill_history_length
        actor_ref.model.infoskill_auxiliary_learning_rate = (
            settings.infoskill_auxiliary_learning_rate
        )
        actor_ref.model.infoskill_auxiliary_weight_decay = (
            settings.infoskill_auxiliary_weight_decay
        )
        actor_ref.model.infoskill_auxiliary_warmup_ratio = (
            settings.infoskill_auxiliary_warmup_ratio
        )
        actor_ref.model.infoskill_fidelity_weight = (
            settings.infoskill_fidelity_weight
        )
        actor_ref.model.infoskill_rate_weight = settings.infoskill_rate_weight
        actor_ref.model.infoskill_grounding_weight = (
            settings.infoskill_grounding_weight
        )
        actor_ref.model.infoskill_auxiliary_max_grad_norm = (
            settings.infoskill_auxiliary_max_grad_norm
        )
    actor_ref.actor.strategy = "fsdp"
    actor_ref.actor.optim.lr = settings.actor_learning_rate
    actor_ref.actor.optim.weight_decay = 0.0
    actor_ref.actor.optim.betas = [0.9, 0.95]
    actor_ref.actor.optim.lr_warmup_steps_ratio = 0.03
    actor_ref.actor.optim.warmup_style = "constant"
    actor_ref.actor.optim.total_training_steps = settings.total_training_steps
    actor_ref.actor.ppo_mini_batch_size = settings.action_minibatch_size
    actor_ref.actor.ppo_micro_batch_size_per_gpu = (
        _policy_micro_batch_size_per_gpu(
            settings.action_minibatch_size,
            settings.num_gpus,
        )
    )
    actor_ref.actor.ppo_max_token_len_per_gpu = settings.policy_max_tokens_per_gpu
    actor_ref.actor.use_dynamic_bsz = True
    actor_ref.actor.ppo_epochs = 1
    actor_ref.actor.shuffle = True
    actor_ref.actor.loss_agg_mode = "seq-mean-token-mean"
    actor_ref.actor.entropy_coeff = 0.001
    actor_ref.actor.clip_ratio = 0.2
    actor_ref.actor.clip_ratio_low = 0.2
    actor_ref.actor.clip_ratio_high = 0.2
    actor_ref.actor.use_kl_loss = True
    actor_ref.actor.kl_loss_coef = 0.01
    actor_ref.actor.kl_loss_type = "low_var_kl"
    actor_ref.actor.grad_clip = 1.0
    actor_ref.actor.fsdp_config.param_offload = False
    actor_ref.actor.fsdp_config.optimizer_offload = False
    actor_ref.rollout.name = "vllm"
    actor_ref.rollout.mode = "sync"
    actor_ref.rollout.n = 1
    actor_ref.rollout.temperature = 1.0
    actor_ref.rollout.top_p = 1.0
    actor_ref.rollout.top_k = -1
    actor_ref.rollout.prompt_length = settings.max_prompt_tokens + prefix_budget
    actor_ref.rollout.response_length = settings.max_response_tokens
    actor_ref.rollout.max_model_len = settings.max_prompt_tokens + settings.max_response_tokens + 5
    actor_ref.rollout.tensor_model_parallel_size = 1
    actor_ref.rollout.gpu_memory_utilization = settings.gpu_memory_utilization
    actor_ref.rollout.max_num_batched_tokens = settings.rollout_max_batched_tokens
    actor_ref.rollout.max_num_seqs = 512
    actor_ref.rollout.log_prob_micro_batch_size_per_gpu = 4
    actor_ref.rollout.log_prob_use_dynamic_bsz = True
    actor_ref.rollout.log_prob_max_token_len_per_gpu = (
        settings.policy_max_tokens_per_gpu
    )
    actor_ref.rollout.enforce_eager = settings.require_hybrid_prefix
    actor_ref.rollout.free_cache_engine = False
    actor_ref.rollout.enable_chunked_prefill = True
    actor_ref.rollout.seed = settings.master_seed
    with open_dict(actor_ref.rollout):
        actor_ref.rollout.infoskill_hybrid_prefix = settings.require_hybrid_prefix
        for name, value in vllm_action_stop_settings("</action>").items():
            setattr(actor_ref.rollout, name, value)
    actor_ref.ref.log_prob_micro_batch_size_per_gpu = 4
    actor_ref.ref.log_prob_use_dynamic_bsz = True
    actor_ref.ref.log_prob_max_token_len_per_gpu = settings.policy_max_tokens_per_gpu
    actor_ref.ref.fsdp_config.param_offload = False
    OmegaConf.resolve(config)
    return config


def _policy_micro_batch_size_per_gpu(
    global_minibatch_size: int,
    world_size: int,
    *,
    preferred: int = 4,
) -> int:
    """Choose a worker-valid divisor without changing the global minibatch."""

    if min(global_minibatch_size, world_size, preferred) <= 0:
        raise ValueError("policy batch dimensions must be positive")
    normalized = global_minibatch_size // world_size
    if normalized <= 0:
        raise ValueError("global policy minibatch must cover every GPU")
    return max(
        candidate
        for candidate in range(1, min(preferred, normalized) + 1)
        if normalized % candidate == 0
    )


def _effective_global_minibatch_size(
    configured_size: int,
    world_size: int,
) -> int:
    """Expose VERL's per-rank floor normalization as an auditable value."""

    if min(configured_size, world_size) <= 0:
        raise ValueError("policy batch dimensions must be positive")
    effective = (configured_size // world_size) * world_size
    if effective <= 0:
        raise ValueError("global policy minibatch must cover every GPU")
    return effective


def _require_hybrid_prefix_runtime() -> None:
    try:
        from vllm import envs as vllm_envs
        from vllm.inputs.data import INFOSKILL_HYBRID_PREFIX_API
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "patched vLLM 0.8.4+infoskill1 is required for Hybrid Prefix Input"
        ) from error
    if INFOSKILL_HYBRID_PREFIX_API != 1:
        raise RuntimeError(
            f"unsupported INFO-SKILL Hybrid Prefix API: {INFOSKILL_HYBRID_PREFIX_API}"
        )
    if not vllm_envs.VLLM_USE_V1:
        raise RuntimeError("INFO-SKILL Hybrid Prefix Input requires VLLM_USE_V1=1")


def _reduce_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (list, tuple)):
            values = [float(item) for item in value]
            if values:
                reduced[key] = statistics.fmean(values)
        else:
            try:
                reduced[key] = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return reduced


def _named_seed(master_seed: int, namespace: str, *parts: object) -> int:
    payload = "\0".join(str(item) for item in (master_seed, namespace, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _require_equal_auxiliary_metrics(
    reports: tuple[object, ...],
) -> dict[str, float]:
    if not reports or not all(isinstance(report, Mapping) for report in reports):
        raise RuntimeError("INFO-SKILL auxiliary workers returned invalid metrics")
    normalized = tuple(_reduce_metrics(report) for report in reports)
    keys = set(normalized[0])
    if any(set(report) != keys for report in normalized[1:]):
        raise RuntimeError("INFO-SKILL auxiliary metric keys differ across ranks")
    for key in keys:
        values = [report[key] for report in normalized]
        if not all(math.isclose(value, values[0], rel_tol=1e-6, abs_tol=1e-6) for value in values[1:]):
            raise RuntimeError(
                f"INFO-SKILL auxiliary metric {key} differs across ranks: {values}"
            )
    return normalized[0]
