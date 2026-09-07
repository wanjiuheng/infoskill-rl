from __future__ import annotations

import logging
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import ray

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
    policy_max_tokens_per_gpu: int = 16_384
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


class VerlRuntime:
    """Own only worker initialization, generation, logprobs, and LoRA updates."""

    def __init__(self, *, worker_group: object, codec: VerlBatchCodec, config: VerlRuntimeConfig) -> None:
        self.worker_group = worker_group
        self.codec = codec
        self.config = config
        self._completed_updates = 0
        self._generation_calls = 0
        self._generation_requests = 0
        self._generation_seconds = 0.0
        self._generation_worker_seconds = 0.0
        self._rollout_session_active = False

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
        metrics.update(
            {
                "runtime/training_sample_count": float(real_sample_count),
                "runtime/training_padding_count": float(padding_count),
                "runtime/training_padded_sample_count": float(len(data)),
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
        del groups, fidelity_targets, global_update
        raise RuntimeError("token-only VERL runtime has no INFO-SKILL auxiliary worker")

    def synchronize_rollout_weights(self) -> None:
        # VERL's hybrid FSDP/vLLM sharding manager synchronizes on the next generation context.
        return None

    def save_portable_state(self, directory: Path) -> Mapping[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        self.worker_group.save_portable_checkpoint(
            str(directory / "actor"), self._completed_updates
        )
        return {
            "format": "infoskill-rank0-portable-v1",
            "portable": True,
            "base_weights_included": False,
        }

    def load_portable_state(self, directory: Path) -> None:
        load_portable_state_after_base_sync(
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
    actor_ref.actor.strategy = "fsdp"
    actor_ref.actor.optim.lr = settings.actor_learning_rate
    actor_ref.actor.optim.weight_decay = 0.0
    actor_ref.actor.optim.betas = [0.9, 0.95]
    actor_ref.actor.optim.lr_warmup_steps_ratio = 0.03
    actor_ref.actor.optim.warmup_style = "constant"
    actor_ref.actor.optim.total_training_steps = settings.total_training_steps
    actor_ref.actor.ppo_mini_batch_size = settings.action_minibatch_size
    actor_ref.actor.ppo_micro_batch_size_per_gpu = 4
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
