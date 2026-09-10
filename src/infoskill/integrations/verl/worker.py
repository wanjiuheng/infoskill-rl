from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from peft import PeftModel
from safetensors.torch import load_file, save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import layered_summon_lora_params
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from infoskill.fsdp_checkpoint import load_peft_adapter_under_full_fsdp_state
from infoskill.checkpoint_effect import (
    compare_named_tensors,
    flatten_lora_model_tensors,
    summarize_named_tensors,
)
from infoskill.integrations.verl.memory_metrics import PhysicalMemorySampler


class PortableActorRolloutRefWorker(ActorRolloutRefWorker):
    """Pinned FSDP1 worker with authoritative rank-0 LoRA optimizer export."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._infoskill_rollout_session_active = False
        self._infoskill_rollout_session_generation_count = 0
        self._infoskill_rollout_memory_snapshot = None
        self._infoskill_cuda_memory_sampler = None
        self._infoskill_cuda_memory_poll_interval_ms = int(
            self.config.model.get("infoskill_cuda_memory_poll_interval_ms", 0)
        )
        if self._infoskill_cuda_memory_poll_interval_ms < 0:
            raise ValueError("CUDA memory polling interval must be non-negative")
        seed = int(self.config.model.get("initialization_seed", 0))
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result = super().init_model()
        self._infoskill_worker_conditioner = None
        if bool(self.config.model.get("infoskill_modules_enabled", False)):
            self._initialize_infoskill_conditioning_modules()
        return result

    def _initialize_infoskill_conditioning_modules(self) -> None:
        from infoskill.integrations.verl.worker_conditioning import (
            InfoSkillWorkerConditioner,
        )
        from infoskill.models import InfoSkillCompressor, LatentProjector
        from infoskill.semantic import FrozenSemanticEncoder, SemanticFeatureCache
        from infoskill.skills import FixedSkillLibrary

        semantic_path = self.config.model.get("infoskill_semantic_model_path")
        skill_bank_path = self.config.model.get("infoskill_skill_bank_path")
        if not isinstance(semantic_path, str) or not semantic_path.strip():
            raise ValueError("INFO-SKILL worker requires a semantic model path")
        if not isinstance(skill_bank_path, str) or not skill_bank_path.strip():
            raise ValueError("INFO-SKILL worker requires a skill bank path")
        latent_dim = int(self.config.model.get("infoskill_latent_dim", 32))
        prefix_length = int(self.config.model.get("infoskill_prefix_length", 5))
        initialization_seed = int(
            self.config.model.get("infoskill_initialization_seed", 0)
        )
        if min(latent_dim, prefix_length) <= 0 or initialization_seed < 0:
            raise ValueError("INFO-SKILL worker dimensions and seed are invalid")

        library = FixedSkillLibrary.load(skill_bank_path)
        records = library.general + library.task_specific + library.mistakes
        device_index = get_torch_device().current_device()
        device = torch.device("cuda", device_index)
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(initialization_seed)
            torch.cuda.manual_seed(initialization_seed)
            semantic = FrozenSemanticEncoder.from_pretrained(
                semantic_path,
                device=device,
            )
            feature_cache = SemanticFeatureCache(semantic)
            feature_cache.warm_skills(records)
            compressor = InfoSkillCompressor(
                semantic.hidden_size,
                latent_dim=latent_dim,
            ).to(device)
            projector = LatentProjector(
                latent_dim=latent_dim,
                policy_hidden_size=int(self.actor_model_config.hidden_size),
                prefix_length=prefix_length,
            ).to(device)
        self._infoskill_compressor = compressor
        self._infoskill_projector = projector
        self._infoskill_semantic_encoder = semantic
        self._infoskill_feature_cache = feature_cache
        self._infoskill_worker_conditioner = InfoSkillWorkerConditioner(
            library=library,
            semantic_encoder=semantic,
            feature_cache=feature_cache,
            compressor=compressor,
            projector=projector,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def condition_infoskill(self, data):
        conditioner = self._infoskill_worker_conditioner
        if conditioner is None:
            raise RuntimeError("INFO-SKILL worker conditioning is not initialized")
        from verl import DataProto

        items = tuple(data.non_tensor_batch["infoskill_work_item"].tolist())
        results = conditioner.condition(items)
        return DataProto.from_dict(
            tensors={"infoskill_row_id": data.batch["infoskill_row_id"]},
            non_tensors={
                "infoskill_conditioning_result": np.asarray(
                    results,
                    dtype=object,
                )
            },
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_infoskill_rollout_session(self) -> None:
        if self._infoskill_rollout_session_active:
            raise RuntimeError("INFO-SKILL rollout session is already active")
        self._infoskill_rollout_session_active = True
        self._infoskill_rollout_session_generation_count = 0
        self._infoskill_rollout_memory_snapshot = None
        get_torch_device().reset_peak_memory_stats()
        self._start_infoskill_cuda_memory_sampler()
        try:
            self.rollout_sharding_manager.__enter__()
        except Exception:
            try:
                self._stop_infoskill_cuda_memory_sampler()
                self.rollout_sharding_manager.__exit__(None, None, None)
            finally:
                self._infoskill_rollout_session_active = False
            raise

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_infoskill_portable_checkpoint_load(self) -> dict[str, object]:
        """Finish dummy vLLM base sync before a restored LoRA can be consumed."""

        if self._infoskill_rollout_session_active:
            raise RuntimeError("portable load preparation requires no active rollout session")
        sharding = self.rollout_sharding_manager
        ready_before = bool(sharding.base_sync_done)
        if not ready_before:
            entered = False
            try:
                sharding.__enter__()
                entered = True
            finally:
                if entered:
                    sharding.__exit__(None, None, None)
        ready_after = bool(sharding.base_sync_done)
        if not ready_after:
            raise RuntimeError("vLLM base sync remained incomplete before portable load")
        return {
            "rank": dist.get_rank(),
            "base_sync_done_before": ready_before,
            "base_sync_done_after": ready_after,
            "warmup_performed": not ready_before,
        }

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences_in_infoskill_session(self, prompts):
        if not self._infoskill_rollout_session_active:
            raise RuntimeError("INFO-SKILL rollout session is not active")
        prompts = prompts.to(get_torch_device().current_device())
        if not self._is_rollout:
            raise RuntimeError("worker has no rollout engine")
        if self._infoskill_rollout_session_generation_count > 0:
            # Baseline sleep(level=1) clears prefix cache after every call.
            # Preserve that semantic boundary while avoiding repeated weight
            # synchronization and vLLM allocator teardown.
            reset = self.rollout_sharding_manager.inference_engine.reset_prefix_cache()
            if reset is False:
                raise RuntimeError("vLLM refused to reset prefix cache between env steps")
        prompts.meta_info.update(
            {
                "eos_token_id": (
                    self.generation_config.eos_token_id
                    if self.generation_config is not None
                    else self.tokenizer.eos_token_id
                ),
                "pad_token_id": (
                    self.generation_config.pad_token_id
                    if self.generation_config is not None
                    else self.tokenizer.pad_token_id
                ),
            }
        )
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)
        output = output.to("cpu")
        self._infoskill_rollout_session_generation_count += 1
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def end_infoskill_rollout_session(self) -> None:
        if not self._infoskill_rollout_session_active:
            return
        try:
            try:
                self._infoskill_rollout_memory_snapshot = _cuda_memory_snapshot()
                self._infoskill_rollout_memory_snapshot.update(
                    self._stop_infoskill_cuda_memory_sampler()
                )
            finally:
                self.rollout_sharding_manager.__exit__(None, None, None)
        finally:
            # Start a fresh peak window for old/ref logprob and actor update.
            get_torch_device().reset_peak_memory_stats()
            self._infoskill_rollout_session_active = False
            self._infoskill_rollout_session_generation_count = 0

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_infoskill_policy_memory_measurement(self) -> None:
        if self._infoskill_cuda_memory_sampler is not None:
            raise RuntimeError("CUDA memory sampler is already active")
        get_torch_device().reset_peak_memory_stats()
        self._start_infoskill_cuda_memory_sampler()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def infoskill_cuda_memory_snapshot(self) -> dict[str, object]:
        policy = _cuda_memory_snapshot()
        policy.update(self._stop_infoskill_cuda_memory_sampler())
        return {
            "rank": dist.get_rank(),
            "rollout": self._infoskill_rollout_memory_snapshot,
            "policy": policy,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def compare_infoskill_portable_actor(self, directory: str) -> dict[str, object]:
        """Compare every live FSDP LoRA tensor with a portable checkpoint."""

        if self._infoskill_rollout_session_active:
            raise RuntimeError("actor comparison requires the rollout session to be closed")
        source = Path(directory)
        expected = load_file(
            str(source / "adapter_model.safetensors"),
            device="cpu",
        )
        actual = layered_summon_lora_params(self.actor_module_fsdp)
        report = compare_named_tensors(expected, actual)
        report["rank"] = dist.get_rank()
        return report

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def infoskill_vllm_lora_snapshot(self) -> dict[str, object]:
        """Inspect the active pinned-vLLM adapter after FSDP-to-vLLM sync."""

        if not self._infoskill_rollout_session_active:
            raise RuntimeError("vLLM LoRA inspection requires an active rollout session")
        sharding = self.rollout_sharding_manager
        manager = getattr(sharding.model_runner, "lora_manager", None)
        if manager is None:
            raise RuntimeError("pinned vLLM model runner has no LoRA manager")
        adapter_manager = getattr(manager, "_adapter_manager", None)
        if adapter_manager is None:
            raise RuntimeError("pinned vLLM LoRA manager has no adapter registry")
        adapters = adapter_manager.list_adapters()
        active_ids = sorted(
            int(adapter_id)
            for adapter_id in sharding.inference_engine.llm_engine.list_loras()
        )
        return {
            "rank": dist.get_rank(),
            "active_adapter_ids": active_ids,
            "registered_adapter_ids": sorted(int(adapter_id) for adapter_id in adapters),
            "adapters": {
                str(adapter_id): {
                    "rank": int(adapter.rank),
                    "summary": summarize_named_tensors(
                        flatten_lora_model_tensors(adapter)
                    ),
                }
                for adapter_id, adapter in adapters.items()
            },
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_infoskill_rollout_prefix_cache(self) -> None:
        if not self._infoskill_rollout_session_active:
            raise RuntimeError("prefix-cache reset requires an active rollout session")
        reset = self.rollout_sharding_manager.inference_engine.reset_prefix_cache()
        if reset is False:
            raise RuntimeError("vLLM refused the checkpoint-effect prefix-cache reset")

    def _start_infoskill_cuda_memory_sampler(self) -> None:
        interval = self._infoskill_cuda_memory_poll_interval_ms
        if interval <= 0:
            return
        if self._infoskill_cuda_memory_sampler is not None:
            raise RuntimeError("CUDA memory sampler is already active")
        device = get_torch_device()
        device_index = device.current_device()
        sampler = PhysicalMemorySampler(
            read_memory=lambda: device.mem_get_info(device_index),
            interval_ms=interval,
        )
        sampler.start()
        self._infoskill_cuda_memory_sampler = sampler

    def _stop_infoskill_cuda_memory_sampler(self) -> dict[str, int]:
        sampler = self._infoskill_cuda_memory_sampler
        if sampler is None:
            return {}
        self._infoskill_cuda_memory_sampler = None
        return sampler.stop()

    def _build_rollout(self, trust_remote_code: bool = False):
        from verl.workers.rollout import vllm_rollout as rollout_package

        from .hybrid_rollout import hybrid_vllm_rollout_class

        original = rollout_package.vLLMRollout
        rollout_package.vLLMRollout = hybrid_vllm_rollout_class()
        try:
            return super()._build_rollout(trust_remote_code=trust_remote_code)
        finally:
            rollout_package.vLLMRollout = original

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_portable_checkpoint(self, directory: str, global_step: int) -> None:
        if not self._is_actor or not self._is_lora or not isinstance(self.actor_module, PeftModel):
            raise RuntimeError("portable checkpoint requires a LoRA actor")
        destination = Path(directory)
        if dist.get_rank() == 0:
            destination.mkdir(parents=True, exist_ok=False)
        dist.barrier()

        lora_parameters = layered_summon_lora_params(self.actor_module_fsdp)
        full_optimizer = FSDP.full_optim_state_dict(
            self.actor_module_fsdp,
            self.actor_optimizer,
            rank0_only=True,
        )
        if dist.get_rank() == 0:
            save_file(lora_parameters, str(destination / "adapter_model.safetensors"))
            peft_config = asdict(self.actor_module.peft_config["default"])
            for key in ("task_type", "peft_type"):
                if hasattr(peft_config.get(key), "value"):
                    peft_config[key] = peft_config[key].value
            if isinstance(peft_config.get("target_modules"), set):
                peft_config["target_modules"] = sorted(peft_config["target_modules"])
            _write_json(destination / "adapter_config.json", peft_config)
            torch.save(full_optimizer, destination / "lora_optimizer_full.pt")
            torch.save(self.actor_lr_scheduler.state_dict(), destination / "lora_scheduler.pt")
            _write_json(
                destination / "actor_manifest.json",
                {
                    "schema_version": 1,
                    "global_step": global_step,
                    "fsdp_version": 1,
                    "optimizer_state": "full_named_rank0_reshardable",
                    "base_weights_included": False,
                },
            )
        dist.barrier()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_portable_checkpoint(self, directory: str) -> None:
        if not self._is_actor or not self._is_lora or not isinstance(self.actor_module, PeftModel):
            raise RuntimeError("portable checkpoint requires a LoRA actor")
        source = Path(directory)
        if not (source / "actor_manifest.json").is_file():
            raise RuntimeError(f"portable actor checkpoint is incomplete: {source}")

        adapter_state = load_file(str(source / "adapter_model.safetensors"), device="cpu")
        result = load_peft_adapter_under_full_fsdp_state(
            fsdp_model=self.actor_module_fsdp,
            peft_model=self.actor_module,
            adapter_state=adapter_state,
        )
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected LoRA keys: {result.unexpected_keys}")
        full_optimizer = (
            torch.load(source / "lora_optimizer_full.pt", map_location="cpu", weights_only=False)
            if dist.get_rank() == 0
            else None
        )
        sharded_optimizer = FSDP.scatter_full_optim_state_dict(
            full_optimizer,
            self.actor_module_fsdp,
            optim=self.actor_optimizer,
        )
        self.actor_optimizer.load_state_dict(sharded_optimizer)
        scheduler_state = torch.load(
            source / "lora_scheduler.pt", map_location="cpu", weights_only=False
        )
        self.actor_lr_scheduler.load_state_dict(scheduler_state)
        dist.barrier()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cuda_memory_snapshot() -> dict[str, int]:
    device = get_torch_device()
    device_index = device.current_device()
    free_bytes, total_bytes = device.mem_get_info(device_index)
    return {
        "allocated_bytes": int(device.memory_allocated(device_index)),
        "reserved_bytes": int(device.memory_reserved(device_index)),
        "peak_allocated_bytes": int(device.max_memory_allocated(device_index)),
        "peak_reserved_bytes": int(device.max_memory_reserved(device_index)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
