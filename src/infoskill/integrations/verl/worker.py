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


class PortableActorRolloutRefWorker(ActorRolloutRefWorker):
    """Pinned FSDP1 worker with authoritative rank-0 LoRA optimizer export."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._infoskill_rollout_session_active = False
        self._infoskill_rollout_session_generation_count = 0
        seed = int(self.config.model.get("initialization_seed", 0))
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return super().init_model()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_infoskill_rollout_session(self) -> None:
        if self._infoskill_rollout_session_active:
            raise RuntimeError("INFO-SKILL rollout session is already active")
        self._infoskill_rollout_session_active = True
        self._infoskill_rollout_session_generation_count = 0
        try:
            self.rollout_sharding_manager.__enter__()
        except Exception:
            try:
                self.rollout_sharding_manager.__exit__(None, None, None)
            finally:
                self._infoskill_rollout_session_active = False
            raise

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
            self.rollout_sharding_manager.__exit__(None, None, None)
        finally:
            self._infoskill_rollout_session_active = False
            self._infoskill_rollout_session_generation_count = 0

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
