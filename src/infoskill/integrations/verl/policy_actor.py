from __future__ import annotations

import itertools
import time
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from infoskill.learning import PolicyUpdateCoordinator


class PrefixEmbeddingInjector:
    """Replace placeholder token embeddings inside the FSDP-owned forward.

    Looking up token embeddings outside the FSDP root is unsafe because its
    parameters may be sharded. A forward hook keeps lookup and replacement
    inside the normal actor forward while retaining the projector graph.
    """

    def __init__(self, token_embedding: nn.Module) -> None:
        self._active: tuple[Tensor, Tensor] | None = None
        self._handle = token_embedding.register_forward_hook(self._replace)

    @contextmanager
    def inject(self, *, prefix: Tensor, prefix_mask: Tensor):
        if self._active is not None:
            raise RuntimeError("INFO-SKILL prefix injection is not reentrant")
        self._active = (prefix, prefix_mask.bool())
        try:
            yield
        finally:
            self._active = None

    def close(self) -> None:
        self._handle.remove()

    def _replace(self, _module: nn.Module, _inputs: object, output: Tensor) -> Tensor:
        if self._active is None:
            return output
        prefix, mask = self._active
        if output.ndim != 3 or mask.shape != output.shape[:2]:
            raise RuntimeError(
                "INFO-SKILL embedding hook observed an unexpected sequence layout"
            )
        expanded = mask.unsqueeze(-1).expand_as(output)
        expected = int(expanded.sum().item())
        if prefix.numel() != expected:
            raise RuntimeError(
                "INFO-SKILL projected prefix does not match placeholder slots"
            )
        return output.masked_scatter(
            expanded,
            prefix.to(device=output.device, dtype=output.dtype).reshape(-1),
        )


def build_infoskill_policy_actor_class():
    """Build the project-owned adapter only inside the pinned VERL runtime."""

    from verl.trainer.ppo.core_algos import (
        agg_loss,
        compute_policy_loss,
        compute_policy_loss_gspo,
        kl_penalty,
    )
    from verl.utils.device import get_torch_device
    from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
    from verl.utils.py_functional import append_to_dict
    from verl.utils.seqlen_balancing import (
        get_reverse_idx,
        rearrange_micro_batches,
    )
    from verl.workers.actor import DataParallelPPOActor

    class InfoSkillDataParallelPPOActor(DataParallelPPOActor):
        """VERL actor that preserves exact M1 replay through policy update."""

        def __init__(
            self,
            *,
            config: object,
            actor_module: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            adapter_module: nn.Module,
            token_embedding: nn.Module,
            projector: nn.Module,
            projector_optimizer: torch.optim.Optimizer,
            actor_scheduler: object,
            projector_scheduler: object,
        ) -> None:
            super().__init__(
                config=config,
                actor_module=actor_module,
                actor_optimizer=actor_optimizer,
            )
            self.infoskill_projector = projector
            self.infoskill_adapter_module = adapter_module
            self._infoskill_injector = PrefixEmbeddingInjector(token_embedding)
            actor_parameters = tuple(
                parameter
                for group in actor_optimizer.param_groups
                for parameter in group["params"]
            )
            self._infoskill_policy_update = PolicyUpdateCoordinator(
                actor_parameters=actor_parameters,
                projector_parameters=projector.parameters(),
                actor_optimizer=actor_optimizer,
                projector_optimizer=projector_optimizer,
                max_grad_norm=float(self.config.grad_clip),
            )
            self.infoskill_actor_scheduler = actor_scheduler
            self.infoskill_projector_scheduler = projector_scheduler
            self.infoskill_update_applied = False

        def _forward_micro_batch(
            self,
            micro_batch,
            temperature,
            calculate_entropy=False,
            *,
            detach_projector_output: bool = False,
        ):
            if "infoskill_prefix_mask" not in micro_batch:
                raise RuntimeError(
                    "INFO-SKILL actor received a token-only micro-batch"
                )
            if self.use_ulysses_sp:
                raise RuntimeError(
                    "INFO-SKILL policy replay does not support Ulysses sequence parallelism"
                )
            prefix, hook_mask = _current_prefix_and_hook_mask(
                projector=self.infoskill_projector,
                replay_latents=micro_batch["infoskill_replay_latents"],
                prefix_mask=micro_batch["infoskill_prefix_mask"],
                attention_mask=micro_batch["attention_mask"],
                remove_padding=self.use_remove_padding,
                detach_projector_output=detach_projector_output,
            )
            with self._infoskill_injector.inject(
                prefix=prefix,
                prefix_mask=hook_mask,
            ):
                return super()._forward_micro_batch(
                    micro_batch,
                    temperature,
                    calculate_entropy,
                )

        def compute_log_prob(self, data: Any, calculate_entropy=False):
            if "infoskill_prefix_mask" not in data.batch:
                raise RuntimeError("INFO-SKILL actor received token-only replay")
            self.actor_module.eval()
            self.infoskill_projector.eval()
            micro_batch_size = data.meta_info["micro_batch_size"]
            temperature = data.meta_info["temperature"]
            use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
            keys = _infoskill_forward_keys()
            batch = data.select(batch_keys=keys).batch
            if use_dynamic_bsz:
                maximum = (
                    data.meta_info["max_token_len"]
                    * self.ulysses_sequence_parallel_size
                )
                micro_batches, indices = rearrange_micro_batches(
                    batch=batch,
                    max_token_len=maximum,
                )
            else:
                micro_batches = batch.split(micro_batch_size)
                indices = None

            log_probs = []
            entropies = []
            for micro_batch in micro_batches:
                with torch.no_grad():
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                log_probs.append(log_prob)
                if calculate_entropy:
                    entropies.append(entropy)
            result = torch.concat(log_probs, dim=0)
            entropy_result = (
                torch.concat(entropies, dim=0) if calculate_entropy else None
            )
            if indices is not None:
                flattened = list(itertools.chain.from_iterable(indices))
                reverse = torch.tensor(
                    get_reverse_idx(flattened),
                    dtype=torch.long,
                    device=result.device,
                )
                result = result[reverse]
                if entropy_result is not None:
                    entropy_result = entropy_result[reverse]
            return result, entropy_result

        def update_policy(self, data: Any):
            if "infoskill_prefix_mask" not in data.batch:
                raise RuntimeError("INFO-SKILL actor received token-only replay")
            if "multi_modal_inputs" in data.non_tensor_batch:
                raise RuntimeError("INFO-SKILL ALFWorld policy update is text-only")
            self.actor_module.train()
            self.infoskill_projector.train()
            self.infoskill_update_applied = False
            temperature = data.meta_info["temperature"]
            multi_turn = data.meta_info.get("multi_turn", False)
            keys = _infoskill_forward_keys() + ["old_log_probs", "advantages"]
            if multi_turn:
                keys.append("loss_mask")
            batch = data.select(batch_keys=keys).batch
            dataloader = batch.split(self.config.ppo_mini_batch_size)
            metrics: dict[str, object] = {}
            reference_logprob_seconds = 0.0
            fuse_kl_ppo_forward = bool(
                self.config.get("infoskill_fuse_kl_ppo_forward", False)
            )

            for _epoch in range(self.config.ppo_epochs):
                for mini_batch in dataloader:
                    if self.config.use_dynamic_bsz:
                        maximum = (
                            self.config.ppo_max_token_len_per_gpu
                            * self.ulysses_sequence_parallel_size
                        )
                        micro_batches, _ = rearrange_micro_batches(
                            batch=mini_batch,
                            max_token_len=maximum,
                        )
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size
                            // self.config.ppo_micro_batch_size_per_gpu
                        )
                        micro_batches = mini_batch.split(
                            self.config.ppo_micro_batch_size_per_gpu
                        )
                    self._infoskill_policy_update.zero_grad()

                    for micro_batch in micro_batches:
                        micro_batch = micro_batch.to(
                            get_torch_device().current_device()
                        )
                        responses = micro_batch["responses"]
                        response_length = responses.size(1)
                        if multi_turn:
                            response_mask = micro_batch["loss_mask"][
                                :, -response_length:
                            ]
                        else:
                            response_mask = micro_batch["attention_mask"][
                                :, -response_length:
                            ]
                        if self.config.use_dynamic_bsz:
                            loss_scale = (
                                len(micro_batch)
                                / self.config.ppo_mini_batch_size
                            )
                        else:
                            loss_scale = 1.0 / self.gradient_accumulation

                        # Reference and actor KL must use the same *current*
                        # projected prefix.  The prefix is detached on both
                        # sides so KL regularizes LoRA only (D009).  Backward
                        # immediately after this actor forward keeps the FSDP
                        # forward/backward lifecycle well formed.
                        ref_log_prob = None
                        if self.config.use_kl_loss:
                            reference_started = time.perf_counter()
                            with (
                                torch.no_grad(),
                                self.infoskill_adapter_module.disable_adapter(),
                            ):
                                _, ref_log_prob = self._forward_micro_batch(
                                    micro_batch,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    detach_projector_output=True,
                                )
                            reference_logprob_seconds += (
                                time.perf_counter() - reference_started
                            )
                            if not fuse_kl_ppo_forward:
                                _, kl_log_prob = self._forward_micro_batch(
                                    micro_batch,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    detach_projector_output=True,
                                )
                                kl_loss = _infoskill_kl_loss(
                                    actor_log_prob=kl_log_prob,
                                    reference_log_prob=ref_log_prob,
                                    response_mask=response_mask,
                                    kl_penalty_fn=kl_penalty,
                                    aggregate_loss_fn=agg_loss,
                                    kl_loss_type=self.config.kl_loss_type,
                                    loss_agg_mode=self.config.loss_agg_mode,
                                )
                                (
                                    kl_loss
                                    * self.config.kl_loss_coef
                                    * loss_scale
                                ).backward()
                                _append_kl_metrics(
                                    append_to_dict,
                                    metrics,
                                    kl_loss,
                                    self.config.kl_loss_coef,
                                )

                        calculate_entropy = self.config.entropy_coeff != 0
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )
                        loss_mode = self.config.policy_loss.get(
                            "loss_mode", "vanilla"
                        )
                        if loss_mode == "vanilla":
                            policy_loss_fn = compute_policy_loss
                        elif loss_mode == "gspo":
                            policy_loss_fn = compute_policy_loss_gspo
                        else:
                            raise ValueError(
                                f"Unsupported loss_mode: {loss_mode}"
                            )
                        clip_ratio = self.config.clip_ratio
                        clip_low = (
                            self.config.clip_ratio_low
                            if self.config.clip_ratio_low is not None
                            else clip_ratio
                        )
                        clip_high = (
                            self.config.clip_ratio_high
                            if self.config.clip_ratio_high is not None
                            else clip_ratio
                        )
                        pg_loss, clipfrac, ppo_kl, clipfrac_lower = policy_loss_fn(
                            old_log_prob=micro_batch["old_log_probs"],
                            log_prob=log_prob,
                            advantages=micro_batch["advantages"],
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_low,
                            cliprange_high=clip_high,
                            clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        policy_loss = pg_loss
                        if calculate_entropy:
                            entropy_loss = agg_loss(
                                loss_mat=entropy,
                                loss_mask=response_mask,
                                loss_agg_mode=self.config.loss_agg_mode,
                            )
                            policy_loss = (
                                policy_loss
                                - entropy_loss * self.config.entropy_coeff
                            )
                        if fuse_kl_ppo_forward and ref_log_prob is not None:
                            # This candidate deliberately lets KL regularize both
                            # LoRA and projector so the numerical actor logprob can
                            # be shared with PPO.  It is an algorithm candidate,
                            # not an equivalent implementation of D009.
                            kl_loss = _infoskill_kl_loss(
                                actor_log_prob=log_prob,
                                reference_log_prob=ref_log_prob,
                                response_mask=response_mask,
                                kl_penalty_fn=kl_penalty,
                                aggregate_loss_fn=agg_loss,
                                kl_loss_type=self.config.kl_loss_type,
                                loss_agg_mode=self.config.loss_agg_mode,
                            )
                            policy_loss = (
                                policy_loss
                                + kl_loss * self.config.kl_loss_coef
                            )
                            _append_kl_metrics(
                                append_to_dict,
                                metrics,
                                kl_loss,
                                self.config.kl_loss_coef,
                            )
                        (policy_loss * loss_scale).backward()
                        append_to_dict(
                            metrics,
                            {
                                "actor/pg_loss": pg_loss.detach().item(),
                                "actor/pg_clipfrac": clipfrac.detach().item(),
                                "actor/ppo_kl": ppo_kl.detach().item(),
                                "actor/pg_clipfrac_lower": (
                                    clipfrac_lower.detach().item()
                                ),
                            },
                        )

                    actor_norm = _actor_global_grad_norm(
                        self.actor_module,
                        fsdp2_type=FSDPModule,
                        fsdp2_norm=fsdp2_clip_grad_norm_,
                    )
                    step_metrics = self._infoskill_policy_update.step(
                        actor_global_grad_norm=actor_norm
                    )
                    self.infoskill_update_applied = (
                        self.infoskill_update_applied
                        or bool(step_metrics["policy/optimizer_step_applied"])
                    )
                    append_to_dict(metrics, dict(step_metrics))
                    append_to_dict(
                        metrics,
                        {"actor/grad_norm": float(actor_norm.detach().item())},
                    )
            self._infoskill_policy_update.zero_grad()
            append_to_dict(
                metrics,
                {
                    "projector/lr": self.infoskill_projector_scheduler.get_last_lr()[
                        0
                    ]
                },
            )
            metrics["perf/reference_logprob_seconds"] = (
                reference_logprob_seconds
            )
            metrics["perf/kl_ppo_forward_fused"] = float(
                fuse_kl_ppo_forward
            )
            return metrics

    return InfoSkillDataParallelPPOActor


def _infoskill_kl_loss(
    *,
    actor_log_prob: Tensor,
    reference_log_prob: Tensor,
    response_mask: Tensor,
    kl_penalty_fn: object,
    aggregate_loss_fn: object,
    kl_loss_type: str,
    loss_agg_mode: str,
) -> Tensor:
    kld = kl_penalty_fn(  # type: ignore[operator]
        logprob=actor_log_prob,
        ref_logprob=reference_log_prob,
        kl_penalty=kl_loss_type,
    )
    return aggregate_loss_fn(  # type: ignore[operator]
        loss_mat=kld,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )


def _append_kl_metrics(
    append_to_dict_fn: object,
    metrics: dict[str, object],
    kl_loss: Tensor,
    kl_loss_coef: float,
) -> None:
    append_to_dict_fn(  # type: ignore[operator]
        metrics,
        {
            "actor/kl_loss": kl_loss.detach().item(),
            "actor/kl_coef": kl_loss_coef,
        },
    )


def _infoskill_forward_keys() -> list[str]:
    return [
        "responses",
        "input_ids",
        "attention_mask",
        "position_ids",
        "infoskill_prefix_mask",
        "infoskill_replay_latents",
    ]


def _current_prefix_and_hook_mask(
    *,
    projector: nn.Module,
    replay_latents: Tensor,
    prefix_mask: Tensor,
    attention_mask: Tensor,
    remove_padding: bool,
    detach_projector_output: bool,
) -> tuple[Tensor, Tensor]:
    if replay_latents.ndim != 2:
        raise ValueError("INFO-SKILL replay latents must be [batch, latent_dim]")
    if prefix_mask.shape != attention_mask.shape:
        raise ValueError("INFO-SKILL prefix and attention masks must match")
    if replay_latents.shape[0] != prefix_mask.shape[0]:
        raise ValueError("INFO-SKILL replay rows must match prefix-mask rows")
    if (prefix_mask.bool() & ~attention_mask.bool()).any():
        raise ValueError("every INFO-SKILL prefix slot must be attended")
    parameter = next(projector.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    latent = replay_latents.detach().to(
        device=attention_mask.device,
        dtype=dtype,
    )
    if detach_projector_output:
        projector_module = getattr(projector, "module", projector)
        with torch.no_grad():
            prefix = projector_module(latent)
    else:
        prefix = projector(latent)
    if prefix.ndim != 3 or prefix.shape[0] != replay_latents.shape[0]:
        raise ValueError(
            "INFO-SKILL projector output must be [batch, prefix, hidden]"
        )
    expected_slots = int(prefix.shape[0] * prefix.shape[1])
    row_slots = prefix_mask.bool().sum(dim=-1)
    if not torch.equal(
        row_slots,
        torch.full_like(row_slots, int(prefix.shape[1])),
    ) or int(prefix_mask.bool().sum().item()) != expected_slots:
        raise ValueError("INFO-SKILL prefix geometry does not match placeholder slots")
    hook_mask = (
        prefix_mask.bool()[attention_mask.bool()].unsqueeze(0)
        if remove_padding
        else prefix_mask.bool()
    )
    return prefix, hook_mask


def _actor_global_grad_norm(
    actor_module: nn.Module,
    *,
    fsdp2_type: type,
    fsdp2_norm: object,
) -> Tensor:
    if isinstance(actor_module, FSDP):
        return actor_module.clip_grad_norm_(max_norm=float("inf"))
    if isinstance(actor_module, fsdp2_type):
        return fsdp2_norm(actor_module.parameters(), max_norm=float("inf"))
    return torch.nn.utils.clip_grad_norm_(
        actor_module.parameters(),
        max_norm=float("inf"),
    )
