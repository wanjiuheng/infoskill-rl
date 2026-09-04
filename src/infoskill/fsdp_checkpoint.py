from __future__ import annotations

from collections.abc import Mapping


def load_peft_adapter_under_full_fsdp_state(
    *,
    fsdp_model: object,
    peft_model: object,
    adapter_state: Mapping[str, object],
) -> object:
    """Load ordinary adapter tensors while every nested FSDP hook expects full state."""
    from peft import set_peft_model_state_dict
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel,
        StateDictType,
    )

    full_state_config = FullStateDictConfig(
        offload_to_cpu=False,
        rank0_only=False,
    )
    with FullyShardedDataParallel.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        full_state_config,
    ):
        with FullyShardedDataParallel.summon_full_params(
            fsdp_model,
            recurse=True,
            writeback=True,
            rank0_only=False,
            offload_to_cpu=False,
        ):
            return set_peft_model_state_dict(
                peft_model,
                adapter_state,
                adapter_name="default",
            )
