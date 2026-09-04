from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from infoskill.fsdp_checkpoint import load_peft_adapter_under_full_fsdp_state


class _Model:
    full_state_active = False
    full_params_active = False


class _FullStateDictConfig:
    def __init__(self, *, offload_to_cpu: bool, rank0_only: bool) -> None:
        self.offload_to_cpu = offload_to_cpu
        self.rank0_only = rank0_only


class _StateDictType:
    FULL_STATE_DICT = "full"


class _FSDP:
    events: list[str] = []

    @staticmethod
    @contextmanager
    def state_dict_type(model, state_type, config):
        if state_type != _StateDictType.FULL_STATE_DICT:
            raise AssertionError("adapter load did not request a full state dict")
        if config.offload_to_cpu or config.rank0_only:
            raise AssertionError("all ranks must load device-resident adapter tensors")
        _FSDP.events.append("enter-full-state")
        model.full_state_active = True
        try:
            yield
        finally:
            model.full_state_active = False
            _FSDP.events.append("exit-full-state")

    @staticmethod
    @contextmanager
    def summon_full_params(model, **kwargs):
        if not model.full_state_active:
            raise AssertionError("full parameters were summoned outside full-state mode")
        if kwargs != {
            "recurse": True,
            "writeback": True,
            "rank0_only": False,
            "offload_to_cpu": False,
        }:
            raise AssertionError(f"unexpected summon settings: {kwargs}")
        _FSDP.events.append("enter-full-params")
        model.full_params_active = True
        try:
            yield
        finally:
            model.full_params_active = False
            _FSDP.events.append("exit-full-params")


class FsdpCheckpointTests(unittest.TestCase):
    def test_peft_adapter_load_is_scoped_by_full_state_and_full_params(self) -> None:
        root = _Model()
        peft_model = types.SimpleNamespace(fsdp_root=root)
        adapter_state = {"lora_A.weight": object()}
        expected = types.SimpleNamespace(unexpected_keys=[])

        def set_adapter(model, state, *, adapter_name):
            self.assertIs(model, peft_model)
            self.assertIs(state, adapter_state)
            self.assertEqual(adapter_name, "default")
            self.assertTrue(root.full_state_active)
            self.assertTrue(root.full_params_active)
            _FSDP.events.append("load-adapter")
            return expected

        peft = types.ModuleType("peft")
        peft.set_peft_model_state_dict = set_adapter
        torch = types.ModuleType("torch")
        distributed = types.ModuleType("torch.distributed")
        fsdp = types.ModuleType("torch.distributed.fsdp")
        fsdp.FullyShardedDataParallel = _FSDP
        fsdp.FullStateDictConfig = _FullStateDictConfig
        fsdp.StateDictType = _StateDictType
        torch.distributed = distributed
        distributed.fsdp = fsdp
        _FSDP.events = []

        with patch.dict(
            sys.modules,
            {
                "peft": peft,
                "torch": torch,
                "torch.distributed": distributed,
                "torch.distributed.fsdp": fsdp,
            },
        ):
            result = load_peft_adapter_under_full_fsdp_state(
                fsdp_model=root,
                peft_model=peft_model,
                adapter_state=adapter_state,
            )

        self.assertIs(result, expected)
        self.assertEqual(
            _FSDP.events,
            [
                "enter-full-state",
                "enter-full-params",
                "load-adapter",
                "exit-full-params",
                "exit-full-state",
            ],
        )


if __name__ == "__main__":
    unittest.main()
