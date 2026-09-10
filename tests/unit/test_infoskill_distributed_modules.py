from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local docs-only environment
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class InfoSkillDistributedModulesTests(unittest.TestCase):
    def test_wraps_both_modules_and_builds_projector_optimizer_domain(self) -> None:
        from infoskill.integrations.verl.distributed_modules import (
            build_distributed_infoskill_modules,
        )

        calls = []

        def wrap(module, **kwargs):
            calls.append((module, kwargs))
            return _Wrapper(module)

        compressor = torch.nn.Linear(3, 2)
        projector = torch.nn.Linear(2, 4)
        modules = build_distributed_infoskill_modules(
            compressor=compressor,
            projector=projector,
            device_index=2,
            total_policy_steps=100,
            ddp_factory=wrap,
        )

        self.assertIs(modules.compressor.module, compressor)
        self.assertIs(modules.projector.module, projector)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["device_ids"], [2])
        self.assertEqual(calls[0][1]["output_device"], 2)
        self.assertFalse(calls[0][1]["broadcast_buffers"])
        self.assertFalse(calls[0][1]["find_unused_parameters"])
        optimized = {
            id(parameter)
            for group in modules.projector_optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimized, {id(item) for item in projector.parameters()})
        group = modules.projector_optimizer.param_groups[0]
        self.assertEqual(group["initial_lr"], 1e-4)
        self.assertEqual(group["weight_decay"], 0.01)
        self.assertEqual(group["betas"], (0.9, 0.95))
        self.assertEqual(group["eps"], 1e-8)

    def test_rejects_invalid_policy_schedule(self) -> None:
        from infoskill.integrations.verl.distributed_modules import (
            build_distributed_infoskill_modules,
        )

        with self.assertRaisesRegex(ValueError, "total_policy_steps"):
            build_distributed_infoskill_modules(
                compressor=torch.nn.Linear(2, 2),
                projector=torch.nn.Linear(2, 2),
                device_index=0,
                total_policy_steps=0,
                ddp_factory=lambda module, **_: _Wrapper(module),
            )


if torch is not None:

    class _Wrapper(torch.nn.Module):
        def __init__(self, module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
