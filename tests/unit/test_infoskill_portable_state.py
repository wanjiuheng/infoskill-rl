from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "requires torch")
class InfoSkillPortableStateTests(unittest.TestCase):
    def test_round_trip_restores_all_modules_optimizers_and_schedulers(self) -> None:
        from infoskill.persistence.infoskill_state import (
            load_infoskill_state,
            save_infoskill_state,
        )

        modules = {
            name: torch.nn.Linear(2, 2)
            for name in ("compressor", "projector", "prior", "fidelity", "grounding")
        }
        projector_optimizer = torch.optim.AdamW(modules["projector"].parameters())
        auxiliary_optimizer = torch.optim.AdamW(
            parameter
            for name, module in modules.items()
            if name != "projector"
            for parameter in module.parameters()
        )
        optimizers = {
            "projector": projector_optimizer,
            "auxiliary": auxiliary_optimizer,
        }
        schedulers = {
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            for name, optimizer in optimizers.items()
        }
        for optimizer in optimizers.values():
            optimizer.zero_grad()
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        for scheduler in schedulers.values():
            scheduler.step()
        expected = {
            name: {key: value.detach().clone() for key, value in module.state_dict().items()}
            for name, module in modules.items()
        }
        random.seed(17)
        torch.manual_seed(23)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            save_infoskill_state(
                temporary,
                modules=modules,
                optimizers=optimizers,
                schedulers=schedulers,
                global_step=7,
            )
            expected_python = random.random()
            expected_torch = torch.rand(1)
            with torch.no_grad():
                for module in modules.values():
                    for parameter in module.parameters():
                        parameter.zero_()
            manifest = load_infoskill_state(
                temporary,
                modules=modules,
                optimizers=optimizers,
                schedulers=schedulers,
                expected_global_step=7,
            )
            actual_python = random.random()
            actual_torch = torch.rand(1)

        self.assertEqual(manifest["global_step"], 7)
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name][key]))
        self.assertTrue(projector_optimizer.state)
        self.assertTrue(auxiliary_optimizer.state)
        self.assertEqual(actual_python, expected_python)
        self.assertTrue(torch.equal(actual_torch, expected_torch))


if __name__ == "__main__":
    unittest.main()
