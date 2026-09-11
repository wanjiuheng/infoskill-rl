from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import nn


MODULE_FILE = "infoskill_modules.pt"
OPTIMIZER_FILE = "infoskill_optimizers.pt"
SCHEDULER_FILE = "infoskill_schedulers.pt"
RNG_FILE = "infoskill_rng_state.pt"
MANIFEST_FILE = "infoskill_manifest.json"


def save_infoskill_state(
    directory: str | Path,
    *,
    modules: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
) -> Mapping[str, object]:
    """Persist rank-0 replicated M1 state without frozen model weights."""

    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    _require_names(modules, {"compressor", "projector", "prior", "fidelity", "grounding"})
    _require_names(optimizers, {"projector", "auxiliary"})
    _require_names(schedulers, {"projector", "auxiliary"})
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    module_state = {
        name: _to_cpu(_unwrap(module).state_dict())
        for name, module in modules.items()
    }
    optimizer_state = {
        name: _to_cpu(optimizer.state_dict())
        for name, optimizer in optimizers.items()
    }
    scheduler_state = {
        name: scheduler.state_dict()
        for name, scheduler in schedulers.items()
    }
    torch.save(module_state, destination / MODULE_FILE)
    torch.save(optimizer_state, destination / OPTIMIZER_FILE)
    torch.save(scheduler_state, destination / SCHEDULER_FILE)
    torch.save(_rng_state(), destination / RNG_FILE)
    manifest = {
        "schema_version": 1,
        "global_step": global_step,
        "format": "infoskill-replicated-rank0-v1",
        "base_weights_included": False,
        "module_names": sorted(modules),
        "optimizer_names": sorted(optimizers),
        "scheduler_names": sorted(schedulers),
        "files": [MODULE_FILE, OPTIMIZER_FILE, SCHEDULER_FILE, RNG_FILE],
    }
    (destination / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_infoskill_state(
    directory: str | Path,
    *,
    modules: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Mapping[str, torch.optim.lr_scheduler.LRScheduler],
    expected_global_step: int | None = None,
) -> Mapping[str, object]:
    """Restore replicated M1 state on every rank from the rank-0 artifact."""

    source = Path(directory)
    manifest_path = source / MANIFEST_FILE
    if not manifest_path.is_file():
        raise RuntimeError(f"portable INFO-SKILL state is incomplete: {source}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported portable INFO-SKILL state manifest")
    if expected_global_step is not None and manifest.get("global_step") != expected_global_step:
        raise RuntimeError("INFO-SKILL and actor checkpoint steps differ")
    _require_manifest_names(manifest, "module_names", modules)
    _require_manifest_names(manifest, "optimizer_names", optimizers)
    _require_manifest_names(manifest, "scheduler_names", schedulers)
    for relative in manifest.get("files", []):
        if not isinstance(relative, str) or not (source / relative).is_file():
            raise RuntimeError(f"portable INFO-SKILL file is missing: {relative}")

    module_state = torch.load(source / MODULE_FILE, map_location="cpu", weights_only=True)
    optimizer_state = torch.load(
        source / OPTIMIZER_FILE,
        map_location="cpu",
        weights_only=False,
    )
    scheduler_state = torch.load(
        source / SCHEDULER_FILE,
        map_location="cpu",
        weights_only=False,
    )
    rng_state = torch.load(
        source / RNG_FILE,
        map_location="cpu",
        weights_only=False,
    )
    for name, module in modules.items():
        _unwrap(module).load_state_dict(module_state[name], strict=True)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_state[name])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(scheduler_state[name])
    _restore_rng_state(rng_state)
    return manifest


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(state: object) -> None:
    if not isinstance(state, dict):
        raise RuntimeError("portable INFO-SKILL RNG state is invalid")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_states = state["torch_cuda"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("portable INFO-SKILL RNG state is incomplete") from error
    if torch.cuda.is_available():
        if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "portable INFO-SKILL CUDA RNG topology differs from this worker"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _unwrap(module: nn.Module) -> nn.Module:
    wrapped = getattr(module, "module", None)
    return wrapped if isinstance(wrapped, nn.Module) else module


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _require_names(values: Mapping[str, object], expected: set[str]) -> None:
    if set(values) != expected:
        raise ValueError(
            f"INFO-SKILL state names differ: {sorted(values)} != {sorted(expected)}"
        )


def _require_manifest_names(
    manifest: Mapping[str, object],
    field: str,
    current: Mapping[str, object],
) -> None:
    recorded = manifest.get(field)
    if not isinstance(recorded, list) or set(recorded) != set(current):
        raise RuntimeError(f"portable INFO-SKILL {field} differ from runtime")
