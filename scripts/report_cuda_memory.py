from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report reliable per-rank CUDA memory metrics for a training run"
    )
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    metrics_path = args.run / "metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError(f"training metrics do not exist: {metrics_path}")
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training = [record for record in records if record.get("phase") == "train"]
    if not training:
        raise RuntimeError(f"no training records in {metrics_path}")
    latest = training[-1]
    cuda = {
        key.removeprefix("perf/cuda/"): value
        for key, value in latest.items()
        if key.startswith("perf/cuda/")
    }
    if not cuda:
        raise RuntimeError(
            "run has no per-rank CUDA metrics; pull the instrumentation update and "
            "run a new persistent-session benchmark"
        )
    payload = {
        "run": str(args.run.resolve()),
        "step": latest.get("step"),
        "core_update_seconds": latest.get("perf/core_update_seconds"),
        "rollout_seconds": latest.get("perf/rollout_seconds"),
        "policy_update_seconds": latest.get("perf/policy_update_seconds"),
        "cuda": dict(sorted(cuda.items())),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
