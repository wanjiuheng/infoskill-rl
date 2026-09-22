#!/usr/bin/env python3
"""Materialize an immutable validation manifest from the public GiGPO protocol."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from infoskill.integrations.webshop.paper_protocol import (
    build_paper128_manifest,
    write_paper128_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", required=True)
    parser.add_argument("--products", required=True)
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--human-instructions", required=True)
    parser.add_argument("--environment-commit", required=True)
    parser.add_argument("--validation-call-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    webshop_root = Path(args.webshop_root).expanduser().resolve(strict=True)
    products_path = Path(args.products).expanduser().resolve(strict=True)
    attributes_path = Path(args.attributes).expanduser().resolve(strict=True)
    human_instructions_path = (
        Path(args.human_instructions).expanduser().resolve(strict=True)
    )
    output_path = Path(args.output).expanduser().resolve()

    expected_names = {
        products_path: "items_shuffle_1000.json",
        attributes_path: "items_ins_v2_1000.json",
        human_instructions_path: "items_human_ins.json",
    }
    for path, expected_name in expected_names.items():
        if path.name != expected_name:
            raise ValueError(
                f"paper protocol requires {expected_name}, got {path.name}"
            )
    raw_products = json.loads(products_path.read_text(encoding="utf-8"))
    if not isinstance(raw_products, list) or len(raw_products) != 1000:
        raise ValueError("paper protocol product snapshot must contain exactly 1000 rows")

    sys.path.insert(0, str(webshop_root))
    from web_agent_site.engine import engine as webshop_engine  # noqa: PLC0415
    from web_agent_site.engine.goal import get_goals  # noqa: PLC0415

    # The upstream loader opens this file even when synthetic goals are used.
    webshop_engine.HUMAN_ATTR_PATH = str(human_instructions_path)
    all_products, _, product_prices, _ = webshop_engine.load_products(
        filepath=str(products_path),
        attrpath=str(attributes_path),
        num_products=None,
        human_goals=False,
    )

    def goal_factory(worker_seed: int):
        # Match WebAgentTextEnv and SimServer: seed before goal generation, then
        # reset the same seed immediately before shuffling the generated list.
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        goals = get_goals(all_products, product_prices, human_goals=False)
        random.seed(worker_seed)
        random.shuffle(goals)
        return goals

    manifest = build_paper128_manifest(
        goal_factory=goal_factory,
        source_files={
            "products": products_path,
            "attributes": attributes_path,
            "human_instructions": human_instructions_path,
        },
        environment_commit=args.environment_commit,
        validation_call_index=args.validation_call_index,
    )
    write_paper128_manifest(output_path, manifest)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "task_sequence_sha256": manifest["task_sequence_sha256"],
                "task_count": len(manifest["tasks"]),
                "validation_call_index": args.validation_call_index,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
