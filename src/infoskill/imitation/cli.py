from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import (
    prepare_alfworld_imitation_data,
    prepare_demonstration_imitation_data,
)
from .handoff import create_handoff
from .skill_bank import build_planner_skill_bank, rewrite_grounding_skill_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infoskill-imitation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--grounding", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--skill-bank", required=True)
    prepare.add_argument("--validation-fraction", type=float, default=0.02)
    prepare.add_argument("--split-seed", type=int, default=0)
    prepare.add_argument("--expected-trajectories", type=int, default=3521)
    webshop = commands.add_parser("prepare-webshop")
    webshop.add_argument("--demonstrations", required=True)
    webshop.add_argument("--human-goals", required=True)
    webshop.add_argument("--output", required=True)
    webshop.add_argument("--validation-fraction", type=float, default=0.02)
    webshop.add_argument("--split-seed", type=int, default=0)
    webshop.add_argument("--expected-trajectories", type=int, default=1012)
    train = commands.add_parser("train")
    train.add_argument("--model", required=True)
    train.add_argument("--base-model-id", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--epochs", type=float, default=2.0)
    train.add_argument("--per-device-batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--max-length", type=int, default=4352)
    train.add_argument("--resume-from-checkpoint")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--adapter", required=True)
    finalize.add_argument("--imitation-manifest", required=True)
    finalize.add_argument("--skill-bank", required=True)
    finalize.add_argument("--base-model-id", required=True)
    finalize.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        grounding = Path(args.grounding)
        output = Path(args.output)
        rewritten_grounding = output.with_name(f"{output.name}-grounding")
        bank = build_planner_skill_bank(
            grounding / "grounding_samples.jsonl", args.skill_bank
        )
        derived = rewrite_grounding_skill_ids(
            grounding, args.skill_bank, rewritten_grounding
        )
        manifest = prepare_alfworld_imitation_data(
            grounding_directory=rewritten_grounding,
            output_directory=output,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            expected_trajectory_count=args.expected_trajectories,
        )
        print(json.dumps({"skill_bank": bank, "grounding": derived, "imitation": manifest}, indent=2))
        return 0
    if args.command == "prepare-webshop":
        from infoskill.integrations.webshop import (
            OfficialWebShopHumanDemonstrationProvider,
        )

        manifest = prepare_demonstration_imitation_data(
            provider=OfficialWebShopHumanDemonstrationProvider(
                args.demonstrations,
                args.human_goals,
            ),
            output_directory=args.output,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
            expected_trajectory_count=args.expected_trajectories,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "train":
        from .train import train_actor_imitation

        data = Path(args.data)
        train_actor_imitation(
            model_path=args.model,
            base_model_id=args.base_model_id,
            train_file=data / "train.jsonl",
            validation_file=data / "validation.jsonl",
            output_directory=args.output,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        return 0
    manifest = create_handoff(
        adapter_directory=args.adapter,
        imitation_manifest=args.imitation_manifest,
        skill_bank=args.skill_bank,
        base_model_id=args.base_model_id,
        output_directory=args.output,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
