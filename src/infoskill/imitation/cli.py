from __future__ import annotations

import argparse
import json
from pathlib import Path

from infoskill.integrations.webshop import (
    REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
    REGISTERED_HUMAN_GOALS_SHA256,
    REGISTERED_TRAIN_TRAJECTORY_COUNT,
)

from .audit import (
    audit_prepared_imitation_data,
    validate_webshop_audit_report,
)
from .dataset import (
    prepare_alfworld_imitation_data,
    prepare_demonstration_imitation_data,
)
from .handoff import create_handoff
from .skill_bank import (
    build_planner_skill_bank,
    build_webshop_skill_bank,
    rewrite_grounding_skill_ids,
)


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
    webshop.add_argument(
        "--expected-trajectories",
        type=int,
        default=REGISTERED_TRAIN_TRAJECTORY_COUNT,
    )
    webshop.add_argument(
        "--expected-demonstrations-sha256",
        default=REGISTERED_HUMAN_DEMONSTRATIONS_SHA256,
    )
    webshop.add_argument(
        "--expected-human-goals-sha256",
        default=REGISTERED_HUMAN_GOALS_SHA256,
    )
    audit_webshop = commands.add_parser("audit-webshop")
    audit_webshop.add_argument("--data", required=True)
    audit_webshop.add_argument("--model", required=True)
    audit_webshop.add_argument("--output", required=True)
    audit_webshop.add_argument("--max-length", type=int, default=16384)
    verify_webshop = commands.add_parser("verify-webshop-audit")
    verify_webshop.add_argument("--data", required=True)
    verify_webshop.add_argument("--audit", required=True)
    verify_webshop.add_argument("--model", required=True)
    verify_webshop.add_argument("--max-length", type=int, default=16384)
    build_webshop_bank = commands.add_parser("build-webshop-skill-bank")
    build_webshop_bank.add_argument("--data", required=True)
    build_webshop_bank.add_argument("--audit", required=True)
    build_webshop_bank.add_argument("--output", required=True)
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
    train.add_argument("--audit")
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
            expected_source_checksums={
                "human_demonstrations": args.expected_demonstrations_sha256,
                "human_goals": args.expected_human_goals_sha256,
            },
        )
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "audit-webshop":
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "WebShop token audit requires transformers"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
        )
        report = audit_prepared_imitation_data(
            args.data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            output_path=args.output,
        )
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    if args.command == "verify-webshop-audit":
        data = Path(args.data).expanduser().resolve()
        report = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("WebShop imitation audit report must be an object")
        source_checksums = {
            name: _sha256_file(data / filename)
            for name, filename in (
                ("manifest", "manifest.json"),
                ("train", "train.jsonl"),
                ("validation", "validation.jsonl"),
            )
        }
        validate_webshop_audit_report(
            report,
            data_directory=data,
            source_checksums=source_checksums,
            expected_tokenizer=args.model,
            expected_max_length=args.max_length,
        )
        print(
            json.dumps(
                {"passed": True, "audit": str(Path(args.audit).resolve())},
                indent=2,
            )
        )
        return 0
    if args.command == "build-webshop-skill-bank":
        report = build_webshop_skill_bank(args.data, args.output, args.audit)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "train":
        from .train import train_actor_imitation

        data = Path(args.data).expanduser().resolve()
        data_manifest = json.loads(
            (data / "manifest.json").read_text(encoding="utf-8")
        )
        if data_manifest.get("environment") == "webshop":
            if not args.audit:
                raise ValueError(
                    "WebShop imitation training requires a passed audit report"
                )
            audit_report = json.loads(
                Path(args.audit).read_text(encoding="utf-8")
            )
            if not isinstance(audit_report, dict):
                raise ValueError("WebShop imitation audit report must be an object")
            source_checksums = {
                name: _sha256_file(data / filename)
                for name, filename in (
                    ("manifest", "manifest.json"),
                    ("train", "train.jsonl"),
                    ("validation", "validation.jsonl"),
                )
            }
            validate_webshop_audit_report(
                audit_report,
                data_directory=data,
                source_checksums=source_checksums,
                expected_tokenizer=args.model,
                expected_max_length=args.max_length,
            )
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
            audit_report=args.audit,
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


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
