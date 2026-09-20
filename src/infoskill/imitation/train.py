from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any


def train_actor_imitation(
    *,
    model_path: str,
    base_model_id: str,
    train_file: str | Path,
    validation_file: str | Path,
    output_directory: str | Path,
    learning_rate: float = 1e-4,
    epochs: float = 2.0,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 4352,
    resume_from_checkpoint: str | Path | None = None,
    audit_report: str | Path | None = None,
) -> None:
    """LoRA SFT with prompt tokens masked from the language-model loss."""

    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning rate and epochs must be positive")
    if min(per_device_batch_size, gradient_accumulation_steps, max_length) <= 0:
        raise ValueError("batch and sequence settings must be positive")
    audit_provenance: dict[str, str] | None = None
    if audit_report is not None:
        audit_path = Path(audit_report).expanduser().resolve()
        audit_provenance = {
            "imitation_audit": str(audit_path),
            "imitation_audit_sha256": _sha256(audit_path),
        }
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "actor imitation requires torch, transformers, peft, and accelerate"
        ) from error

    from infoskill.persistence.model_identity import verify_policy_model_identity

    policy_model_identity = verify_policy_model_identity(
        model_path,
        model_id=base_model_id,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    class PromptResponseDataset(Dataset):
        def __init__(self, path: str | Path) -> None:
            self.rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not self.rows:
                raise ValueError(f"empty imitation dataset: {path}")

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = self.rows[index]
            return encode_training_example(
                tokenizer,
                prompt=str(row["prompt"]),
                response=str(row["response"]),
                max_length=max_length,
            )

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(item["input_ids"]) for item in features)
        inputs, masks, labels = [], [], []
        for item in features:
            padding = width - len(item["input_ids"])
            inputs.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    destination = Path(output_directory)
    arguments = TrainingArguments(
        output_dir=str(destination),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to=[],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=0,
        data_seed=0,
        **_evaluation_strategy_kwargs(TrainingArguments),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=PromptResponseDataset(train_file),
        eval_dataset=PromptResponseDataset(validation_file),
        data_collator=collate,
    )
    trainer.train(
        resume_from_checkpoint=(
            str(Path(resume_from_checkpoint).expanduser().resolve())
            if resume_from_checkpoint is not None
            else None
        )
    )
    final_adapter = destination / "final-adapter"
    trainer.save_model(str(final_adapter))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_adapter))
        training_manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "kind": "actor_imitation_lora_sft",
            "model_path": str(Path(model_path).expanduser().resolve()),
            "policy_model": policy_model_identity.as_dict(),
            "train_file_sha256": _sha256(Path(train_file)),
            "validation_file_sha256": _sha256(Path(validation_file)),
            "learning_rate": learning_rate,
            "epochs": epochs,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_batch_size": (
                per_device_batch_size
                * gradient_accumulation_steps
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
            "max_length": max_length,
            "prompt_encoding": "qwen-chat-template-add-generation-prompt",
            "response_loss": "response-only",
            "seed": 0,
            "global_step": int(trainer.state.global_step),
        }
        if audit_provenance is not None:
            training_manifest.update(audit_provenance)
        _write_training_manifest(
            final_adapter / "imitation-training-manifest.json",
            training_manifest,
        )


def encode_training_example(
    tokenizer: object,
    *,
    prompt: str,
    response: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Encode exactly the online chat prompt and mask it from SFT loss."""

    prompt_ids = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_ids = [int(value) for value in prompt_ids]
    if len(prompt_ids) >= max_length:
        raise ValueError(
            "imitation prompt leaves no response-loss tokens within max_length"
        )
    response_ids = tokenizer(  # type: ignore[operator]
        response,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - len(prompt_ids) - 1,
    )["input_ids"]
    response_ids = [int(value) for value in response_ids]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("imitation tokenizer requires an EOS token")
    response_ids.append(int(eos_token_id))
    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def _evaluation_strategy_kwargs(
    training_arguments_type: type[object],
) -> dict[str, str]:
    """Bridge the Transformers evaluation_strategy -> eval_strategy rename."""

    parameters = inspect.signature(training_arguments_type.__init__).parameters
    if "eval_strategy" in parameters:
        return {"eval_strategy": "steps"}
    if "evaluation_strategy" in parameters:
        return {"evaluation_strategy": "steps"}
    raise RuntimeError(
        "installed Transformers TrainingArguments exposes no evaluation strategy"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_training_manifest(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
