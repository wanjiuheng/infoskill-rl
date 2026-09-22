from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


_IDENTITY_VERSION = "infoskill-policy-model-v2"
_METADATA_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


@dataclass(frozen=True, slots=True)
class PinnedPolicyModel:
    model_id: str
    revision: str
    sha256: str


_PINNED_POLICY_MODELS = {
    "alfworld-7b-sft-checkpoint-140": PinnedPolicyModel(
        model_id="alfworld-7b-sft-checkpoint-140",
        revision="Alfworld-7B-SFT/checkpoint-140",
        sha256="ede304d8ae0fb27df55a9bcf22482b8a7d83626a4711f9525e0388f7b3d39d99",
    ),
    "qwen2.5-7b-instruct": PinnedPolicyModel(
        model_id="qwen2.5-7b-instruct",
        revision="Qwen/Qwen2.5-7B-Instruct",
        sha256="8305dee0a659a8f9e0650129eaaf584006338a42f237d071ef5cdbaed91fc14a",
    ),
    "qwen3-1.7b": PinnedPolicyModel(
        model_id="qwen3-1.7b",
        revision="Qwen/Qwen3-1.7B",
        sha256="9b0a7e2fff78e5746a5564de06f43386d277891b672be818fa45c10855716c5b",
    ),
}


@dataclass(frozen=True, slots=True)
class PolicyModelFileIdentity:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PolicyModelIdentity:
    schema_version: int
    algorithm: str
    path: str
    model_id: str | None
    revision: str | None
    sha256: str
    file_count: int
    total_bytes: int
    files: tuple[PolicyModelFileIdentity, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def fingerprint_policy_model(path: str | Path) -> PolicyModelIdentity:
    """Hash behavior-defining top-level Hugging Face model files."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"policy model directory does not exist: {root}")
    if not (root / "config.json").is_file():
        raise RuntimeError(f"policy model has no config.json: {root}")
    selected = sorted(
        (
            candidate
            for candidate in root.iterdir()
            if candidate.is_file() and _is_identity_file(candidate.name)
        ),
        key=lambda candidate: candidate.name,
    )
    weights = [
        candidate
        for candidate in selected
        if candidate.suffix == ".safetensors"
        or (
            candidate.suffix == ".bin"
            and candidate.name.startswith("pytorch_model")
        )
    ]
    if not weights:
        raise RuntimeError(f"policy model has no supported weight files: {root}")

    files = tuple(
        PolicyModelFileIdentity(
            relative_path=candidate.name,
            size_bytes=candidate.stat().st_size,
            sha256=_sha256_file(candidate),
        )
        for candidate in selected
    )
    combined = hashlib.sha256()
    combined.update(f"{_IDENTITY_VERSION}\0".encode())
    for item in files:
        combined.update(
            f"{item.relative_path}\0{item.size_bytes}\0{item.sha256}\n".encode()
        )
    return PolicyModelIdentity(
        schema_version=1,
        algorithm=_IDENTITY_VERSION,
        path=str(root),
        model_id=None,
        revision=None,
        sha256=combined.hexdigest(),
        file_count=len(files),
        total_bytes=sum(item.size_bytes for item in files),
        files=files,
    )


def verify_policy_model_identity(
    path: str | Path,
    *,
    model_id: str | None,
) -> PolicyModelIdentity:
    """Verify a local path against an independently registered policy model."""

    if not model_id or not model_id.strip():
        raise ValueError("policy_model_id is required for training")
    normalized_id = model_id.strip()
    pinned = get_pinned_policy_model(normalized_id)
    identity = fingerprint_policy_model(path)
    if identity.sha256 != pinned.sha256:
        raise RuntimeError(
            "policy model fingerprint mismatch: "
            f"registered {pinned.sha256}, got {identity.sha256}"
        )
    return replace(
        identity,
        model_id=pinned.model_id,
        revision=pinned.revision,
    )


def get_pinned_policy_model(model_id: str) -> PinnedPolicyModel:
    """Return the independently registered identity for a stable model ID."""

    pinned = _PINNED_POLICY_MODELS.get(model_id)
    if pinned is None:
        raise ValueError(f"policy_model_id is not registered: {model_id}")
    return pinned


def provenance_matches_pinned_model(
    provenance: object,
    *,
    model_id: str,
) -> bool:
    """Check a checkpoint identity manifest against the current registration."""

    if not isinstance(provenance, dict):
        return False
    identity = provenance.get("policy_model")
    if not isinstance(identity, dict):
        return False
    pinned = get_pinned_policy_model(model_id)
    return (
        identity.get("algorithm") == _IDENTITY_VERSION
        and identity.get("model_id") == pinned.model_id
        and identity.get("revision") == pinned.revision
        and identity.get("sha256") == pinned.sha256
    )


def _is_identity_file(name: str) -> bool:
    return (
        name in _METADATA_FILES
        or name.endswith(".py")
        or name.endswith(".safetensors")
        or (name.startswith("pytorch_model") and name.endswith(".bin"))
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fingerprint a Hugging Face policy model directory"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-id")
    args = parser.parse_args(argv)
    identity = (
        verify_policy_model_identity(args.model, model_id=args.model_id)
        if args.model_id
        else fingerprint_policy_model(args.model)
    )
    print(json.dumps(identity.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
