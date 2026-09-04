from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_bytes(source: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _git(source: Path, *args: str) -> str:
    return _git_bytes(source, *args).decode("utf-8").strip()


def _sha256_git_blob(source: Path, commit: str, relative: str) -> str:
    content = _git_bytes(source, "show", f"{commit}:{relative}")
    return hashlib.sha256(content).hexdigest()


def verify(source: Path, patch_root: Path, *, require_clean: bool = True) -> dict[str, object]:
    manifest = json.loads((patch_root / "manifest.json").read_text(encoding="utf-8"))
    actual_commit = _git(source, "rev-parse", "HEAD")
    if actual_commit != manifest["upstream_commit"]:
        raise RuntimeError(
            f"vLLM commit mismatch: expected {manifest['upstream_commit']}, got {actual_commit}"
        )
    status = _git(source, "status", "--short")
    if require_clean and status:
        raise RuntimeError("vLLM source is not clean; refuse to build an unauditable runtime")

    for relative, expected in manifest["upstream_file_sha256"].items():
        actual = _sha256_git_blob(source, manifest["upstream_commit"], relative)
        if actual != expected:
            raise RuntimeError(f"upstream checksum mismatch for {relative}: {actual}")
    for relative, expected in manifest["patch_sha256"].items():
        actual = _sha256(patch_root / relative)
        if actual != expected:
            raise RuntimeError(f"patch checksum mismatch for {relative}: {actual}")
    return {
        "runtime_id": manifest["runtime_id"],
        "upstream_commit": actual_commit,
        "source_clean": not bool(status),
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the pinned INFO-SKILL vLLM patch input")
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--patch-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "third_party"
        / "patches"
        / "vllm-0.8.4",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    report = verify(
        args.source.resolve(),
        args.patch_root.resolve(),
        require_clean=not args.allow_dirty,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
