from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = {
    "products": "data/items_shuffle.json",
    "product_attributes": "data/items_ins_v2.json",
    "human_instructions": "data/items_human_ins.json",
    "human_demonstrations": "baseline_models/data/il_trajs_finalized_images.jsonl",
    "human_goals": "baseline_models/data/human_goals.json",
}

REQUIRED_MODULES = (
    "bs4",
    "flask",
    "gym",
    "pyserini",
    "rank_bm25",
    "spacy",
    "thefuzz",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.webshop_root).expanduser().resolve()
    files = {
        name: _file_status(root / relative)
        for name, relative in REQUIRED_FILES.items()
    }
    index_root = root / "search_engine/indexes"
    index_files = (
        tuple(path for path in index_root.rglob("*") if path.is_file())
        if index_root.is_dir()
        else ()
    )
    modules = {
        name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES
    }
    web_agent_site = root / "web_agent_site"
    modules["web_agent_site"] = web_agent_site.is_dir()
    java = _java_status()
    disk = shutil.disk_usage(root if root.exists() else root.parent)
    report = {
        "schema_version": 1,
        "webshop_root": str(root),
        "python": sys.executable,
        "files": files,
        "search_index": {
            "path": str(index_root),
            "present": bool(index_files),
            "file_count": len(index_files),
        },
        "python_modules": modules,
        "java": java,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    report["data_ready"] = all(item["present"] for item in files.values())
    report["runtime_ready"] = (
        all(modules.values()) and java["present"] and bool(index_files)
    )
    report["formal_ready"] = report["data_ready"] and report["runtime_ready"]
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    print(text, end="")
    return 0 if report["formal_ready"] else 2


def _file_status(path: Path) -> dict[str, object]:
    present = path.is_file()
    return {
        "path": str(path),
        "present": present,
        "bytes": path.stat().st_size if present else None,
    }


def _java_status() -> dict[str, object]:
    executable = shutil.which("java")
    if executable is None:
        return {"present": False, "executable": None, "version": None}
    completed = subprocess.run(
        [executable, "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version = (completed.stderr or completed.stdout).splitlines()
    return {
        "present": completed.returncode == 0,
        "executable": executable,
        "version": version[0] if version else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
