from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
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
    "cleantext",
    "faiss",
    "flask",
    "gym",
    "pyserini",
    "rank_bm25",
    "selenium",
    "spacy",
    "thefuzz",
)

EXPECTED_VERSIONS = {
    "beautifulsoup4": "4.11.1",
    "cleantext": "1.1.4",
    "faiss-cpu": "1.7.4",
    "Flask": "2.1.2",
    "gym": "0.24.0",
    "pyserini": "0.17.0",
    "rank-bm25": "0.2.2",
    "selenium": "4.2.0",
    "spacy": "3.7.2",
    "thefuzz": "0.19.0",
    "Werkzeug": "2.1.0",
}

MINIMUM_FREE_DISK_BYTES = 15 * 1024**3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", required=True)
    parser.add_argument("--webshop-data-root")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.webshop_root).expanduser().resolve()
    data_root = (
        Path(args.webshop_data_root).expanduser().resolve()
        if args.webshop_data_root
        else root
    )
    files = {
        name: _file_status(data_root / relative)
        for name, relative in REQUIRED_FILES.items()
    }
    index_root = data_root / "search_engine/indexes"
    index_files = (
        tuple(path for path in index_root.rglob("*") if path.is_file())
        if index_root.is_dir()
        else ()
    )
    index_manifest = _index_manifest_status(data_root / "search_engine/index-manifest.json")
    modules = {
        name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES
    }
    modules["en_core_web_sm"] = importlib.util.find_spec("en_core_web_sm") is not None
    package_versions = {
        name: _package_version(name) for name in EXPECTED_VERSIONS
    }
    version_matches = {
        name: package_versions[name] == expected
        for name, expected in EXPECTED_VERSIONS.items()
    }
    web_agent_site = _web_agent_site_status(root)
    java = _java_status()
    disk_probe = data_root if data_root.exists() else _existing_parent(data_root)
    disk = shutil.disk_usage(disk_probe)
    report = {
        "schema_version": 1,
        "webshop_root": str(root),
        "webshop_data_root": str(data_root),
        "python": sys.executable,
        "files": files,
        "search_index": {
            "path": str(index_root),
            "present": bool(index_files) and index_manifest["complete"],
            "file_count": len(index_files),
            "manifest": index_manifest,
        },
        "python_modules": modules,
        "python_package_versions": package_versions,
        "python_package_version_matches": version_matches,
        "web_agent_site": web_agent_site,
        "java": java,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    report["data_ready"] = all(item["present"] for item in files.values())
    report["disk_ready"] = disk.free >= MINIMUM_FREE_DISK_BYTES
    report["runtime_ready"] = (
        all(modules.values())
        and all(version_matches.values())
        and web_agent_site["importable"]
        and java["compatible"]
        and report["search_index"]["present"]
    )
    report["formal_ready"] = (
        report["data_ready"]
        and report["runtime_ready"]
        and report["disk_ready"]
    )
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


def _index_manifest_status(path: Path) -> dict[str, object]:
    status: dict[str, object] = {"path": str(path), "complete": False}
    if not path.is_file():
        return status
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        status["error"] = str(error)
        return status
    status["document_count"] = report.get("document_count")
    status["complete"] = (
        report.get("schema_version") == 1
        and report.get("status") == "complete"
        and isinstance(report.get("document_count"), int)
        and report["document_count"] > 100_000
        and isinstance(report.get("documents_sha256"), str)
        and report.get("index_file_count", 0) > 0
        and report.get("probe_hits", 0) > 0
    )
    return status


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _java_status() -> dict[str, object]:
    executable = shutil.which("java")
    if executable is None:
        return {
            "present": False,
            "compatible": False,
            "executable": None,
            "version": None,
        }
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
        "compatible": (
            completed.returncode == 0
            and bool(version)
            and _is_java_11(version[0])
        ),
        "executable": executable,
        "version": version[0] if version else None,
    }


def _is_java_11(version_line: str) -> bool:
    lowered = version_line.lower()
    return 'version "11.' in lowered or "openjdk 11." in lowered


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _web_agent_site_status(root: Path) -> dict[str, object]:
    source = root / "web_agent_site"
    if not source.is_dir():
        return {
            "source_present": False,
            "importable": False,
            "error": "source directory is missing",
        }
    environment = dict(os.environ)
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(root) if not previous else f"{root}{os.pathsep}{previous}"
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return {
            "source_present": True,
            "importable": False,
            "error": "import timed out after 30 seconds",
        }
    error = (completed.stderr or completed.stdout).strip()
    return {
        "source_present": True,
        "importable": completed.returncode == 0,
        "error": error[-2000:] if error else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
