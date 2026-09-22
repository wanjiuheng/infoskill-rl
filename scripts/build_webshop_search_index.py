from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


MINIMUM_FREE_DISK_BYTES = 50 * 1024**3
MINIMUM_FULL_DOCUMENT_COUNT = 100_000


def _document_from_product(product: dict[str, object]) -> dict[str, object]:
    """Match the full-index document contract in upstream WebShop."""
    options = product.get("options", {})
    assert isinstance(options, dict)
    option_text = ", and ".join(
        f"{name}: {', '.join(values)}"
        for name, values in options.items()
    )
    bullet_points = product["BulletPoints"]
    assert isinstance(bullet_points, list) and bullet_points
    return {
        "id": product["asin"],
        "contents": " ".join(
            [
                product["Title"],
                product["Description"],
                bullet_points[0],
                option_text,
            ]
        ).lower(),
        "product": product,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_status(data_root: Path) -> dict[str, dict[str, object]]:
    paths = {
        "products": data_root / "data/items_shuffle.json",
        "attributes": data_root / "data/items_ins_v2.json",
        "human_instructions": data_root / "data/items_human_ins.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing full WebShop {name}: {path}")
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }


def _write_documents(products: list[dict[str, object]], output: Path) -> tuple[int, str, str]:
    count = 0
    digest = hashlib.sha256()
    probe = ""
    with output.open("wb") as destination:
        for product in products:
            document = _document_from_product(product)
            if not probe:
                words = re.findall(r"[a-z0-9]{3,}", str(document["contents"]))
                probe = words[0] if words else ""
            row = (json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8")
            destination.write(row)
            digest.update(row)
            count += 1
            if count % 100_000 == 0:
                print(f"indexed documents prepared: {count}", flush=True)
    if count <= MINIMUM_FULL_DOCUMENT_COUNT or not probe:
        raise RuntimeError(
            "WebShop full product source produced too few searchable documents: "
            f"{count} (requires more than {MINIMUM_FULL_DOCUMENT_COUNT})"
        )
    return count, digest.hexdigest(), probe


def _load_products(webshop_root: Path, sources: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    sys.path.insert(0, str(webshop_root))
    from web_agent_site.engine import engine

    # Upstream load_products otherwise reads this file from its code tree.
    engine.HUMAN_ATTR_PATH = sources["human_instructions"]["path"]
    products, *_ = engine.load_products(
        filepath=sources["products"]["path"],
        attrpath=sources["attributes"]["path"],
        num_products=None,
    )
    return products


def _check_index(index_dir: Path, query: str) -> int:
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))
    try:
        hits = searcher.search(query, k=1)
        if not hits:
            raise RuntimeError(f"full WebShop index returned no hits for {query!r}")
        raw = searcher.doc(hits[0].docid).raw()
        if not json.loads(raw).get("id"):
            raise RuntimeError("full WebShop index did not store raw product documents")
        return len(hits)
    finally:
        close = getattr(searcher, "close", None)
        if callable(close):
            close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", required=True)
    parser.add_argument("--webshop-data-root", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")

    webshop_root = Path(args.webshop_root).expanduser().resolve(strict=True)
    data_root = Path(args.webshop_data_root).expanduser().resolve(strict=True)
    if data_root == webshop_root or webshop_root in data_root.parents:
        raise RuntimeError("WebShop index output must be outside the source tree")
    search_root = data_root / "search_engine"
    resources = search_root / "resources"
    indexes = search_root / "indexes"
    manifest = search_root / "index-manifest.json"
    search_root.mkdir(parents=True, exist_ok=True)
    if os.name != "posix":
        raise RuntimeError("WebShop Lucene index builder requires Linux")
    import fcntl

    lock = (search_root / ".index-build.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another full WebShop index build is running") from error
    if any(path.exists() for path in (resources, indexes, manifest)):
        raise RuntimeError("full WebShop index output already exists; refusing to overwrite")
    unfinished = tuple(search_root.glob(".index-build-*"))
    if unfinished:
        raise RuntimeError(
            "unfinished WebShop index staging exists; inspect it before retrying: "
            + ", ".join(str(path) for path in unfinished)
        )
    if shutil.disk_usage(data_root).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("fewer than 50 GiB free on WebShop data filesystem")

    started = time.monotonic()
    sources = _source_status(data_root)
    stage = search_root / f".index-build-{uuid.uuid4().hex}"
    stage.mkdir()
    resource_stage = stage / "resources"
    index_stage = stage / "indexes"
    resource_stage.mkdir()
    index_stage.mkdir()
    print(f"STAGING={stage}", flush=True)
    print("Loading the complete product corpus with upstream WebShop normalization", flush=True)
    products = _load_products(webshop_root, sources)
    count, documents_sha, probe = _write_documents(
        products, resource_stage / "documents.jsonl"
    )
    del products
    print(f"FULL_PRODUCT_COUNT={count}", flush=True)
    print("Building the single complete Lucene index", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(resource_stage),
            "--index",
            str(index_stage),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(args.threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ],
        check=True,
    )
    index_files = [path for path in index_stage.rglob("*") if path.is_file()]
    if not index_files:
        raise RuntimeError("Pyserini completed without creating index files")
    probe_hits = _check_index(index_stage, probe)
    report = {
        "schema_version": 1,
        "artifact": "full WebShop product Lucene index",
        "webshop_source": str(webshop_root),
        "webshop_data_root": str(data_root),
        "sources": sources,
        "document_count": count,
        "documents_sha256": documents_sha,
        "documents_bytes": (resource_stage / "documents.jsonl").stat().st_size,
        "index_file_count": len(index_files),
        "probe_query": probe,
        "probe_hits": probe_hits,
        "threads": args.threads,
        "pyserini_version": importlib.metadata.version("pyserini"),
        "duration_seconds": time.monotonic() - started,
        "status": "complete",
    }
    (stage / "index-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(resource_stage, resources)
    os.replace(index_stage, indexes)
    os.replace(stage / "index-manifest.json", manifest)
    stage.rmdir()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"INDEX_MANIFEST={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
