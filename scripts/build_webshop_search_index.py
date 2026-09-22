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


def _catalog_layout(data_root: Path, catalog: str) -> dict[str, Path]:
    if catalog == "full":
        suffix = ""
        product_name = "items_shuffle.json"
        attribute_name = "items_ins_v2.json"
        manifest_name = "index-manifest.json"
    elif catalog == "paper1000":
        suffix = "_1k"
        product_name = "items_shuffle_1000.json"
        attribute_name = "items_ins_v2_1000.json"
        manifest_name = "index-1k-manifest.json"
    else:
        raise ValueError(f"unsupported WebShop catalog: {catalog}")
    search_root = data_root / "search_engine"
    return {
        "products": data_root / "data" / product_name,
        "attributes": data_root / "data" / attribute_name,
        "human_instructions": data_root / "data/items_human_ins.json",
        "resources": search_root / f"resources{suffix}",
        "indexes": search_root / f"indexes{suffix}",
        "manifest": search_root / manifest_name,
    }


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


def _source_status(
    data_root: Path, catalog: str
) -> dict[str, dict[str, object]]:
    layout = _catalog_layout(data_root, catalog)
    paths = {
        name: layout[name]
        for name in ("products", "attributes", "human_instructions")
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {catalog} WebShop {name}: {path}")
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }


def _write_documents(
    products: list[dict[str, object]],
    output: Path,
    *,
    minimum_count: int,
    exact_count: int | None = None,
) -> tuple[int, str, str]:
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
    invalid_count = count <= minimum_count or (
        exact_count is not None and count != exact_count
    )
    if invalid_count or not probe:
        raise RuntimeError(
            "WebShop product source produced an unexpected searchable document count: "
            f"{count}"
        )
    return count, digest.hexdigest(), probe


def _load_products(
    webshop_root: Path,
    sources: dict[str, dict[str, object]],
    *,
    catalog: str,
) -> list[dict[str, object]]:
    sys.path.insert(0, str(webshop_root))
    from web_agent_site.engine import engine

    # Upstream load_products otherwise reads this file from its code tree.
    engine.HUMAN_ATTR_PATH = sources["human_instructions"]["path"]
    products, *_ = engine.load_products(
        filepath=sources["products"]["path"],
        attrpath=sources["attributes"]["path"],
        num_products=1000 if catalog == "paper1000" else None,
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
    parser.add_argument(
        "--catalog",
        choices=("full", "paper1000"),
        default="full",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32")

    webshop_root = Path(args.webshop_root).expanduser().resolve(strict=True)
    data_root = Path(args.webshop_data_root).expanduser().resolve(strict=True)
    if data_root == webshop_root or webshop_root in data_root.parents:
        raise RuntimeError("WebShop index output must be outside the source tree")
    layout = _catalog_layout(data_root, args.catalog)
    search_root = data_root / "search_engine"
    resources = layout["resources"]
    indexes = layout["indexes"]
    manifest = layout["manifest"]
    search_root.mkdir(parents=True, exist_ok=True)
    if os.name != "posix":
        raise RuntimeError("WebShop Lucene index builder requires Linux")
    import fcntl

    lock = (search_root / ".index-build.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another WebShop index build is running") from error
    if any(path.exists() for path in (resources, indexes, manifest)):
        raise RuntimeError("full WebShop index output already exists; refusing to overwrite")
    unfinished = tuple(search_root.glob(".index-build-*"))
    if unfinished:
        raise RuntimeError(
            "unfinished WebShop index staging exists; inspect it before retrying: "
            + ", ".join(str(path) for path in unfinished)
        )
    minimum_free = (
        MINIMUM_FREE_DISK_BYTES if args.catalog == "full" else 1024**3
    )
    if shutil.disk_usage(data_root).free < minimum_free:
        raise RuntimeError("insufficient free space on WebShop data filesystem")

    started = time.monotonic()
    sources = _source_status(data_root, args.catalog)
    stage = search_root / f".index-build-{args.catalog}-{uuid.uuid4().hex}"
    stage.mkdir()
    resource_stage = stage / "resources"
    index_stage = stage / "indexes"
    resource_stage.mkdir()
    index_stage.mkdir()
    print(f"STAGING={stage}", flush=True)
    print(
        f"Loading the {args.catalog} product corpus with upstream WebShop normalization",
        flush=True,
    )
    products = _load_products(webshop_root, sources, catalog=args.catalog)
    count, documents_sha, probe = _write_documents(
        products,
        resource_stage / "documents.jsonl",
        minimum_count=(MINIMUM_FULL_DOCUMENT_COUNT if args.catalog == "full" else 0),
        exact_count=(1000 if args.catalog == "paper1000" else None),
    )
    del products
    print(f"PRODUCT_COUNT={count}", flush=True)
    print(f"Building the {args.catalog} Lucene index", flush=True)
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
        "artifact": f"{args.catalog} WebShop product Lucene index",
        "catalog": args.catalog,
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
