from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from infoskill.episode import TaskSpec

from .environment import WebShopEnvironment
from .paper_protocol import load_paper128_manifest, paper128_tasks


RawEnvironmentFactory = Callable[[str, int, int], object]


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _paper_source_files(data_root: Path) -> dict[str, Path]:
    return {
        "products": data_root / "data/items_shuffle_1000.json",
        "attributes": data_root / "data/items_ins_v2_1000.json",
        "human_instructions": data_root / "data/items_human_ins.json",
    }


def load_bound_paper128_manifest(
    *,
    webshop_source: str | Path,
    webshop_data_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    source = Path(webshop_source).expanduser().resolve(strict=True)
    data_root = Path(webshop_data_root).expanduser().resolve(strict=True)
    manifest = load_paper128_manifest(
        Path(manifest_path),
        source_files=_paper_source_files(data_root),
        environment_commit=_git_head(source),
    )
    # Fail before a model runtime reserves GPU memory.  A frozen task list is
    # insufficient if the search backend was built from a different catalog.
    _required_index(data_root, manifest)
    return manifest


def _required_index(data_root: Path, manifest: Mapping[str, Any]) -> Path:
    index = data_root / "search_engine/indexes_1k"
    report_path = data_root / "search_engine/index-1k-manifest.json"
    if not index.is_dir() or not report_path.is_file():
        raise FileNotFoundError(
            "paper128 requires search_engine/indexes_1k and index-1k-manifest.json"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise RuntimeError("paper128 WebShop index is incomplete")
    if report.get("catalog") != "paper1000" or report.get("document_count") != 1000:
        raise RuntimeError("paper128 WebShop index does not contain exactly 1000 products")
    recorded = manifest["source_files"]
    indexed = report.get("sources")
    if not isinstance(recorded, Mapping) or not isinstance(indexed, Mapping):
        raise RuntimeError("paper128 WebShop source identities are incomplete")
    for name in ("products", "attributes", "human_instructions"):
        expected = recorded[name]
        actual = indexed.get(name)
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise RuntimeError(f"paper128 WebShop index source is missing: {name}")
        if (
            actual.get("sha256") != expected.get("sha256")
            or actual.get("bytes") != expected.get("bytes")
        ):
            raise RuntimeError(f"paper128 WebShop index source differs: {name}")
    return index


def _open_shared_server(
    source_root: Path,
    data_root: Path,
    index: Path,
) -> tuple[object, type]:
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    import web_agent_site.engine.engine as upstream_engine
    import web_agent_site.envs.web_agent_text_env as text_environment
    from pyserini.search.lucene import LuceneSearcher

    files = _paper_source_files(data_root)
    old_human = upstream_engine.HUMAN_ATTR_PATH
    old_search = text_environment.init_search_engine

    def external_search_engine(num_products: int | None = None) -> object:
        if num_products != 1000:
            raise ValueError("paper128 WebShop runtime requires num_products=1000")
        return LuceneSearcher(str(index))

    try:
        upstream_engine.HUMAN_ATTR_PATH = str(files["human_instructions"])
        text_environment.init_search_engine = external_search_engine
        bootstrap = text_environment.WebAgentTextEnv(
            observation_mode="text",
            file_path=str(files["products"]),
            attr_path=str(files["attributes"]),
            human_goals=False,
            num_products=1000,
            seed=0,
        )
        return bootstrap.server, text_environment.WebAgentTextEnv
    finally:
        upstream_engine.HUMAN_ATTR_PATH = old_human
        text_environment.init_search_engine = old_search


class WebShopEnvironmentFactory:
    """Create isolated browser sessions over one immutable small-catalog server."""

    def __init__(
        self,
        *,
        task_sessions: Mapping[str, tuple[TaskSpec, int]],
        raw_environment_factory: RawEnvironmentFactory,
    ) -> None:
        self._task_sessions = dict(task_sessions)
        self._raw_environment_factory = raw_environment_factory

    @classmethod
    def from_paths(
        cls,
        *,
        webshop_source: str | Path,
        webshop_data_root: str | Path,
        manifest_path: str | Path,
        max_steps: int,
    ) -> "WebShopEnvironmentFactory":
        if max_steps != 15:
            raise ValueError("paper128 WebShop evaluation requires max_steps=15")
        source = Path(webshop_source).expanduser().resolve(strict=True)
        data_root = Path(webshop_data_root).expanduser().resolve(strict=True)
        manifest = load_bound_paper128_manifest(
            webshop_source=source,
            webshop_data_root=data_root,
            manifest_path=manifest_path,
        )
        index = _required_index(data_root, manifest)
        server, environment_class = _open_shared_server(source, data_root, index)
        tasks = paper128_tasks(manifest)
        task_sessions = {
            task.task_id: (task, int(item["session_index"]))
            for task, item in zip(tasks, manifest["tasks"], strict=True)
        }
        goals = getattr(server, "goals")
        for item in manifest["tasks"]:
            goals[int(item["session_index"])] = dict(item["goal"])

        def create_raw(task_id: str, rollout_id: int, seed: int) -> object:
            return environment_class(
                observation_mode="text",
                server=server,
                seed=seed,
                session_prefix=f"infoskill-{task_id}-r{rollout_id}-",
            )

        return cls(
            task_sessions=task_sessions,
            raw_environment_factory=create_raw,
        )

    def create(
        self,
        task: TaskSpec,
        *,
        rollout_id: int,
        seed: int,
    ) -> WebShopEnvironment:
        frozen = self._task_sessions.get(task.task_id)
        if frozen is None:
            raise ValueError("task is not part of the frozen paper128 manifest")
        expected, session_index = frozen
        if task != expected:
            raise ValueError("task differs from the frozen paper128 manifest")
        raw = self._raw_environment_factory(task.task_id, rollout_id, seed)
        return WebShopEnvironment(raw, task=task, session_index=session_index)
