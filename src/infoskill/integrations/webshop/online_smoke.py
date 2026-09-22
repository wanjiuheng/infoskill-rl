"""CPU-only WebShop environment smoke against external, full-corpus assets.

This is a diagnostic entrypoint, not the formal rollout or evaluation runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .demonstrations import normalize_available_actions, normalize_goal
from .policy import render_webshop_policy_message
from .splits import WebShopSplit, split_for_goal_index


def map_runtime_goals(
    official_goals: Sequence[str],
    runtime_goals: Sequence[Mapping[str, object]],
) -> tuple[dict[int, int], tuple[str, ...]]:
    """Map runtime positions to the first matching official goal index."""
    lookup: dict[str, int] = {}
    for index, goal in enumerate(official_goals):
        lookup.setdefault(normalize_goal(goal), index)
    positions: dict[int, int] = {}
    unknown: list[str] = []
    for position, goal in enumerate(runtime_goals):
        text = goal.get("instruction_text")
        normalized = normalize_goal(text)
        official_index = lookup.get(normalized)
        if official_index is None:
            unknown.append(normalized)
        else:
            positions.setdefault(official_index, position)
    return positions, tuple(unknown)


def _required_assets(source_root: Path, data_root: Path) -> dict[str, Path]:
    paths = {
        "products": data_root / "data/items_shuffle.json",
        "attributes": data_root / "data/items_ins_v2.json",
        "human_instructions": data_root / "data/items_human_ins.json",
        "human_goals": data_root / "baseline_models/data/human_goals.json",
        "index": data_root / "search_engine/indexes",
        "index_manifest": data_root / "search_engine/index-manifest.json",
    }
    if not (source_root / "web_agent_site").is_dir():
        raise FileNotFoundError(f"WebShop source is missing: {source_root}")
    for name, path in paths.items():
        if not (path.is_dir() if name == "index" else path.is_file()):
            raise FileNotFoundError(f"WebShop {name} is missing: {path}")
    manifest = json.loads(paths["index_manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("document_count", 0) <= 100_000:
        raise RuntimeError("external full WebShop index manifest is not complete")
    for name in ("products", "attributes", "human_instructions"):
        expected = manifest.get("sources", {}).get(name, {})
        if (
            Path(str(expected.get("path", ""))).resolve() != paths[name].resolve()
            or expected.get("bytes") != paths[name].stat().st_size
        ):
            raise RuntimeError(f"external WebShop {name} differs from index source")
    return paths


def _open_external_text_env(source_root: Path, paths: Mapping[str, Path]) -> object:
    """Bind upstream constructor to external assets only while it initializes."""
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    import web_agent_site.engine.engine as upstream_engine
    import web_agent_site.envs.web_agent_text_env as text_environment
    from pyserini.search.lucene import LuceneSearcher

    old_human = upstream_engine.HUMAN_ATTR_PATH
    old_search = text_environment.init_search_engine

    def external_search_engine(num_products: int | None = None) -> object:
        if num_products is not None:
            raise ValueError("online WebShop smoke requires the complete product index")
        return LuceneSearcher(str(paths["index"]))

    try:
        upstream_engine.HUMAN_ATTR_PATH = str(paths["human_instructions"])
        text_environment.init_search_engine = external_search_engine
        return text_environment.WebAgentTextEnv(
            observation_mode="text",
            file_path=str(paths["products"]),
            attr_path=str(paths["attributes"]),
            human_goals=True,
            num_products=None,
            seed=42,
        )
    finally:
        upstream_engine.HUMAN_ATTR_PATH = old_human
        text_environment.init_search_engine = old_search


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _page_goal_text(instruction: str) -> str:
    """Remove the upstream HTML heading, keeping the actual shopping goal."""
    text = instruction.strip()
    if text.lower().startswith("instruction:"):
        text = text[len("instruction:") :].strip()
    if not text:
        raise RuntimeError("WebShop reset returned an empty goal")
    return text


def run_smoke(source_root: Path, data_root: Path) -> dict[str, object]:
    paths = _required_assets(source_root, data_root)
    official = json.loads(paths["human_goals"].read_text(encoding="utf-8"))
    if not isinstance(official, list) or len(official) <= 1500:
        raise RuntimeError("official WebShop human_goals.json is incomplete")
    raw_environment = _open_external_text_env(source_root, paths)
    try:
        runtime_goals = raw_environment.server.goals
        positions, unknown = map_runtime_goals(official, runtime_goals)
        if unknown:
            raise RuntimeError(
                f"{len(unknown)} runtime WebShop goals have no official goal index; "
                f"first_sha256={_digest(unknown[0])}"
            )
        examples: dict[str, dict[str, object]] = {}
        for split in WebShopSplit:
            candidates = (
                index
                for index in sorted(positions)
                if split_for_goal_index(index, goal_count=len(official)) is split
            )
            official_index = next(candidates, None)
            if official_index is None:
                raise RuntimeError(f"runtime WebShop goals contain no {split.value} task")
            runtime_position = positions[official_index]
            observation, _ = raw_environment.reset(session=runtime_position)
            instruction = _page_goal_text(raw_environment.instruction_text)
            if normalize_goal(instruction) != normalize_goal(official[official_index]):
                raise RuntimeError(f"WebShop {split.value} reset returned a different goal")
            actions = normalize_available_actions(raw_environment.get_available_actions())
            if "search[<your query>]" not in actions:
                raise RuntimeError(f"WebShop {split.value} search action is missing")
            prompt = render_webshop_policy_message(
                task_description=instruction,
                current_observation=observation,
                available_actions=actions,
            )
            query = runtime_goals[runtime_position]["query"]
            if not isinstance(query, str) or not query.strip():
                raise RuntimeError(f"WebShop {split.value} goal has no search query")
            search_observation, reward, done, _ = raw_environment.step(
                f"search[{query}]"
            )
            if not isinstance(search_observation, str) or not search_observation.strip():
                raise RuntimeError(f"WebShop {split.value} search returned no observation")
            post_actions = normalize_available_actions(raw_environment.get_available_actions())
            if done or not post_actions:
                raise RuntimeError(f"WebShop {split.value} search did not continue")
            examples[split.value] = {
                "official_goal_index": official_index,
                "runtime_goal_position": runtime_position,
                "prompt_sha256": _digest(prompt),
                "initial_observation_sha256": _digest(observation),
                "search_observation_sha256": _digest(search_observation),
                "initial_action_count": len(actions),
                "post_search_action_count": len(post_actions),
                "search_reward": float(reward),
            }
        return {
            "schema_version": 1,
            "status": "passed",
            "scope": "CPU environment path, official split, search transition, prompt render",
            "not_proven": ["warmstart-online prompt parity", "policy rollout", "500-task evaluation"],
            "source_root": str(source_root),
            "data_root": str(data_root),
            "index_manifest": str(paths["index_manifest"]),
            "official_goal_count": len(official),
            "runtime_goal_count": len(runtime_goals),
            "matched_unique_official_goals": len(positions),
            "unmatched_runtime_goals": len(unknown),
            "examples": examples,
        }
    finally:
        raw_environment.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", required=True)
    parser.add_argument("--webshop-data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    source_root = Path(args.webshop_root).expanduser().resolve(strict=True)
    data_root = Path(args.webshop_data_root).expanduser().resolve(strict=True)
    report = run_smoke(source_root, data_root)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"ONLINE_SMOKE_REPORT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
