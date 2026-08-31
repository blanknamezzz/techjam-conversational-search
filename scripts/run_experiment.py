from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from starter.baseline_agent import BaselineAgent
from starter.config import AgentConfig


def load_split_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible Track 4 experiment")
    parser.add_argument("--variant", choices=("baseline", "v6", "v9"), required=True)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--split-ids")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    split_ids = load_split_ids(Path(args.split_ids) if args.split_ids else None)
    if split_ids is not None:
        samples = [sample for sample in samples if str(sample["sample_id"]) in split_ids]
    catalog_ids, categories, products = catalog_index(args.catalog)

    started = time.perf_counter()
    if args.variant == "baseline":
        agent = BaselineAgent(args.catalog)
    else:
        agent = Agent(args.catalog, AgentConfig.for_variant(args.variant))
    result = evaluate(agent, samples, catalog_ids, categories, products)
    result["experiment"] = {
        "variant": args.variant,
        "split_ids": args.split_ids,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "dense_available": bool(getattr(getattr(agent, "dense", None), "available", False)),
        "llm_available": bool(getattr(getattr(agent, "llm", None), "available", False)),
        "llm_metrics": agent.llm.metrics() if getattr(agent, "llm", None) is not None else None,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
