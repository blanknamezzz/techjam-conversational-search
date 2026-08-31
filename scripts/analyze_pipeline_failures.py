from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent
from starter.config import AgentConfig


def rank_of(values: list[str], target: str) -> int | None:
    return values.index(target) + 1 if target in values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify failures across retrieval, fusion, and ranking")
    parser.add_argument(
        "--variant",
        choices=("v6", "v9"),
        default="v6",
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--split-ids")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.split_ids:
        ids = {
            line.strip()
            for line in Path(args.split_ids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        samples = [sample for sample in samples if sample["sample_id"] in ids]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog, AgentConfig.for_variant(args.variant))

    rows: list[dict] = []
    counts: Counter[str] = Counter()
    scenario_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        session_id = f"diagnostic_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        hit_turn = None
        turn_rows: list[dict] = []
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, TOP_K)
            diagnostic = agent.diagnostics[session_id][-1]
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            turn_rows.append({
                "turn": turn,
                "query": diagnostic["query"],
                "ask_attribute": diagnostic["ask_attribute"],
                "sparse_rank": rank_of(diagnostic["sparse_ids"], target),
                "dense_rank": rank_of(diagnostic["dense_ids"], target),
                "fused_rank": rank_of(diagnostic["fused_ids"], target),
                "reranked_rank": rank_of(diagnostic["ranked_ids"], target),
                "top10_rank": rank_of(ranked, target),
            })
            if override_applied and target in ranked:
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )

        if hit_turn is not None:
            category = "success"
        elif not any(row["sparse_rank"] or row["dense_rank"] for row in turn_rows):
            category = "retrieval_failure"
        elif not any(row["fused_rank"] for row in turn_rows):
            category = "fusion_failure"
        elif any(row["reranked_rank"] and row["reranked_rank"] > TOP_K for row in turn_rows):
            category = "ranking_failure"
        else:
            category = "timing_or_state_failure"
        counts[category] += 1
        scenario_counts[sample["scenario_type"]][category] += 1
        rows.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "classification": category,
            "hit_turn": hit_turn,
            "turns": turn_rows,
        })

    result = {
        "variant": args.variant,
        "sample_count": len(samples),
        "classification_counts": dict(sorted(counts.items())),
        "scenario_counts": {
            name: dict(sorted(values.items())) for name, values in sorted(scenario_counts.items())
        },
        "sessions": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
