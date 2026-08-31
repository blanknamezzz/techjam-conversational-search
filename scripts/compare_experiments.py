from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluator result files")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output")
    args = parser.parse_args()

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    reference_sessions = {row["sample_id"]: row for row in reference["sessions"]}
    candidate_sessions = {row["sample_id"]: row for row in candidate["sessions"]}
    if reference_sessions.keys() != candidate_sessions.keys():
        raise ValueError("experiment files do not contain the same sample IDs")

    transitions: Counter[str] = Counter()
    scenarios: dict[str, Counter[str]] = {}
    failures: list[dict] = []
    for sample_id, before in reference_sessions.items():
        after = candidate_sessions[sample_id]
        if not before["hit"] and after["hit"]:
            transition = "miss_to_hit"
        elif before["hit"] and not after["hit"]:
            transition = "hit_to_miss"
        elif not before["hit"] and not after["hit"]:
            transition = "both_miss"
        elif after["best_rank"] < before["best_rank"]:
            transition = "rank_improved"
        elif after["best_rank"] > before["best_rank"]:
            transition = "rank_worsened"
        else:
            transition = "same_rank"
        transitions[transition] += 1
        scenarios.setdefault(after["scenario_type"], Counter())[transition] += 1
        if not after["hit"]:
            failures.append({
                "sample_id": sample_id,
                "scenario_type": after["scenario_type"],
                "reference_hit": before["hit"],
            })

    output = {
        "reference": args.reference,
        "candidate": args.candidate,
        "metric_delta": {
            key: round(float(candidate[key]) - float(reference[key]), 6)
            for key in ("hit_rate_at_10", "mrr", "efficiency", "recommended_technical_score")
        } | {"mttc": round(float(candidate["mttc"]) - float(reference["mttc"]), 6)},
        "transitions": dict(sorted(transitions.items())),
        "scenario_transitions": {
            name: dict(sorted(counter.items())) for name, counter in sorted(scenarios.items())
        },
        "remaining_failures": failures,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
