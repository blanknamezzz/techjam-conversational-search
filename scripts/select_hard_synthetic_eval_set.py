from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SCENARIO_COUNTS = {
    "tune": {"buying": 120, "browsing": 120, "intent_override": 45, "boundary": 15},
    "holdout": {"buying": 40, "browsing": 40, "intent_override": 15, "boundary": 5},
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difficulty_key(session: dict) -> tuple[int, int, float]:
    return (
        int(not bool(session["hit"])),
        int(session["first_hit_turn"] or 11),
        -float(session["reciprocal_rank"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a scenario-balanced hard tune/holdout set from V6 candidate results"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/synthetic/synthetic_candidates.jsonl"),
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    candidates = {row["sample_id"]: row for row in load_jsonl(args.candidates)}
    results = json.loads(args.results.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in results["sessions"]:
        grouped[str(session["scenario_type"])].append(session)
    for values in grouped.values():
        values.sort(key=difficulty_key, reverse=True)

    selected: dict[str, list[dict]] = {"tune": [], "holdout": []}
    selected_metrics: dict[str, dict] = {}
    for scenario in sorted(SCENARIO_COUNTS["tune"]):
        tune_needed = SCENARIO_COUNTS["tune"][scenario]
        holdout_needed = SCENARIO_COUNTS["holdout"][scenario]
        total_needed = tune_needed + holdout_needed
        hardest = grouped[scenario][:total_needed]
        if len(hardest) < total_needed:
            raise RuntimeError(f"Not enough {scenario} candidates")

        holdout_positions = {
            round(index * (total_needed - 1) / max(1, holdout_needed - 1))
            for index in range(holdout_needed)
        }
        holdout_sessions = [item for index, item in enumerate(hardest) if index in holdout_positions]
        tune_sessions = [item for index, item in enumerate(hardest) if index not in holdout_positions]
        if len(holdout_sessions) != holdout_needed or len(tune_sessions) != tune_needed:
            raise RuntimeError(f"Unexpected split size for {scenario}")
        for split, sessions in (("tune", tune_sessions), ("holdout", holdout_sessions)):
            for session in sessions:
                selected[split].append(candidates[str(session["sample_id"])])
                selected_metrics[str(session["sample_id"])] = session

    for split in selected:
        selected[split].sort(key=lambda row: str(row["sample_id"]))
        for index, row in enumerate(selected[split], start=1):
            row["sample_id"] = f"synthetic_{split}_{index:04d}"

    tune_path = args.output_dir / "synthetic_tune.jsonl"
    holdout_path = args.output_dir / "synthetic_holdout.jsonl"
    write_jsonl(tune_path, selected["tune"])
    write_jsonl(holdout_path, selected["holdout"])
    tune_ids = {row["ground_truth"]["parent_asin"] for row in selected["tune"]}
    holdout_ids = {row["ground_truth"]["parent_asin"] for row in selected["holdout"]}
    metadata = {
        "source_candidates": str(args.candidates),
        "source_results": str(args.results),
        "selection": "scenario-balanced hardest V6 sessions by miss, MTTC, then reciprocal rank",
        "tune_count": len(selected["tune"]),
        "holdout_count": len(selected["holdout"]),
        "tune_scenarios": dict(sorted(Counter(row["scenario_type"] for row in selected["tune"]).items())),
        "holdout_scenarios": dict(sorted(Counter(row["scenario_type"] for row in selected["holdout"]).items())),
        "target_overlap_between_splits": len(tune_ids & holdout_ids),
        "tune_sha256": sha256(tune_path),
        "holdout_sha256": sha256(holdout_path),
        "git_policy": "data/synthetic/ is ignored and must never be committed",
    }
    (args.output_dir / "hard_split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
