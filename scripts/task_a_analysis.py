from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATALOG_FIELDS = (
    "parent_asin",
    "title",
    "features",
    "description",
    "price",
    "categories",
    "details",
    "average_rating",
    "rating_number",
    "store",
)
TEXT_FIELDS = ("title", "features", "description", "details", "store")
SPLIT_SEED = "techjam-task-a-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(float(percentile(values, 0.95)), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
    }


def analyze_catalog(path: Path) -> dict[str, Any]:
    missing = Counter()
    identifiers: set[str] = set()
    duplicate_identifiers = 0
    leaf_categories: Counter[str] = Counter()
    text_lengths: dict[str, list[float]] = defaultdict(list)
    prices: list[float] = []
    ratings: list[float] = []
    rating_numbers: list[float] = []
    row_count = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            row_count += 1
            identifier = str(product.get("parent_asin", ""))
            if identifier in identifiers:
                duplicate_identifiers += 1
            identifiers.add(identifier)

            for field in CATALOG_FIELDS:
                if is_missing(product.get(field)):
                    missing[field] += 1
            for field in TEXT_FIELDS:
                text_lengths[field].append(float(len(flatten_text(product.get(field)).split())))

            categories = product.get("categories") or []
            if categories:
                leaf_categories[str(categories[-1])] += 1
            if isinstance(product.get("price"), (int, float)):
                prices.append(float(product["price"]))
            if isinstance(product.get("average_rating"), (int, float)):
                ratings.append(float(product["average_rating"]))
            if isinstance(product.get("rating_number"), (int, float)):
                rating_numbers.append(float(product["rating_number"]))

    return {
        "path": str(path),
        "row_count": row_count,
        "unique_parent_asin": len(identifiers),
        "duplicate_parent_asin_rows": duplicate_identifiers,
        "missing": {
            field: {
                "count": missing[field],
                "rate": round(missing[field] / row_count, 6) if row_count else None,
            }
            for field in CATALOG_FIELDS
        },
        "text_word_lengths": {
            field: numeric_summary(values) for field, values in sorted(text_lengths.items())
        },
        "price": numeric_summary(prices),
        "average_rating": numeric_summary(ratings),
        "rating_number": numeric_summary(rating_numbers),
        "top_leaf_categories": leaf_categories.most_common(20),
    }


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{value}".encode()).hexdigest()


def split_samples(samples: list[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, Any]]:
    targets = [str(sample["ground_truth"]["parent_asin"]) for sample in samples]
    target_counts = Counter(targets)
    duplicate_targets = sorted(target for target, count in target_counts.items() if count > 1)
    if duplicate_targets:
        raise ValueError(
            "Target products repeat across public sessions; update the group-stratified splitter before continuing: "
            + ", ".join(duplicate_targets[:10])
        )

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_scenario[str(sample["scenario_type"])].append(sample)

    dev_ids: list[str] = []
    validation_ids: list[str] = []
    split_counts: dict[str, dict[str, int]] = {}
    for scenario, rows in sorted(by_scenario.items()):
        ordered = sorted(rows, key=lambda row: stable_key(str(row["sample_id"])))
        validation_count = round(len(ordered) * 0.20)
        validation = ordered[:validation_count]
        development = ordered[validation_count:]
        validation_ids.extend(str(row["sample_id"]) for row in validation)
        dev_ids.extend(str(row["sample_id"]) for row in development)
        split_counts[scenario] = {"development": len(development), "validation": len(validation)}

    dev_ids.sort()
    validation_ids.sort()
    return dev_ids, validation_ids, {
        "seed": SPLIT_SEED,
        "method": "deterministic SHA-256 ordering, stratified by scenario; targets verified unique",
        "development_count": len(dev_ids),
        "validation_count": len(validation_ids),
        "target_overlap": 0,
        "scenario_counts": split_counts,
    }


def analyze_public_set(path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    samples = load_jsonl(path)
    scenario_counts = Counter(str(sample.get("scenario_type")) for sample in samples)
    difficulty_counts = Counter(str(sample.get("difficulty_bucket")) for sample in samples)
    category_counts = Counter(str(sample.get("category_bucket")) for sample in samples)
    targets = [str(sample["ground_truth"]["parent_asin"]) for sample in samples]
    dev_ids, validation_ids, split = split_samples(samples)
    return {
        "path": str(path),
        "row_count": len(samples),
        "unique_sample_ids": len({str(sample["sample_id"]) for sample in samples}),
        "unique_targets": len(set(targets)),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "split": split,
    }, dev_ids, validation_ids


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Task A data and create deterministic public-set splits")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/task_a_data_summary.json"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    catalog = analyze_catalog(args.catalog)
    public_set, dev_ids, validation_ids = analyze_public_set(args.public_set)
    summary = {"catalog": catalog, "public_set": public_set}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_lines(args.splits_dir / "task_a_dev_ids.txt", dev_ids)
    write_lines(args.splits_dir / "task_a_validation_ids.txt", validation_ids)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
