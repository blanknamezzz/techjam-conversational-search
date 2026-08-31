from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


GENERAL_CATEGORIES = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
    "women",
    "men",
    "girls",
    "boys",
}
SCENARIO_RATIOS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}
TITLE_STOPWORDS = {
    "the", "and", "for", "with", "women", "womens", "men", "mens",
    "girls", "boys", "from", "this", "that", "size", "pack", "set",
    "black", "white", "blue", "red",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def category_bucket(product: dict) -> str:
    values: list[str] = []
    for raw in product.get("categories") or []:
        values.extend(part.strip() for part in str(raw).split(",") if part.strip())
    specific = [value for value in values if value.casefold() not in GENERAL_CATEGORIES]
    value = specific[-1] if specific else (values[-1] if values else "other")
    return re.sub(r"\s+", " ", value).strip()[:80] or "other"


def is_eligible(product: dict, excluded_ids: set[str]) -> bool:
    parent_asin = str(product.get("parent_asin") or "")
    if not parent_asin or parent_asin in excluded_ids:
        return False
    if not str(product.get("title") or "").strip():
        return False
    if not product.get("categories"):
        return False
    evidence = [
        product.get("features"),
        product.get("details"),
        product.get("description"),
        product.get("price"),
    ]
    return sum(value not in (None, "", [], {}) for value in evidence) >= 2


def title_terms(product: dict) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", str(product.get("title") or "").casefold())
        if len(token) > 2 and token not in TITLE_STOPWORDS
    )


def ambiguity_scores(products: list[dict]) -> dict[str, float]:
    exact_counts: Counter[str] = Counter()
    signature_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    category_counts: Counter[str] = Counter()
    records: list[tuple[str, str, tuple[str, ...], str]] = []
    for product in products:
        parent_asin = str(product["parent_asin"])
        bucket = category_bucket(product).casefold()
        tokens = title_terms(product)
        exact = " ".join(tokens)
        signature = tuple(sorted(set(tokens))[:3])
        exact_counts[exact] += 1
        signature_counts[(bucket, signature)] += 1
        category_counts[bucket] += 1
        records.append((parent_asin, bucket, signature, exact))

    scores: dict[str, float] = {}
    for parent_asin, bucket, signature, exact in records:
        exact_size = exact_counts[exact]
        signature_size = signature_counts[(bucket, signature)]
        category_size = category_counts[bucket]
        scores[parent_asin] = (
            6.0 * min(exact_size - 1, 4)
            + 2.0 * min(signature_size - 1, 12)
            + min(category_size / 250.0, 5.0)
        )
    return scores


def balanced_targets(
    products: list[dict],
    count: int,
    rng: random.Random,
    per_bucket_limit: int,
) -> list[dict]:
    scores = ambiguity_scores(products)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        grouped[category_bucket(product)].append(product)
    for values in grouped.values():
        rng.shuffle(values)
        values.sort(key=lambda product: scores[str(product["parent_asin"])])

    buckets = sorted(grouped)
    rng.shuffle(buckets)
    selected: list[dict] = []
    used: Counter[str] = Counter()
    while len(selected) < count:
        progressed = False
        for bucket in buckets:
            if len(selected) >= count:
                break
            if used[bucket] >= per_bucket_limit or not grouped[bucket]:
                continue
            selected.append(grouped[bucket].pop())
            used[bucket] += 1
            progressed = True
        if not progressed:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only selected {len(selected)} of {count} requested targets")
    rng.shuffle(selected)
    return selected


def scenario_sequence(count: int, rng: random.Random) -> list[str]:
    counts = {
        name: round(count * ratio)
        for name, ratio in SCENARIO_RATIOS.items()
    }
    difference = count - sum(counts.values())
    counts["browsing"] += difference
    values = [name for name, amount in counts.items() for _ in range(amount)]
    rng.shuffle(values)
    return values


def safe_profile(product: dict, index: int) -> dict:
    rating = product.get("average_rating")
    average_rating = float(rating) if isinstance(rating, (int, float)) else 4.0
    title_terms = [
        token.casefold()
        for token in re.findall(r"[A-Za-z]{4,}", str(product.get("title") or ""))
    ]
    tags = list(dict.fromkeys(title_terms))[:3] or ["quality", "comfort"]
    frequencies = ("occasional", "monthly", "frequent")
    styles = ("balanced", "quality-focused", "value-focused")
    return {
        "average_prior_rating": round(max(1.0, min(5.0, average_rating)), 2),
        "preference_tags": tags,
        "purchase_frequency": frequencies[index % len(frequencies)],
        "rating_style": styles[index % len(styles)],
        "summary": "Synthetic aggregate profile for local robustness evaluation only.",
    }


def rows_for_split(products: list[dict], split: str, rng: random.Random) -> list[dict]:
    scenarios = scenario_sequence(len(products), rng)
    rows: list[dict] = []
    for index, (product, scenario) in enumerate(zip(products, scenarios), start=1):
        parent_asin = str(product["parent_asin"])
        rows.append({
            "category_bucket": category_bucket(product),
            "difficulty_bucket": "synthetic_unseen_target",
            "ground_truth": {"parent_asin": parent_asin},
            "sample_id": f"synthetic_{split}_{index:04d}",
            "scenario_type": scenario,
            "user_profile": safe_profile(product, index),
        })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local-only synthetic evaluation sessions from unseen catalog targets"
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--candidate-count", type=int, default=1500)
    parser.add_argument("--seed", default="techjam-local-synthetic-hard-v2")
    args = parser.parse_args()

    if args.candidate_count <= 0:
        raise ValueError("candidate count must be positive")
    public_rows = load_jsonl(args.public_set)
    excluded_ids = {str(row["ground_truth"]["parent_asin"]) for row in public_rows}
    products: list[dict] = []
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            if is_eligible(product, excluded_ids):
                products.append(product)

    rng = random.Random(args.seed)
    selected = balanced_targets(products, args.candidate_count, rng, per_bucket_limit=30)
    candidate_rows = rows_for_split(selected, "candidate", rng)

    candidate_path = args.output_dir / "synthetic_candidates.jsonl"
    write_jsonl(candidate_path, candidate_rows)
    metadata = {
        "seed": args.seed,
        "catalog": str(args.catalog),
        "public_target_exclusions": len(excluded_ids),
        "eligible_products": len(products),
        "candidate_count": len(candidate_rows),
        "target_overlap_with_public": len(
            ({row["ground_truth"]["parent_asin"] for row in candidate_rows})
            & excluded_ids
        ),
        "sampling_strategy": "category-balanced high title/signature ambiguity",
        "candidate_scenarios": dict(sorted(Counter(row["scenario_type"] for row in candidate_rows).items())),
        "candidate_sha256": sha256(candidate_path),
        "git_policy": "data/synthetic/ is ignored and must never be committed",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
