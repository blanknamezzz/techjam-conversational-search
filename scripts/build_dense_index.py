from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.text_utils import product_embedding_text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the offline product embedding matrix")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-output", default="artifacts/models/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="artifacts/dense")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    catalog_path = Path(args.catalog)
    output_dir = Path(args.output_dir)
    model_output = Path(args.model_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    product_ids: list[str] = []
    texts: list[str] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            product_ids.append(str(product["parent_asin"]))
            texts.append(product_embedding_text(product))
            if args.limit is not None and len(product_ids) >= args.limit:
                break

    started = time.perf_counter()
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    model.save_pretrained(model_output)
    np.save(output_dir / "product_embeddings.npy", embeddings)
    (output_dir / "product_ids.json").write_text(
        json.dumps(product_ids, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "model_source": args.model,
        "model_path": str(model_output),
        "product_count": len(product_ids),
        "dimension": int(embeddings.shape[1]),
        "normalized": True,
        "build_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
