from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import Candidate


class DenseRetriever:
    """Optional local Sentence Transformer retriever with a silent offline fallback."""

    def __init__(
        self,
        model_path: str,
        index_path: str,
        ids_path: str,
        expected_ids: list[str] | None = None,
    ) -> None:
        self.available = False
        self.model = None
        self.embeddings: np.ndarray | None = None
        self.product_ids: list[str] = []

        model_dir = Path(model_path)
        embedding_file = Path(index_path)
        id_file = Path(ids_path)
        if not model_dir.exists() or not embedding_file.exists() or not id_file.exists():
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(str(model_dir), local_files_only=True)
            self.embeddings = np.load(embedding_file, mmap_mode="r")
            self.product_ids = [str(item) for item in json.loads(id_file.read_text(encoding="utf-8"))]
            ids_match = expected_ids is None or self.product_ids == expected_ids
            self.available = (
                self.embeddings.ndim == 2
                and len(self.product_ids) == len(self.embeddings)
                and ids_match
            )
        except Exception:
            self.available = False
            self.model = None
            self.embeddings = None
            self.product_ids = []

    def search(self, query: str, top_k: int) -> list[Candidate]:
        if not self.available or self.model is None or self.embeddings is None or not query.strip():
            return []
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        scores = np.asarray(self.embeddings @ vector)
        limit = min(top_k, len(scores))
        if limit <= 0:
            return []
        indices = np.argpartition(scores, -limit)[-limit:]
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        return [
            Candidate(
                parent_asin=self.product_ids[int(index)],
                ranks={"dense": rank},
                scores={"dense": float(scores[int(index)])},
            )
            for rank, index in enumerate(indices, start=1)
        ]
