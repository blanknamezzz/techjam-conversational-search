from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .models import Candidate
from .text_utils import flatten_text, product_search_text, terms


class CatalogStore:
    """Read-only product store plus an in-memory FTS5/Porter index."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, dict] = {}
        self.search_text: dict[str, str] = {}
        self.product_ids: list[str] = []
        self._build()

    def _build(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.products[parent_asin] = product
                self.search_text[parent_asin] = product_search_text(product)
                self.product_ids.append(parent_asin)
                batch.append(
                    (
                        parent_asin,
                        flatten_text(product.get("title")),
                        flatten_text(product.get("categories")),
                        flatten_text(product.get("features")),
                        flatten_text(product.get("details")),
                        flatten_text(product.get("store")),
                        flatten_text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def sparse_search(self, query: str, top_k: int) -> list[Candidate]:
        query_terms = terms(query, expand=True)
        if not query_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in query_terms)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 7.0, 4.5, 3.0, 3.0, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [
            Candidate(
                parent_asin=str(parent_asin),
                ranks={"bm25": rank},
                scores={"bm25": -float(score)},
            )
            for rank, (parent_asin, score) in enumerate(rows, start=1)
        ]

    def product(self, parent_asin: str) -> dict:
        return self.products[parent_asin]
