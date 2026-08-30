from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from starter.catalog import CatalogStore
from starter.models import Candidate, Constraint, SessionState
from starter.reranker import MATCH, VIOLATION, Reranker


class RerankerTest(unittest.TestCase):
    def test_explicit_budget_and_color_violation_is_gated(self) -> None:
        products = [
            {
                "parent_asin": "A", "title": "Black leather boots", "categories": ["Boots"],
                "features": [], "details": {}, "store": "X", "description": [], "price": 80.0,
                "average_rating": 4.0, "rating_number": 10,
            },
            {
                "parent_asin": "B", "title": "Brown nylon boots", "categories": ["Boots"],
                "features": [], "details": {}, "store": "X", "description": [], "price": 120.0,
                "average_rating": 5.0, "rating_number": 1000,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in products), encoding="utf-8")
            catalog = CatalogStore(path)
            state = SessionState("s", {}, category="boots")
            state.add_constraint(Constraint("color", "black", 1, hard=True))
            state.add_constraint(Constraint("budget", "budget under $100", 1, hard=True))
            candidates = [
                Candidate("A", scores={"rrf": 0.01}),
                Candidate("B", scores={"rrf": 0.01}),
            ]
            ranked = Reranker(catalog, hard_gate=True).rank(candidates, state)
        self.assertEqual(ranked[0].parent_asin, "A")
        self.assertEqual(ranked[0].evidence["color:0"], MATCH)
        self.assertEqual(ranked[1].evidence["hard_gate"], VIOLATION)

    def test_exact_constraint_phrase_is_recorded_as_ranking_evidence(self) -> None:
        product = {
            "parent_asin": "A", "title": "Trail boot", "categories": ["Boots"],
            "features": ["waterproof sealed zipper"], "details": {}, "store": "X",
            "description": [], "price": 80.0, "average_rating": 4.0, "rating_number": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps(product) + "\n", encoding="utf-8")
            catalog = CatalogStore(path)
            state = SessionState("s", {}, category="boots")
            state.add_constraint(Constraint("feature", "waterproof sealed zipper", 2))
            candidate = Candidate("A", scores={"rrf": 0.01})
            ranked = Reranker(
                catalog,
                hard_gate=True,
                exact_constraint_weight=0.70,
            ).rank([candidate], state)
        self.assertEqual(ranked[0].scores["exact_constraint_matches"], 1.0)
