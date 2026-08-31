from __future__ import annotations

import math
import re

from .catalog import CatalogStore
from .models import Candidate, Constraint, SessionState
from .state_tracker import budget_limit
from .text_utils import COLORS, MATERIALS, normalize_text, terms


MATCH = "MATCH"
VIOLATION = "VIOLATION"
UNKNOWN = "UNKNOWN"


class ConstraintMatcher:
    def __init__(self, catalog: CatalogStore) -> None:
        self.catalog = catalog

    def evaluate(self, parent_asin: str, state: SessionState) -> dict[str, str]:
        product = self.catalog.product(parent_asin)
        text = self.catalog.search_text[parent_asin]
        evidence: dict[str, str] = {}
        for index, constraint in enumerate(state.active_constraints()):
            evidence[f"{constraint.attribute}:{index}"] = self._constraint_status(product, text, constraint)
        if state.category:
            evidence["category"] = self._category_status(product, state.category)
        maximum = budget_limit(state)
        if maximum is not None:
            price = product.get("price")
            if isinstance(price, (int, float)):
                evidence["budget"] = MATCH if float(price) <= maximum else VIOLATION
            else:
                evidence["budget"] = UNKNOWN
        return evidence

    @staticmethod
    def _constraint_status(product: dict, text: str, constraint: Constraint) -> str:
        value = normalize_text(constraint.value)
        query_terms = set(terms(value))
        if constraint.attribute == "color":
            expected = query_terms & COLORS
            known = set(re.findall(r"[a-z0-9]+", text)) & COLORS
            if expected & known:
                return VIOLATION if constraint.negated else MATCH
            if expected and known:
                return MATCH if constraint.negated else VIOLATION
            return UNKNOWN
        if constraint.attribute == "material":
            expected = query_terms & MATERIALS
            known = set(re.findall(r"[a-z0-9]+", text)) & MATERIALS
            if expected & known:
                return VIOLATION if constraint.negated else MATCH
            if expected and known:
                return MATCH if constraint.negated else VIOLATION
            return UNKNOWN
        if constraint.attribute == "budget":
            return UNKNOWN
        if not query_terms:
            return UNKNOWN
        coverage = sum(bool(re.search(rf"\b{re.escape(term)}\b", text)) for term in query_terms) / len(query_terms)
        if value in text or coverage >= 0.6:
            return VIOLATION if constraint.negated else MATCH
        if constraint.negated:
            return MATCH
        return UNKNOWN

    @staticmethod
    def _category_status(product: dict, category: str) -> str:
        category_text = normalize_text(
            f"{product.get('title') or ''} {' '.join(str(x) for x in product.get('categories') or [])}"
        )
        wanted = set(terms(category))
        present = set(terms(category_text))
        if not wanted:
            return UNKNOWN
        return MATCH if wanted & present else UNKNOWN


class Reranker:
    def __init__(
        self,
        catalog: CatalogStore,
        *,
        hard_gate: bool = True,
        exact_constraint_weight: float = 0.0,
    ) -> None:
        self.catalog = catalog
        self.matcher = ConstraintMatcher(catalog)
        self.hard_gate = hard_gate
        self.exact_constraint_weight = exact_constraint_weight

    def rank(self, candidates: list[Candidate], state: SessionState) -> list[Candidate]:
        hard_keys = {
            f"{constraint.attribute}:{index}"
            for index, constraint in enumerate(state.active_constraints())
            if constraint.hard
        }
        if budget_limit(state) is not None:
            hard_keys.add("budget")

        for candidate in candidates:
            product = self.catalog.product(candidate.parent_asin)
            evidence = self.matcher.evaluate(candidate.parent_asin, state)
            candidate.evidence = evidence
            matches = sum(value == MATCH for value in evidence.values())
            violations = sum(value == VIOLATION for value in evidence.values())
            unknown = sum(value == UNKNOWN for value in evidence.values())
            hard_violation = any(evidence.get(key) == VIOLATION for key in hard_keys)

            rrf = candidate.scores.get("rrf", 0.0)
            dense = max(0.0, candidate.scores.get("dense", 0.0))
            lexical_coverage = self._lexical_coverage(candidate.parent_asin, state.query_text())
            exact_matches = self._exact_constraint_matches(candidate.parent_asin, state)
            quality = self._quality_score(product)
            score = (
                75.0 * rrf
                + 1.20 * lexical_coverage
                + 0.45 * dense
                + 0.42 * matches
                - 0.70 * violations
                - 0.03 * unknown
                + 0.08 * quality
                + self.exact_constraint_weight * exact_matches
            )
            candidate.scores["exact_constraint_matches"] = float(exact_matches)
            if self.hard_gate and hard_violation:
                score -= 100.0
                candidate.evidence["hard_gate"] = VIOLATION
            candidate.final_score = score

        return sorted(candidates, key=lambda item: (-item.final_score, item.parent_asin))

    def _exact_constraint_matches(self, parent_asin: str, state: SessionState) -> int:
        text = self.catalog.search_text[parent_asin]
        return sum(
            normalize_text(constraint.value) in text
            for constraint in state.active_constraints()
            if not constraint.negated and len(normalize_text(constraint.value)) >= 3
        )

    def _lexical_coverage(self, parent_asin: str, query: str) -> float:
        query_terms = set(terms(query))
        if not query_terms:
            return 0.0
        text = self.catalog.search_text[parent_asin]
        return sum(bool(re.search(rf"\b{re.escape(term)}\b", text)) for term in query_terms) / len(query_terms)

    @staticmethod
    def _quality_score(product: dict) -> float:
        rating = product.get("average_rating")
        rating_number = product.get("rating_number")
        rating_value = float(rating) / 5.0 if isinstance(rating, (int, float)) else 0.0
        count_value = math.log1p(float(rating_number)) / 14.0 if isinstance(rating_number, (int, float)) else 0.0
        return min(1.0, 0.7 * rating_value + 0.3 * count_value)
