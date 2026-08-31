from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Constraint:
    """One normalized requirement extracted from the conversation."""

    attribute: str
    value: str
    source_turn: int
    hard: bool = False
    negated: bool = False
    confidence: float = 1.0


@dataclass
class SessionState:
    session_id: str
    user_profile: dict
    intent: str = "browsing"
    category: str | None = None
    constraints: dict[str, list[Constraint]] = field(default_factory=dict)
    free_text: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference: set[str] = field(default_factory=set)
    override_count: int = 0

    def add_constraint(self, constraint: Constraint, *, replace: bool = False) -> None:
        values = self.constraints.setdefault(constraint.attribute, [])
        if replace:
            values.clear()
        key = (constraint.value.casefold(), constraint.negated)
        if any((item.value.casefold(), item.negated) == key for item in values):
            return
        values.append(constraint)

    def active_constraints(self) -> list[Constraint]:
        return [item for values in self.constraints.values() for item in values]

    def query_text(self) -> str:
        parts: list[str] = []
        if self.category:
            parts.append(self.category)
        parts.extend(item.value for item in self.active_constraints() if not item.negated)
        parts.extend(self.free_text[-2:])
        return " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))


@dataclass
class Candidate:
    parent_asin: str
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    final_score: float = 0.0
