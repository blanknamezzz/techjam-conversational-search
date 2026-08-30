from __future__ import annotations

import re

from .models import Constraint, SessionState
from .text_utils import COLORS, MATERIALS, classify_constraint, normalize_text


CATEGORY_RE = re.compile(r"\blooking for\s+(.+?)(?=\.\s|,\s*but\b|;|$)", re.I)
PRICE_MAX_RE = re.compile(
    r"(?:under|below|less than|no more than|cannot exceed|up to)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.I,
)
PRICE_AROUND_RE = re.compile(r"(?:budget around|around)\s*\$\s*(\d+(?:\.\d+)?)", re.I)
SIZE_RE = re.compile(r"\b(?:size|width)\s*[:=]?\s*([a-z0-9.+-]+(?:\s+[a-z0-9.+-]+)?)", re.I)


class StateTracker:
    """Deterministic conversation state with explicit override handling."""

    def update(self, state: SessionState, message: str, turn: int) -> SessionState:
        text = normalize_text(message)
        state.messages.append(message)

        is_override = any(
            marker in text
            for marker in ("ignore my earlier", "instead", "changed my mind", "rather than")
        )
        if is_override:
            state.override_count += 1
            state.constraints.clear()
            state.free_text.clear()
            state.asked_attributes.clear()
            state.no_preference.clear()
            state.intent = "buying"

        category_match = CATEGORY_RE.search(message)
        if category_match:
            state.category = category_match.group(1).strip(" ,.;")

        if "still exploring" in text or "browse" in text or "recommend" in text:
            state.intent = "browsing"
        if any(marker in text for marker in ("key requirement", "what i need is", "must", "cannot")):
            state.intent = "buying"

        self._record_no_preference(state, text)
        extracted_spans: list[str] = []

        for marker, hard in (
            ("a key requirement is:", True),
            ("what i need is:", True),
            ("for that, what matters is:", False),
        ):
            if marker in text:
                tail = text.split(marker, 1)[1].strip(" .")
                values = [part.strip(" .") for part in tail.split(";") if part.strip(" .")]
                for value in values:
                    self._add_value(state, value, turn, hard=hard)
                extracted_spans.extend(values)

        price_match = PRICE_MAX_RE.search(text)
        if price_match:
            self._add_value(
                state,
                f"budget under ${price_match.group(1)}",
                turn,
                hard=True,
                replace=True,
            )
        elif (around_match := PRICE_AROUND_RE.search(text)):
            self._add_value(
                state,
                f"budget around ${around_match.group(1)}",
                turn,
                hard=False,
                replace=True,
            )

        tokens = set(re.findall(r"[a-z0-9]+", text))
        for color in sorted(tokens & COLORS):
            if self._has_constraint(state, "color", color):
                continue
            self._add_value(state, color, turn, hard="must" in text, replace=is_override)
        for material in sorted(tokens & MATERIALS):
            if self._has_constraint(state, "material", material):
                continue
            negated = bool(re.search(rf"\b(?:no|not|avoid|without)\s+{re.escape(material)}\b", text))
            self._add_value(
                state,
                material,
                turn,
                hard=negated or "must" in text,
                negated=negated,
                replace=is_override,
            )

        if size_match := SIZE_RE.search(text):
            self._add_value(state, f"size {size_match.group(1)}", turn, hard="must" in text, replace=True)

        self._capture_free_text(state, message, extracted_spans, is_override)
        return state

    @staticmethod
    def _has_constraint(state: SessionState, attribute: str, value: str) -> bool:
        return any(
            item.value.casefold() == value.casefold()
            for item in state.constraints.get(attribute, [])
        )

    def _add_value(
        self,
        state: SessionState,
        value: str,
        turn: int,
        *,
        hard: bool,
        negated: bool = False,
        replace: bool = False,
    ) -> None:
        cleaned = value.strip(" ,.;")
        if not cleaned:
            return
        attribute = classify_constraint(cleaned)
        state.add_constraint(
            Constraint(
                attribute=attribute,
                value=cleaned,
                source_turn=turn,
                hard=hard,
                negated=negated,
            ),
            replace=replace,
        )

    @staticmethod
    def _record_no_preference(state: SessionState, text: str) -> None:
        match = re.search(
            r"(?:no preference|don't have (?:a|an additional) preference) for\s+([a-z_]+)",
            text,
        )
        if match:
            state.no_preference.add(match.group(1))

    @staticmethod
    def _capture_free_text(
        state: SessionState,
        message: str,
        extracted_spans: list[str],
        is_override: bool,
    ) -> None:
        lowered = message.casefold()
        boilerplate = (
            "those options are not quite right",
            "i don't have a preference for",
            "i don't have an additional preference for",
            "for that, what matters is:",
            "a key requirement is:",
            "what i need is:",
        )
        if any(marker in lowered for marker in boilerplate):
            return
        fragment = message
        category_match = CATEGORY_RE.search(fragment)
        if category_match:
            fragment = fragment[category_match.end():].strip(" ,.;")
        if "still exploring" in fragment.casefold():
            fragment = ""
        if is_override and ":" in fragment:
            fragment = fragment.split(":", 1)[1]
        if fragment and not any(fragment.casefold() == item.casefold() for item in extracted_spans):
            state.free_text.append(fragment.strip(" ,.;"))


def budget_limit(state: SessionState) -> float | None:
    for constraint in reversed(state.constraints.get("budget", [])):
        match = re.search(r"\$\s*(\d+(?:\.\d+)?)", constraint.value)
        if match:
            return float(match.group(1))
    return None
