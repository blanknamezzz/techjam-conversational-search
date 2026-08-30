from __future__ import annotations

from .models import SessionState


QUESTION_TEXT = {
    "material": "Do you have a preferred material?",
    "feature": "Which feature matters most to you?",
    "color": "Do you have a color preference?",
    "size": "Do you have a size or fit requirement?",
    "other": "Are there any other must-have details I should prioritize?",
}


class ClarificationPolicy:
    """Small deterministic policy that always recommends while asking at most one field."""

    def __init__(self, max_questions: int = 4, last_question_turn: int = 5) -> None:
        self.max_questions = max_questions
        self.last_question_turn = last_question_turn

    def choose(self, state: SessionState, turn: int) -> tuple[str | None, str]:
        if len(state.asked_attributes) >= self.max_questions or turn > self.last_question_turn:
            return None, "Here are the strongest matches for your current preferences."

        present = set(state.constraints)
        priority = ["material", "feature", "color", "size", "other"]
        for attribute in priority:
            if attribute in state.no_preference:
                continue
            if attribute in state.asked_attributes and attribute != "other":
                continue
            if attribute in present and attribute != "other":
                continue
            if attribute == "other" and state.asked_attributes.count("other") >= 2:
                continue
            state.asked_attributes.append(attribute)
            return attribute, QUESTION_TEXT[attribute]

        return None, "Here are the strongest matches for your current preferences."
