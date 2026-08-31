from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import SessionState


ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}


def load_local_env(path: str | Path = ".env") -> None:
    """Load only TECHJAM_LLM_* values without overriding the process environment."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("TECHJAM_LLM_"):
            continue
        cleaned = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, cleaned)


@dataclass(frozen=True)
class LLMResult:
    analysis: dict[str, Any] | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class OptionalLLMClient:
    """Guarded OpenAI-compatible JSON client for optional online variants."""

    def __init__(self, env_file: str | Path = ".env") -> None:
        load_local_env(env_file)
        self.api_key = os.getenv("TECHJAM_LLM_API_KEY", "").strip()
        self.model = os.getenv("TECHJAM_LLM_MODEL", "").strip()
        explicit_url = os.getenv("TECHJAM_LLM_API_URL", "").strip()
        base_url = os.getenv("TECHJAM_LLM_BASE_URL", "").strip().rstrip("/")
        self.url = explicit_url or (f"{base_url}/chat/completions" if base_url else "")
        try:
            self.timeout = max(0.5, float(os.getenv("TECHJAM_LLM_TIMEOUT_SECONDS", "15")))
        except ValueError:
            self.timeout = 15.0
        self.available = bool(self.api_key and self.model and self.url)
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_latency_ms = 0.0
        self.errors: dict[str, int] = {}

    def should_call(
        self,
        message: str,
        turn: int,
        state: SessionState,
        policy: str = "first_or_complex",
    ) -> bool:
        if not self.available:
            return False
        lowered = message.casefold()
        no_preference = any(marker in lowered for marker in (
            "don't have a preference",
            "don't have an additional preference",
            "no preference for",
        ))
        override_or_reference = any(marker in lowered for marker in (
            "actually", "instead", "rather than", "changed my mind",
            "ignore my earlier", "first one", "that one",
        ))
        meaningful_negation = not no_preference and any(marker in lowered for marker in (
            "except", "avoid", "without", "don't want", " do not want", "not in",
        ))
        complex_request = override_or_reference or meaningful_negation or not state.query_text()
        if policy == "complex_only":
            return complex_request
        if policy == "exploratory_or_complex":
            return complex_request or (turn == 1 and state.intent == "browsing")
        if policy == "first_or_complex":
            return turn == 1 or complex_request
        raise ValueError(f"unknown LLM trigger policy: {policy}")

    def metrics(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_count / self.call_count, 6) if self.call_count else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "average_latency_ms": round(self.total_latency_ms / self.call_count, 3) if self.call_count else None,
            "errors": dict(sorted(self.errors.items())),
        }

    def analyze(self, message: str, state: SessionState) -> LLMResult:
        if not self.available:
            return LLMResult(None, error="disabled")
        self.call_count += 1
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract shopping intent into strict JSON. Return one JSON object only with: "
                        "intent ('buying' or 'browsing' or null), override (boolean), category (string or null), "
                        "constraints (array of objects with attribute, value, hard, negated, confidence), and "
                        "rewritten_query (short string or null). Allowed attributes are category, material, "
                        "color, size, style, brand, budget, feature, use_case, other. Never invent a product ID."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "current_state": {
                            "intent": state.intent,
                            "category": state.category,
                            "constraints": [
                                {
                                    "attribute": item.attribute,
                                    "value": item.value,
                                    "hard": item.hard,
                                    "negated": item.negated,
                                }
                                for item in state.active_constraints()
                            ],
                        },
                        "new_message": message,
                    }),
                },
            ],
        }
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            analysis = self._parse_json_content(content)
            usage = body.get("usage") or {}
            result = LLMResult(
                analysis=analysis,
                prompt_tokens=self._nonnegative_int(usage.get("prompt_tokens")),
                completion_tokens=self._nonnegative_int(usage.get("completion_tokens")),
            )
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            result = LLMResult(None, error=type(error).__name__)
        self.total_latency_ms += (time.perf_counter() - started) * 1000.0
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        if result.analysis is not None:
            self.success_count += 1
        else:
            self.error_count += 1
            error_name = result.error or "UnknownError"
            self.errors[error_name] = self.errors.get(error_name, 0) + 1
        return result

    @staticmethod
    def _parse_json_content(content: object) -> dict[str, Any]:
        if not isinstance(content, str):
            raise TypeError("model content must be a string")
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        value = json.loads(cleaned)
        if not isinstance(value, dict):
            raise TypeError("model output must be a JSON object")
        return value

    @staticmethod
    def _nonnegative_int(value: object) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    @staticmethod
    def validated_constraints(analysis: dict[str, Any], min_confidence: float) -> list[dict[str, Any]]:
        raw_constraints = analysis.get("constraints")
        if not isinstance(raw_constraints, list):
            return []
        result: list[dict[str, Any]] = []
        for item in raw_constraints:
            if not isinstance(item, dict):
                continue
            attribute = item.get("attribute")
            value = item.get("value")
            confidence = item.get("confidence", 0.0)
            if attribute not in ALLOWED_ATTRIBUTES or not isinstance(value, str) or not value.strip():
                continue
            if not isinstance(confidence, (int, float)) or float(confidence) < min_confidence:
                continue
            result.append({
                "attribute": attribute,
                "value": value.strip()[:240],
                "hard": item.get("hard") is True,
                "negated": item.get("negated") is True,
                "confidence": min(1.0, max(0.0, float(confidence))),
            })
        return result
