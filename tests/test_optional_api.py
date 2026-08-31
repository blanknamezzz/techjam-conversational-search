from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from starter.optional_api import OptionalLLMClient
from starter.models import SessionState


class OptionalLLMClientTest(unittest.TestCase):
    def test_loads_only_explicit_local_llm_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "TECHJAM_LLM_API_KEY=test-value\n"
                "TECHJAM_LLM_BASE_URL=https://example.invalid/v1\n"
                "TECHJAM_LLM_MODEL=test-model\n"
                "UNRELATED_SECRET=must-not-load\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                client = OptionalLLMClient(env_file)
                self.assertTrue(client.available)
                self.assertEqual(client.url, "https://example.invalid/v1/chat/completions")
                self.assertNotIn("UNRELATED_SECRET", os.environ)

    def test_parses_json_code_fence_and_validates_constraints(self) -> None:
        payload = {
            "constraints": [
                {"attribute": "material", "value": "leather", "confidence": 0.9},
                {"attribute": "invalid", "value": "x", "confidence": 1.0},
            ]
        }
        parsed = OptionalLLMClient._parse_json_content(
            "```json\n" + json.dumps(payload) + "\n```"
        )
        constraints = OptionalLLMClient.validated_constraints(parsed, 0.7)
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0]["attribute"], "material")

    def test_selective_policy_skips_no_preference_but_calls_on_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "TECHJAM_LLM_API_KEY=test-value\n"
                "TECHJAM_LLM_BASE_URL=https://example.invalid/v1\n"
                "TECHJAM_LLM_MODEL=test-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                client = OptionalLLMClient(env_file)
                state = SessionState("session", {}, category="boots")
                self.assertFalse(client.should_call(
                    "I don't have a preference for color.", 2, state, "complex_only"
                ))
                self.assertTrue(client.should_call(
                    "Actually, ignore my earlier preference.", 3, state, "complex_only"
                ))
