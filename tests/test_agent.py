from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from starter.agent import Agent
from starter.config import AgentConfig


class AgentContractTest(unittest.TestCase):
    def test_v6_agent_returns_catalog_valid_contract(self) -> None:
        products = [
            {
                "parent_asin": "A",
                "title": "Black leather hiking boot",
                "categories": ["Women", "Boots"],
                "features": ["waterproof"],
                "details": {},
                "store": "Example",
                "description": [],
                "price": 80.0,
            },
            {
                "parent_asin": "B",
                "title": "Cotton shirt",
                "categories": ["Women", "Shirts"],
                "features": [],
                "details": {},
                "store": "Example",
                "description": [],
                "price": 20.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in products), encoding="utf-8")
            agent = Agent(path, AgentConfig.for_variant("v6"))
            agent.reset("session", {"summary": "prefers durable products"})
            response = agent.respond("session", "I'm looking for black hiking boots.", 1, 10)
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], {"material", "feature", "color", "size", "other", None})
        self.assertEqual(response["recommendations"][0]["parent_asin"], "A")
