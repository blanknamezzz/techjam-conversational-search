from __future__ import annotations

import unittest

from starter.models import SessionState
from starter.state_tracker import StateTracker


class StateTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = SessionState("s", {})
        self.tracker = StateTracker()

    def test_accumulates_category_and_revealed_constraints(self) -> None:
        self.tracker.update(
            self.state,
            "I'm looking for Women's Boots. A key requirement is: leather.",
            1,
        )
        self.tracker.update(
            self.state,
            "For that, what matters is: color: black; waterproof.",
            2,
        )
        self.assertEqual(self.state.category, "Women's Boots")
        self.assertIn("leather", self.state.query_text())
        self.assertIn("black", self.state.query_text())
        self.assertIn("waterproof", self.state.query_text())

    def test_override_removes_old_preference_but_keeps_category(self) -> None:
        self.tracker.update(self.state, "I'm looking for Boots. I prefer black.", 1)
        self.tracker.update(
            self.state,
            "Actually, ignore my earlier preference. What I need is: leather.",
            3,
        )
        query = self.state.query_text()
        self.assertIn("Boots", query)
        self.assertIn("leather", query)
        self.assertNotIn("black", query)
        self.assertEqual(self.state.override_count, 1)
        self.assertTrue(self.state.constraints["material"][0].hard)

    def test_records_boundary_no_preference(self) -> None:
        self.tracker.update(
            self.state,
            "I don't have a preference for material; please use your judgment.",
            2,
        )
        self.assertIn("material", self.state.no_preference)
