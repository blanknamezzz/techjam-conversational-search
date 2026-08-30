from __future__ import annotations

import unittest

from starter.fusion import reciprocal_rank_fusion
from starter.models import Candidate


class FusionTest(unittest.TestCase):
    def test_rrf_rewards_candidates_present_in_both_channels(self) -> None:
        sparse = [Candidate("A", ranks={"bm25": 1}), Candidate("B", ranks={"bm25": 2})]
        dense = [Candidate("C", ranks={"dense": 1}), Candidate("B", ranks={"dense": 2})]
        result = reciprocal_rank_fusion({"bm25": sparse, "dense": dense})
        self.assertEqual(result[0].parent_asin, "B")
        self.assertEqual(result[0].ranks, {"bm25": 2, "dense": 2})
