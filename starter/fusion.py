from __future__ import annotations

from .models import Candidate


def reciprocal_rank_fusion(
    channels: dict[str, list[Candidate]],
    *,
    rank_constant: int = 60,
    channel_weights: dict[str, float] | None = None,
) -> list[Candidate]:
    weights = channel_weights or {}
    merged: dict[str, Candidate] = {}
    for channel_name, candidates in channels.items():
        weight = weights.get(channel_name, 1.0)
        for fallback_rank, incoming in enumerate(candidates, start=1):
            rank = incoming.ranks.get(channel_name, fallback_rank)
            candidate = merged.setdefault(incoming.parent_asin, Candidate(incoming.parent_asin))
            candidate.ranks[channel_name] = rank
            candidate.scores.update(incoming.scores)
            candidate.scores["rrf"] = (
                candidate.scores.get("rrf", 0.0)
                + weight / (rank_constant + rank)
            )
    return sorted(
        merged.values(),
        key=lambda item: (-item.scores.get("rrf", 0.0), item.parent_asin),
    )
