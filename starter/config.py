from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AgentConfig:
    variant: str = "v6"
    sparse_top_k: int = 160
    dense_top_k: int = 160
    fusion_k: int = 60
    dense_rrf_weight: float = 0.25
    enable_dense: bool = True
    enable_rerank: bool = True
    enable_hard_gate: bool = True
    max_clarifications: int = 4
    last_question_turn: int = 5
    exact_constraint_weight: float = 0.0
    enable_llm: bool = False
    llm_min_confidence: float = 0.70
    dense_model_path: str = "artifacts/models/all-MiniLM-L6-v2"
    dense_index_path: str = "artifacts/dense/product_embeddings.npy"
    dense_ids_path: str = "artifacts/dense/product_ids.json"

    @classmethod
    def for_variant(cls, variant: str) -> "AgentConfig":
        normalized = variant.casefold()
        if normalized == "v1":
            return cls(
                variant="v1",
                enable_dense=False,
                enable_rerank=False,
                enable_hard_gate=False,
            )
        if normalized == "v2":
            return cls(
                variant="v2",
                enable_dense=True,
                dense_rrf_weight=1.0,
                enable_rerank=False,
                enable_hard_gate=False,
            )
        if normalized == "v3":
            return cls(variant="v3")
        if normalized == "v4":
            return cls(variant="v4", last_question_turn=8)
        if normalized == "v5":
            return cls(
                variant="v5",
                last_question_turn=8,
                exact_constraint_weight=0.35,
            )
        if normalized == "v6":
            return cls(
                variant="v6",
                last_question_turn=8,
                exact_constraint_weight=0.70,
            )
        if normalized == "v7":
            return cls(
                variant="v7",
                last_question_turn=8,
                exact_constraint_weight=0.70,
                enable_llm=True,
            )
        raise ValueError(f"unknown agent variant: {variant}")

    @classmethod
    def from_environment(cls) -> "AgentConfig":
        return cls.for_variant(os.getenv("TECHJAM_VARIANT", "v6"))
