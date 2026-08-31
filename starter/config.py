from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AgentConfig:
    variant: str = "v6"
    sparse_top_k: int = 160
    dense_top_k: int = 200
    fusion_k: int = 60
    dense_rrf_weight: float = 0.25
    override_dense_rrf_scale: float = 0.60
    enable_dense: bool = True
    enable_rerank: bool = True
    enable_hard_gate: bool = True
    max_clarifications: int = 4
    last_question_turn: int = 5
    exact_constraint_weight: float = 0.0
    enable_llm: bool = False
    llm_min_confidence: float = 0.70
    llm_trigger_policy: str = "first_or_complex"
    dense_model_path: str = "artifacts/models/all-MiniLM-L6-v2"
    dense_index_path: str = "artifacts/dense/product_embeddings.npy"
    dense_ids_path: str = "artifacts/dense/product_ids.json"

    @classmethod
    def for_variant(cls, variant: str) -> "AgentConfig":
        normalized = variant.casefold()
        if normalized == "v6":
            return cls(
                variant="v6",
                last_question_turn=8,
                exact_constraint_weight=0.70,
            )
        if normalized == "v9":
            return cls(
                variant="v9",
                last_question_turn=8,
                exact_constraint_weight=0.70,
                enable_llm=True,
                llm_min_confidence=0.80,
                llm_trigger_policy="exploratory_or_complex",
            )
        raise ValueError(f"unknown agent variant: {variant}")

    @classmethod
    def from_environment(cls) -> "AgentConfig":
        return cls.for_variant(os.getenv("TECHJAM_VARIANT", "v6"))
