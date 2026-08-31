from __future__ import annotations

from pathlib import Path
import time

from .catalog import CatalogStore
from .clarification import ClarificationPolicy
from .config import AgentConfig
from .dense_retriever import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .models import SessionState
from .models import Constraint
from .optional_api import LLMResult, OptionalLLMClient
from .reranker import Reranker
from .state_tracker import StateTracker


class Agent:
    """Offline-first conversational hybrid-search agent implementing the official contract."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_environment()
        self.catalog = CatalogStore(catalog_path)
        self.state_tracker = StateTracker()
        self.clarification = ClarificationPolicy(
            self.config.max_clarifications,
            self.config.last_question_turn,
        )
        self.reranker = Reranker(
            self.catalog,
            hard_gate=self.config.enable_hard_gate,
            exact_constraint_weight=self.config.exact_constraint_weight,
        )
        self.dense = DenseRetriever(
            self.config.dense_model_path,
            self.config.dense_index_path,
            self.config.dense_ids_path,
            expected_ids=self.catalog.product_ids,
        ) if self.config.enable_dense else None
        self.llm = OptionalLLMClient() if self.config.enable_llm else None
        self.sessions: dict[str, SessionState] = {}
        self.diagnostics: dict[str, list[dict]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            user_profile=dict(user_profile or {}),
        )
        self.diagnostics[session_id] = []

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        started = time.perf_counter()
        override_count_before = self.sessions[session_id].override_count
        state = self.state_tracker.update(self.sessions[session_id], user_message, turn)
        llm_result = LLMResult(None, error="disabled")
        rewritten_query = ""
        if self.llm is not None and self.llm.should_call(
            user_message,
            turn,
            state,
            self.config.llm_trigger_policy,
        ):
            llm_result = self.llm.analyze(user_message, state)
            if llm_result.analysis is not None:
                rewritten_query = self._apply_llm_analysis(
                    state,
                    llm_result.analysis,
                    override_count_before,
                )
        query = state.query_text() or user_message
        if rewritten_query and rewritten_query.casefold() not in query.casefold():
            query = f"{query} {rewritten_query}".strip()

        sparse = self.catalog.sparse_search(query, self.config.sparse_top_k)
        channels = {"bm25": sparse}
        if self.dense is not None and self.dense.available:
            try:
                channels["dense"] = self.dense.search(query, self.config.dense_top_k)
            except Exception:
                channels["dense"] = []
        dense_weight = self.config.dense_rrf_weight
        if state.override_count:
            # After an explicit preference reset the query is intentionally
            # short. Downweight semantic expansion until the new intent has
            # accumulated enough evidence, reducing stale semantic neighbors.
            dense_weight *= self.config.override_dense_rrf_scale
        candidates = reciprocal_rank_fusion(
            channels,
            rank_constant=self.config.fusion_k,
            channel_weights={"bm25": 1.0, "dense": dense_weight},
        )

        ranked = self.reranker.rank(candidates, state) if self.config.enable_rerank else candidates
        ask_attribute, message = self.clarification.choose(state, turn)
        recommendations = [
            {"parent_asin": candidate.parent_asin}
            for candidate in ranked[:max(0, top_k)]
        ]
        self.diagnostics[session_id].append({
            "turn": turn,
            "intent": state.intent,
            "query": query,
            "ask_attribute": ask_attribute,
            "sparse_candidates": len(sparse),
            "dense_candidates": len(channels.get("dense", [])),
            "fused_candidates": len(candidates),
            "llm_used": llm_result.analysis is not None,
            "llm_error": llm_result.error,
            "sparse_ids": [item.parent_asin for item in sparse],
            "dense_ids": [item.parent_asin for item in channels.get("dense", [])],
            "fused_ids": [item.parent_asin for item in candidates],
            "ranked_ids": [item.parent_asin for item in ranked],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        })
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": llm_result.prompt_tokens,
                "completion_tokens": llm_result.completion_tokens,
            },
        }

    def _apply_llm_analysis(
        self,
        state: SessionState,
        analysis: dict,
        override_count_before: int,
    ) -> str:
        intent = analysis.get("intent")
        if intent in {"buying", "browsing"}:
            state.intent = intent
        if analysis.get("override") is True and state.override_count == override_count_before:
            state.override_count += 1
            state.constraints.clear()
            state.free_text.clear()
            state.asked_attributes.clear()
            state.no_preference.clear()
        category = analysis.get("category")
        if isinstance(category, str) and category.strip():
            state.category = category.strip()[:160]
        if self.llm is not None:
            for item in self.llm.validated_constraints(analysis, self.config.llm_min_confidence):
                if item["attribute"] == "category" and not state.category:
                    state.category = item["value"]
                    continue
                state.add_constraint(Constraint(
                    attribute=item["attribute"],
                    value=item["value"],
                    source_turn=len(state.messages),
                    hard=item["hard"],
                    negated=item["negated"],
                    confidence=item["confidence"],
                ))
        rewritten = analysis.get("rewritten_query")
        return rewritten.strip()[:300] if isinstance(rewritten, str) else ""
