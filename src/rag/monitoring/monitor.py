from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from rag.config import EvaluationConfig
from rag.types import RAGResponse, RetrievedChunk


class RAGASMonitor:
    """Async RAGAS quality monitor with Axiom event emission.

    Samples live traffic at the configured rate, evaluates with RAGAS
    (faithfulness, answer_relevancy, context_precision), and emits
    structured events to Axiom. Always runs fire-and-forget — never
    blocks the serving path.

    Runtime dependencies (not required at import time):
        ragas + datasets  →  uv sync --extra eval
        axiom-py          →  uv sync --extra monitoring

    If AXIOM_API_KEY or AXIOM_DATASET are not set, event emission is
    silently skipped — evaluation scores are still computed.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    async def sample(
        self,
        query: str,
        response: RAGResponse,
        chunks: list[RetrievedChunk],
    ) -> None:
        """Evaluate one request if selected by the configured sampling rate.

        Intended to be called from the pipeline as a background task:
            asyncio.create_task(monitor.sample(query, response, chunks))
        """
        import random

        if not _should_sample(self._config.sample_rate, random.random()):
            return

        scores = await asyncio.to_thread(_run_ragas, query, response, chunks)
        event = _format_axiom_event(scores, query, response, self._config.min_score_threshold)
        await asyncio.to_thread(_emit_to_axiom, event)


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────


def _should_sample(sample_rate: float, random_value: float) -> bool:
    """Return True if this request should be sampled for evaluation.

    Accepts an explicit random_value for deterministic testing — callers
    pass random.random() at the call site.
    """
    return random_value < sample_rate


def _build_ragas_dataset(
    query: str,
    response: RAGResponse,
    chunks: list[RetrievedChunk],
) -> dict[str, list[str] | list[list[str]]]:
    """Build the RAGAS-compatible evaluation input for one sample.

    Column names match the RAGAS 0.2.x SingleTurnSample schema.
    """
    return {
        "user_input": [query],
        "response": [response.answer],
        "retrieved_contexts": [[chunk.document for chunk in chunks]],
    }


def _scores_below_threshold(
    scores: dict[str, float],
    threshold: float,
) -> list[str]:
    """Return metric names whose score is strictly below the threshold."""
    return [metric for metric, score in scores.items() if score < threshold]


def _format_axiom_event(
    scores: dict[str, float],
    query: str,
    response: RAGResponse,
    threshold: float,
) -> dict[str, object]:
    """Build the structured event payload for Axiom ingestion.

    Includes operational metadata from the CompletionRecord when present
    (model, provider, tokens, cost, latency). Cache hits have no record.
    """
    failing = _scores_below_threshold(scores, threshold)
    event: dict[str, object] = {
        "_time": datetime.now(tz=timezone.utc).isoformat(),
        "query": query,
        "complexity": response.complexity,
        "cached": response.cached,
        "below_threshold": failing,
        "alert": len(failing) > 0,
    }
    if response.record is not None:
        event["model"] = response.record.model
        event["provider"] = response.record.provider
        event["total_tokens"] = response.record.total_tokens
        event["cost_usd"] = response.record.cost_usd
        event["latency_ms"] = response.record.latency_ms
    event.update(scores)
    return event


# ── Side-effecting runtime helpers (not unit-tested) ─────────────────────────


def _run_ragas(
    query: str,
    response: RAGResponse,
    chunks: list[RetrievedChunk],
) -> dict[str, float]:
    """Run RAGAS evaluation synchronously (must be called via asyncio.to_thread).

    Imports ragas lazily — requires uv sync --extra eval.
    Returns metric_name → score (0.0–1.0) for all configured metrics.
    """
    from datasets import Dataset  # type: ignore[import-untyped]
    from ragas import evaluate  # type: ignore[import-untyped]
    from ragas.metrics import (  # type: ignore[import-untyped]
        answer_relevancy,
        context_precision,
        faithfulness,
    )

    dataset = Dataset.from_dict(_build_ragas_dataset(query, response, chunks))
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    row: dict[str, object] = result.to_pandas().iloc[0].to_dict()
    return {
        k: float(v)  # type: ignore[arg-type]
        for k, v in row.items()
        if k in {"faithfulness", "answer_relevancy", "context_precision"}
    }


def _emit_to_axiom(event: dict[str, object]) -> None:
    """Emit a structured event to Axiom (must be called via asyncio.to_thread).

    Reads AXIOM_API_KEY and AXIOM_DATASET from environment. Silently
    skips emission if either variable is absent — useful for local dev.
    Requires uv sync --extra monitoring.
    """
    api_key = os.environ.get("AXIOM_API_KEY", "")
    dataset = os.environ.get("AXIOM_DATASET", "")
    if not api_key or not dataset:
        return

    from axiom import Client  # type: ignore[import-untyped]

    client = Client(token=api_key)
    client.ingest_events(dataset=dataset, events=[event])
