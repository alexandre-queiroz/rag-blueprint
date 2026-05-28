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
        ragas + datasets + langchain-google-genai  →  uv sync --extra eval
        opentelemetry-sdk + opentelemetry-exporter-otlp-proto-http  →  uv sync --extra monitoring

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
        import logging
        import random

        log = logging.getLogger(__name__)

        if not _should_sample(self._config.sample_rate, random.random()):
            return

        try:
            scores = await asyncio.to_thread(_run_ragas, query, response, chunks)
            event = _format_axiom_event(scores, query, response, self._config.min_score_threshold)
            await asyncio.to_thread(_emit_to_axiom, event)
            log.info("ragas evaluation complete: %s", scores)
        except Exception:
            log.exception("ragas evaluation failed — skipping Axiom emit")


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

    Column names match the SingleTurnSample schema introduced in RAGAS 0.3.x+.
    """
    return {
        "user_input": [query],
        "response": [response.answer],
        "retrieved_contexts": [[chunk.document for chunk in chunks]],
        "reference": [response.answer],  # Proxy reference for reference-dependent metrics like context_precision
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

    Uses Google Gemini Flash Lite as the evaluation LLM and gemini-embedding-001
    as the embedding model — consistent with the rest of the system and avoids
    any dependency on an OpenAI key (RAGAS defaults to OpenAI internally).
    Requires GOOGLE_API_KEY in the environment.
    """
    import sys
    from types import ModuleType
    # Mock the missing deprecated module so ragas can import without failing
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        mock_module = ModuleType("langchain_community.chat_models.vertexai")
        class MockChatVertexAI:
            pass
        mock_module.ChatVertexAI = MockChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = mock_module

    from datasets import Dataset  # type: ignore[import-untyped]
    from langchain_google_genai import (  # type: ignore[import-untyped]
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas import evaluate  # type: ignore[import-untyped]
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]
    from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
    from ragas.metrics import (  # type: ignore[import-untyped]
        answer_relevancy,
        context_precision,
        faithfulness,
    )

    # LangChain wrappers are used here as RAGAS's required adapter interface —
    # not as orchestration. ADR-003 rejects LangChain for connecting pipeline
    # layers; this is an isolated call inside a background evaluation thread.
    # See the exception note at the bottom of ADR-003 for the full rationale.
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=google_api_key,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key,
        )
    )

    dataset = Dataset.from_dict(_build_ragas_dataset(query, response, chunks))
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    row: dict[str, object] = result.to_pandas().iloc[0].to_dict()
    return {
        k: float(v)  # type: ignore[arg-type]
        for k, v in row.items()
        if k in {"faithfulness", "answer_relevancy", "context_precision"}
    }


def _emit_to_axiom(event: dict[str, object]) -> None:
    """Emit a RAGAS evaluation span to Axiom via OpenTelemetry OTLP.

    Each evaluation becomes one span named 'ragas.evaluation' with all
    quality scores and operational metadata as span attributes. Axiom
    ingests OTLP traces natively — no Axiom SDK required.

    Reads AXIOM_API_KEY and AXIOM_DATASET from environment. Silently
    skips emission if either variable is absent — useful for local dev.
    Must be called via asyncio.to_thread() — OTLP export is blocking I/O.
    Requires uv sync --extra monitoring.
    """
    api_key = os.environ.get("AXIOM_API_KEY", "")
    dataset = os.environ.get("AXIOM_DATASET", "")
    if not api_key or not dataset:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-untyped]
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore[import-untyped]

    exporter = OTLPSpanExporter(
        endpoint="https://api.axiom.co/v1/traces",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Axiom-Dataset": dataset,
        },
    )
    provider = TracerProvider(resource=Resource.create({"service.name": "rag-production"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("rag.monitor")

    with tracer.start_as_current_span("ragas.evaluation") as span:
        for key, value in event.items():
            if key == "_time":
                continue
            attr: str | float | int | bool
            if isinstance(value, (str, float, int, bool)):
                attr = value
            elif isinstance(value, list):
                attr = ", ".join(str(v) for v in value)
            else:
                attr = str(value)
            span.set_attribute(f"rag.{key}", attr)

    provider.shutdown()  # flushes the span before returning
