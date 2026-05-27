from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rag.cache.semantic_cache import SemanticCache
from rag.classifier.classifier import classify
from rag.config import Config
from rag.gateway.gateway import LLMGateway
from rag.types import ComplexityLabel, RAGResponse

if TYPE_CHECKING:
    import chromadb

    from rag.monitoring.monitor import RAGASMonitor


class RAGPipeline:
    """End-to-end serving path orchestrator.

    Wires: classify → cache → vector search → gateway → (optional) monitor.

    All dependencies are constructor-injected — this class does not
    instantiate clients or read environment variables directly.

    The chromadb and monitoring imports are deferred to call time so the
    module loads cleanly in environments without the full Chroma Cloud SDK
    or the eval/monitoring optional dependency groups.

    Usage::

        pipeline = RAGPipeline(
            config=load_config(),
            cache=SemanticCache(config.semantic_cache),
            collection=get_collection(client, config.vector_db.collection),
            gateway=LLMGateway(config.llm_gateway, config.token_budget, breakers),
            monitor=RAGASMonitor(config.evaluation),  # optional
        )
        response = await pipeline.query("What is RAG?")
    """

    def __init__(
        self,
        config: Config,
        cache: SemanticCache,
        collection: chromadb.Collection,
        gateway: LLMGateway,
        monitor: RAGASMonitor | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._collection = collection
        self._gateway = gateway
        self._monitor = monitor

    async def query(self, query: str) -> RAGResponse:
        """Run a query end-to-end through the serving pipeline.

        Steps:
        1. Validate and normalize the query — raises ValueError for blank input.
        2. Classify complexity — heuristics → embedding → LLM.
        3. Semantic cache check — hit returns immediately, skipping Chroma and LLM.
        4. Hybrid vector search (Chroma Cloud native RRF).
        5. LLM Gateway — token budget, model selection, fallback chain, graceful degradation.
        6. Cache write — fire-and-forget, does not block the returned response.
        7. RAGAS monitor — fire-and-forget if monitor is configured; skipped on cache hits.

        Raises:
            ValueError: if query is blank after stripping whitespace.
            TokenBudgetExceededError: if the assembled prompt exceeds the configured token limit.
        """
        # Deferred import — chromadb Cloud SDK may not be available in all environments
        from rag.vector_db.client import hybrid_search

        normalized = _validate_query(query)

        # Classify — determines primary model and fallback chain
        complexity: ComplexityLabel = await classify(normalized, self._config.classifier)

        # Cache check — short-circuit before any vector or LLM call
        cached = await self._cache.get(normalized)
        if cached is not None:
            return cached

        # Hybrid vector search — Chroma SDK is synchronous
        chunks = hybrid_search(
            self._collection,
            normalized,
            self._config.vector_db.top_k,
        )

        # LLM Gateway — raises TokenBudgetExceededError if prompt is over budget
        response = await self._gateway.complete(normalized, chunks, complexity)

        # Fire-and-forget: cache write and RAGAS monitoring do not block the caller.
        #
        # KNOWN LIMITATION — task lifecycle in serverless/ephemeral environments:
        #   asyncio.create_task() schedules work on the running event loop, but the
        #   task is NOT awaited before the response is returned. In long-lived servers
        #   (uvicorn, gunicorn) this is fine — the loop continues running after the
        #   request handler returns. In ephemeral environments (AWS Lambda, Vercel,
        #   Cloud Run with --max-instances=1 scale-to-zero) the process may be frozen
        #   or terminated before the background task completes, causing silent cache
        #   misses and lost monitoring events.
        #
        #   Escalation path: replace create_task() with a durable background queue
        #   (Redis Streams, SQS, Cloud Tasks). The queue write is synchronous and
        #   cheap; a separate worker process consumes the queue and performs cache
        #   writes and RAGAS evaluations outside the request lifecycle.
        asyncio.create_task(self._cache.set(normalized, response))
        if self._monitor is not None:
            asyncio.create_task(self._monitor.sample(normalized, response, chunks))

        return response


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_query(query: str) -> str:
    """Return normalized query or raise ValueError if blank.

    Normalizes by stripping leading/trailing whitespace. Rejects empty
    or whitespace-only strings before they reach any external service.
    """
    normalized = query.strip()
    if not normalized:
        raise ValueError("Query must not be empty or whitespace-only")
    return normalized
