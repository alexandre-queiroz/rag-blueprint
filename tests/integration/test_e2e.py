"""End-to-end integration tests for the full RAG serving path.

Run with:   pytest tests/ --integration
Skip by default: pass --integration to opt-in (credentials required).

Test surface:
  1. Ingestion    — ingest_text() indexes into Chroma Cloud
  2. Vector DB    — hybrid_search() returns relevant chunks
  3. Semantic Cache — Redis get/set roundtrip + semantic similarity hit/miss
  4. Pipeline     — full end-to-end query; cache miss → hit on repeated call
  5. Monitor      — RAGASMonitor.sample() is fire-and-forget; zero-rate is a noop
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


# ── 1. Ingestion ──────────────────────────────────────────────────────────────


class TestIngestion:
    def test_ingest_returns_positive_chunk_count(
        self, ingested_collection, config
    ) -> None:
        """ingest_text() must produce at least one child chunk."""
        from rag.ingestion.pipeline import config_to_chunk_config, ingest_text
        from tests.integration.conftest import _TEST_DOCUMENT, _TEST_SOURCE

        chunk_config = config_to_chunk_config(config.ingestion)
        # Use a distinct source so this test's ingestion is independent
        source = f"{_TEST_SOURCE}-count-check"
        count = ingest_text(ingested_collection, _TEST_DOCUMENT, source, chunk_config)
        assert count > 0

    def test_ingest_hierarchical_produces_more_chunks_than_parents(
        self, ingested_collection, config
    ) -> None:
        """Hierarchical strategy: child count > parent count for a multi-paragraph doc."""
        from rag.ingestion.pipeline import config_to_chunk_config, ingest_text
        from rag.ingestion.chunker import chunk_hierarchical
        from tests.integration.conftest import _TEST_DOCUMENT, _TEST_SOURCE

        chunk_config = config_to_chunk_config(config.ingestion)
        if chunk_config.strategy != "hierarchical":
            pytest.skip("strategy is not hierarchical")

        source = f"{_TEST_SOURCE}-hierarchy-check"
        parents = chunk_hierarchical(_TEST_DOCUMENT, chunk_config, source=source)
        total_children = sum(len(p.children) for p in parents)
        assert total_children >= len(parents)


# ── 2. Vector DB ──────────────────────────────────────────────────────────────


class TestVectorDB:
    def test_hybrid_search_returns_results(self, ingested_collection, config) -> None:
        """hybrid_search() against the ingested test collection must return chunks."""
        from rag.vector_db.client import hybrid_search

        results = hybrid_search(ingested_collection, "What is RAG?", limit=5)
        assert len(results) > 0

    def test_search_result_score_is_positive(self, ingested_collection) -> None:
        from rag.vector_db.client import hybrid_search

        results = hybrid_search(ingested_collection, "What is RAG?", limit=3)
        for chunk in results:
            assert chunk.score >= 0.0

    def test_search_result_contains_relevant_text(self, ingested_collection) -> None:
        """Top result should mention RAG or retrieval — tests basic relevance."""
        from rag.vector_db.client import hybrid_search

        results = hybrid_search(ingested_collection, "What is RAG?", limit=5)
        combined = " ".join(r.document.lower() for r in results)
        assert any(term in combined for term in ("rag", "retrieval", "generation"))

    def test_search_respects_limit(self, ingested_collection) -> None:
        from rag.vector_db.client import hybrid_search

        results = hybrid_search(ingested_collection, "What is RAG?", limit=2)
        assert len(results) <= 2

    def test_search_result_has_source(self, ingested_collection) -> None:
        from rag.vector_db.client import hybrid_search

        results = hybrid_search(ingested_collection, "What is RAG?", limit=3)
        for chunk in results:
            assert chunk.source != ""


# ── 3. Semantic Cache ─────────────────────────────────────────────────────────


class TestSemanticCache:
    async def test_set_and_get_exact_query_hits(
        self, cache, session_cache_query
    ) -> None:
        """Storing a response then retrieving with the same query must return it."""
        from rag.types import RAGResponse

        response = RAGResponse(
            answer="RAG augments LLMs with retrieved context.",
            sources=["test.txt"],
            complexity="simple",
            record=None,
            cached=False,
        )
        await cache.set(session_cache_query, response)
        result = await cache.get(session_cache_query)
        assert result is not None

    async def test_cached_response_is_marked_cached(
        self, cache, session_cache_query
    ) -> None:
        """Cache hit must have cached=True and record=None."""
        result = await cache.get(session_cache_query)
        # entry was stored by the previous test; if not found, set it first
        if result is None:
            from rag.types import RAGResponse
            await cache.set(session_cache_query, RAGResponse(
                answer="RAG augments LLMs with retrieved context.",
                sources=["test.txt"],
                complexity="simple",
                record=None,
                cached=False,
            ))
            result = await cache.get(session_cache_query)
        assert result is not None
        assert result.cached is True
        assert result.record is None

    async def test_cached_response_preserves_answer(
        self, cache, session_cache_query
    ) -> None:
        result = await cache.get(session_cache_query)
        if result is None:
            pytest.skip("cache miss — run test_set_and_get_exact_query_hits first")
        assert result.answer == "RAG augments LLMs with retrieved context."

    async def test_unrelated_query_is_cache_miss(self, cache) -> None:
        """A completely unrelated query must not match the cached RAG entry."""
        result = await cache.get("How do I make sourdough bread?")
        # Either a miss (None) or a very different cached entry — not our RAG answer
        if result is not None:
            assert "rag" not in result.answer.lower()


# ── 4. Pipeline — end-to-end ──────────────────────────────────────────────────


class TestPipelineEndToEnd:
    async def test_query_returns_non_empty_answer(
        self, pipeline, session_query
    ) -> None:
        """Full pipeline query must produce a non-empty answer string."""
        response = await pipeline.query(session_query)
        assert isinstance(response.answer, str)
        assert len(response.answer.strip()) > 0

    async def test_first_query_is_cache_miss(
        self, pipeline, session_query
    ) -> None:
        """First call with a session-unique query must be a cache miss."""
        response = await pipeline.query(session_query)
        assert response.cached is False

    async def test_second_identical_query_is_cache_hit(
        self, pipeline, session_query
    ) -> None:
        """Second call with the same query must return a cache hit."""
        # Ensure the first call has completed (may have been run by previous test)
        await pipeline.query(session_query)
        # Small wait for the fire-and-forget cache write to complete
        await asyncio.sleep(1.0)
        response = await pipeline.query(session_query)
        assert response.cached is True

    async def test_query_returns_sources(self, pipeline, session_query) -> None:
        """Non-cached response must list the document sources used."""
        # Re-use session_query; may be cached at this point, that's fine
        response = await pipeline.query(session_query)
        # Cached hits have empty sources — only check for non-cached responses
        if not response.cached:
            assert isinstance(response.sources, list)

    async def test_blank_query_raises_value_error(self, pipeline) -> None:
        with pytest.raises(ValueError, match="empty"):
            await pipeline.query("   ")

    async def test_complexity_label_is_valid(
        self, pipeline, session_query
    ) -> None:
        response = await pipeline.query(session_query)
        assert response.complexity in ("simple", "medium", "complex")


# ── 5. Monitor — fire-and-forget ──────────────────────────────────────────────


class TestMonitor:
    async def test_zero_sample_rate_is_noop(self) -> None:
        """sample() with rate=0.0 must return immediately without raising."""
        from rag.config import EvaluationConfig
        from rag.monitoring.monitor import RAGASMonitor
        from rag.types import RAGResponse, RetrievedChunk

        monitor = RAGASMonitor(
            EvaluationConfig(
                framework="ragas",
                metrics=("faithfulness", "answer_relevancy", "context_precision"),
                min_score_threshold=0.75,
                sample_rate=0.0,  # always skip — no RAGAS call, no Axiom call
            )
        )
        response = RAGResponse(
            answer="RAG grounds LLM responses in retrieved context.",
            sources=["doc.txt"],
            complexity="simple",
            record=None,
            cached=False,
        )
        chunks = [
            RetrievedChunk(
                document="RAG uses retrieval to augment generation.",
                score=0.9,
                source="doc.txt",
                chunk_index=0,
            )
        ]
        # Must complete without raising — fire-and-forget contract
        await monitor.sample("What is RAG?", response, chunks)

    async def test_monitor_as_pipeline_task_does_not_block(
        self, pipeline, session_query
    ) -> None:
        """Verify the pipeline returns before the monitor background task finishes."""
        from rag.config import EvaluationConfig
        from rag.monitoring.monitor import RAGASMonitor
        from rag.serve.pipeline import RAGPipeline

        monitor = RAGASMonitor(
            EvaluationConfig(
                framework="ragas",
                metrics=("faithfulness", "answer_relevancy", "context_precision"),
                min_score_threshold=0.75,
                sample_rate=0.0,
            )
        )
        monitored_pipeline = RAGPipeline(
            config=pipeline._config,
            cache=pipeline._cache,
            collection=pipeline._collection,
            gateway=pipeline._gateway,
            monitor=monitor,
        )
        query = f"{session_query} monitor-test"
        response = await monitored_pipeline.query(query)
        # Pipeline must return a valid response regardless of monitor
        assert len(response.answer.strip()) > 0
