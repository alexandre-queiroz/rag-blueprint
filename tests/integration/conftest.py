"""Session-scoped fixtures for integration tests.

All fixtures require real infrastructure credentials. Load .env before the
session starts — tests are skipped automatically when credentials are absent
(controlled by the --integration flag defined in tests/conftest.py).

Test isolation:
  - Chroma: a unique collection per session (rag-blueprint-test-<uuid8>) is
    created at setup and deleted at teardown — no collision with production data.
  - Redis: test queries use a per-session UUID suffix so every run starts with a
    guaranteed cache miss regardless of leftover data from prior runs.
"""
from __future__ import annotations

import os
import uuid

import pytest
from dotenv import load_dotenv

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

_SESSION_ID = uuid.uuid4().hex[:8]
_TEST_COLLECTION = f"rag-blueprint-test-{_SESSION_ID}"
_TEST_SOURCE = f"integration-test-{_SESSION_ID}.txt"

# A small, self-contained RAG reference document used across all tests.
_TEST_DOCUMENT = """\
Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language
model outputs by incorporating relevant information retrieved from external knowledge bases.

RAG systems consist of three main components: an ingestion pipeline that processes and
indexes documents, a retrieval layer that fetches relevant chunks for a given query, and
a generation layer where the language model synthesizes a response grounded in the
retrieved context.

The key advantage of RAG over pure language model generation is factual grounding. The
model is constrained to answer from retrieved evidence, which significantly reduces
hallucination and improves the reliability of generated responses.

Hybrid search combines dense vector retrieval with sparse keyword matching via
Reciprocal Rank Fusion (RRF), returning results that are both semantically relevant and
lexically precise. This approach outperforms pure vector or keyword search alone.
"""


# ── Config ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def config():  # type: ignore[return]
    from rag.config import load_config
    return load_config()


# ── Chroma Cloud ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def chroma_client():  # type: ignore[return]
    from rag.vector_db.client import get_client
    return get_client()


@pytest.fixture(scope="session")
def test_collection(chroma_client):  # type: ignore[return]
    """Fresh Chroma collection for the test session — deleted at teardown."""
    from rag.vector_db.client import get_collection
    collection = get_collection(chroma_client, _TEST_COLLECTION)
    yield collection
    try:
        chroma_client.delete_collection(_TEST_COLLECTION)
    except Exception:
        pass  # best-effort cleanup


@pytest.fixture(scope="session")
def ingested_collection(test_collection, config):  # type: ignore[return]
    """Test collection pre-loaded with the test document."""
    from rag.ingestion.pipeline import config_to_chunk_config, ingest_text
    chunk_config = config_to_chunk_config(config.ingestion)
    ingest_text(test_collection, _TEST_DOCUMENT, _TEST_SOURCE, chunk_config)
    return test_collection


# ── Redis ─────────────────────────────────────────────────────────────────────
# As Redis uses asyncio, these fixtures must be function-scoped to prevent
# "Event loop is closed" errors caused by pytest-asyncio recreating the loop
# for each test function.


@pytest.fixture
async def redis_client():  # type: ignore[return]
    import redis.asyncio as aioredis
    username = os.environ.get("REDIS_USERNAME", "default")
    if username == "default" or not username:
        username = None
    client: aioredis.Redis = aioredis.Redis(  # type: ignore[type-arg]
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        username=username,
        password=os.environ.get("REDIS_PASSWORD", ""),
        decode_responses=True,
    )
    yield client
    await client.aclose()


@pytest.fixture
def cache(redis_client, config):  # type: ignore[return]
    from rag.cache.semantic_cache import SemanticCache
    return SemanticCache(redis_client, config.semantic_cache)


# ── Unique query (guarantees cache miss on first call) ────────────────────────


@pytest.fixture(scope="session")
def session_query() -> str:
    """A session-unique RAG query — guaranteed cache miss on first call."""
    return f"What is RAG? [{_SESSION_ID}]"


@pytest.fixture(scope="session")
def session_cache_query() -> str:
    """Separate unique query for cache-specific tests."""
    return f"Explain retrieval-augmented generation. [{_SESSION_ID}]"


# ── Circuit breaker + LLM Gateway ────────────────────────────────────────────


@pytest.fixture(scope="session")
def circuit_breakers(config):  # type: ignore[return]
    from rag.circuit_breaker.breaker import CircuitBreakerRegistry
    return CircuitBreakerRegistry(config.circuit_breaker)


@pytest.fixture(scope="session")
def gateway(config, circuit_breakers):  # type: ignore[return]
    from rag.gateway.gateway import LLMGateway
    return LLMGateway(config.llm_gateway, config.token_budget, circuit_breakers)


# ── RAGPipeline ───────────────────────────────────────────────────────────────


@pytest.fixture
def pipeline(config, cache, ingested_collection, gateway):  # type: ignore[return]
    from rag.serve.pipeline import RAGPipeline
    return RAGPipeline(
        config=config,
        cache=cache,
        collection=ingested_collection,
        gateway=gateway,
    )
