from __future__ import annotations

import hashlib
import json
import math
import os
from typing import cast

import litellm
import redis.asyncio as aioredis

from rag.config import SemanticCacheConfig
from rag.types import ComplexityLabel, RAGResponse

_EMBED_MODEL = "text-embedding-3-small"
_KEY_PREFIX = "sc:"


def get_redis_client() -> aioredis.Redis:  # type: ignore[type-arg]
    """Create async Redis client from environment variables."""
    return aioredis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        username=os.environ.get("REDIS_USERNAME", "default"),
        password=os.environ.get("REDIS_PASSWORD", ""),
        decode_responses=True,
    )


class SemanticCache:
    """Redis-backed semantic cache.

    Embeds incoming queries with text-embedding-3-small and compares them against
    all cached entries via cosine similarity. A hit at or above the configured
    threshold returns the cached response, skipping Chroma Cloud and the LLM entirely.

    Storage layout per entry (Redis hash):
        sc:{query_hash} → { query, embedding (JSON), response (JSON) }

    NOTE: lookup is O(n) via SCAN — acceptable for POC scale. Escalation path for
    production: Redis Search vector index (RediSearch), which provides sub-linear
    approximate nearest-neighbor lookup without structural changes to the interface.
    """

    def __init__(self, client: aioredis.Redis, config: SemanticCacheConfig) -> None:  # type: ignore[type-arg]
        self._redis = client
        self._config = config

    async def get(self, query: str) -> RAGResponse | None:
        """Return a cached response if a semantically similar query exists above threshold.

        Embeds the query once, then scans all cached entries to find the best
        cosine similarity match. Returns None on miss.
        """
        query_vec = await _embed(query)

        best_raw: dict[str, str] | None = None
        best_score = -1.0

        async for key in self._redis.scan_iter(f"{_KEY_PREFIX}*"):
            raw: dict[str, str] = await self._redis.hgetall(key)
            if not raw:
                continue
            cached_vec: list[float] = json.loads(raw["embedding"])
            score = _cosine_similarity(query_vec, cached_vec)
            if score > best_score:
                best_score = score
                best_raw = raw

        if best_raw is None or best_score < self._config.similarity_threshold:
            return None

        return _deserialize_response(best_raw)

    async def set(self, query: str, response: RAGResponse) -> None:
        """Embed the query and store the response in Redis with TTL."""
        query_vec = await _embed(query)
        key = f"{_KEY_PREFIX}{_query_key(query)}"

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(  # type: ignore[attr-defined]
                key,
                mapping={
                    "query": query,
                    "embedding": json.dumps(query_vec),
                    "response": json.dumps(_serialize_response(response)),
                },
            )
            pipe.expire(key, self._config.ttl_seconds)  # type: ignore[attr-defined]
            await pipe.execute()


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _embed(text: str) -> list[float]:
    """Embed a single string using text-embedding-3-small."""
    response = await litellm.aembedding(
        model=_EMBED_MODEL,
        input=[text],
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return [float(x) for x in response.data[0].embedding]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _query_key(query: str) -> str:
    """Deterministic 16-character key derived from the normalized query."""
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _serialize_response(response: RAGResponse) -> dict[str, object]:
    """Serialize the fields needed to reconstruct a cache-hit RAGResponse."""
    return {
        "answer": response.answer,
        "sources": response.sources,
        "complexity": response.complexity,
    }


def _deserialize_response(raw: dict[str, str]) -> RAGResponse:
    """Reconstruct a RAGResponse from a Redis hash. record is None — no LLM was called."""
    data: dict[str, object] = json.loads(raw["response"])
    return RAGResponse(
        answer=str(data["answer"]),
        sources=list(cast(list[str], data["sources"])),
        complexity=cast(ComplexityLabel, data["complexity"]),
        record=None,
        cached=True,
    )
