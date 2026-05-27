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

_KEY_PREFIX = "sc:"

# API key env var resolved from the embedding model prefix at call time.
_EMBED_API_KEY_ENV: dict[str, str] = {
    "gemini/": "GOOGLE_API_KEY",
    "google/": "GOOGLE_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "text-embedding": "OPENAI_API_KEY",  # OpenAI shorthand (no prefix)
}


def get_redis_client() -> aioredis.Redis:  # type: ignore[type-arg]
    """Create async Redis client from environment variables.

    If username is 'default', we pass None to use legacy AUTH (no username),
    as Redis Cloud free-tier databases do not support ACL-style auth.
    """
    username = os.environ.get("REDIS_USERNAME", "default")
    if username == "default" or not username:
        username = None
    return aioredis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        username=username,
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

    KNOWN LIMITATION — O(n) latency under load:
        Each `get()` call does a full SCAN of all cache keys via the network and
        embeds each stored vector into memory for comparison. At low entry counts
        (< ~10k) this is imperceptible. At scale, SCAN latency grows linearly with
        the number of cached entries, adding dozens of ms per request.

        Escalation path (no interface change required):
        Enable RediSearch and create a vector index on the `embedding` field.
        Replace the SCAN loop with a single `FT.SEARCH … KNN` query — O(log n)
        approximate nearest-neighbour, single round-trip, native to Redis Stack.
    """

    def __init__(self, client: aioredis.Redis, config: SemanticCacheConfig) -> None:  # type: ignore[type-arg]
        self._redis = client
        self._config = config

    async def get(self, query: str) -> RAGResponse | None:
        """Return a cached response if a semantically similar query exists above threshold.

        Embeds the query once, then scans all cached entries to find the best
        cosine similarity match. Returns None on miss.
        """
        query_vec = await _embed(query, self._config.embedding_model)

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
        query_vec = await _embed(query, self._config.embedding_model)
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


async def _embed(text: str, model: str) -> list[float]:
    """Embed a single string using the configured embedding model.

    The API key is resolved from the model prefix via _EMBED_API_KEY_ENV —
    no hardcoded provider assumption. Supports Google (gemini/) and OpenAI.
    """
    api_key_env = next(
        (env for prefix, env in _EMBED_API_KEY_ENV.items() if model.startswith(prefix)),
        "OPENAI_API_KEY",
    )
    response = await litellm.aembedding(
        model=model,
        input=[text],
        api_key=os.environ.get(api_key_env, ""),
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
