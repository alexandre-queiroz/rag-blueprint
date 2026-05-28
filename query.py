"""Interactive CLI for querying the RAG pipeline.

Usage:
    python query.py "What is the semantic cache escalation path?"
    python query.py                         # interactive mode (read from stdin)

Credentials are read from .env (copy .env.example → .env and fill in values).
"""
from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from rag.cache.semantic_cache import SemanticCache, get_redis_client
from rag.circuit_breaker.breaker import CircuitBreakerRegistry
from rag.config import load_config
from rag.gateway.gateway import LLMGateway
from rag.serve.pipeline import RAGPipeline
from rag.vector_db.client import get_client, get_collection


async def main(query: str) -> None:
    config = load_config()

    redis = get_redis_client()
    cache = SemanticCache(redis, config.semantic_cache)

    chroma = get_client()
    collection = get_collection(chroma, config.vector_db.collection)

    circuit_breakers = CircuitBreakerRegistry(config.circuit_breaker)
    gateway = LLMGateway(config.llm_gateway, config.token_budget, circuit_breakers)

    pipeline = RAGPipeline(
        config=config,
        cache=cache,
        collection=collection,
        gateway=gateway,
    )

    print(f"\n Query: {query}\n")
    response = await pipeline.query(query)

    # Give fire-and-forget tasks (cache write, monitor) a chance to complete
    # before the event loop closes. Not needed in long-lived servers (uvicorn).
    await asyncio.sleep(1.5)

    print(f" Answer:\n{response.answer}\n")
    if response.cached:
        print(" [cache hit]")
    else:
        print(f" Sources: {', '.join(response.sources) or '—'}")
        print(f" Complexity: {response.complexity}")
        if response.record:
            rec = response.record
            print(f" Model: {rec.model}  |  tokens: {rec.total_tokens}  |  ${rec.cost_usd:.5f}  |  {rec.latency_ms:.0f}ms")

    await redis.aclose()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        print("Enter your query (Ctrl+D to exit):")
        try:
            q = input("> ").strip()
        except EOFError:
            sys.exit(0)

    if not q:
        print("Empty query.")
        sys.exit(1)

    asyncio.run(main(q))
