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

from dataclasses import replace

from rag.cache.semantic_cache import SemanticCache, get_redis_client
from rag.circuit_breaker.breaker import CircuitBreakerRegistry
from rag.config import load_config
from rag.gateway.gateway import LLMGateway
from rag.monitoring.monitor import RAGASMonitor
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

    # Force sample_rate=1.0 in the CLI so every query is evaluated and sent to
    # Axiom — the config's 10% rate is intended for high-volume production traffic.
    monitor = RAGASMonitor(replace(config.evaluation, sample_rate=1.0))

    pipeline = RAGPipeline(
        config=config,
        cache=cache,
        collection=collection,
        gateway=gateway,
        monitor=monitor,
    )

    print(f"\n Query: {query}\n")
    response = await pipeline.query(query)

    print(f" Answer:\n{response.answer}\n")
    if response.cached:
        print(" [cache hit]")
    else:
        print(f" Sources: {', '.join(response.sources) or '—'}")
        print(f" Complexity: {response.complexity}")
        if response.record:
            rec = response.record
            print(f" Model: {rec.model}  |  tokens: {rec.total_tokens}  |  ${rec.cost_usd:.5f}  |  {rec.latency_ms:.0f}ms")

    # Drain all background tasks (cache write, RAGAS monitor) before the event
    # loop closes. In long-lived servers (uvicorn) this is not needed — the loop
    # stays alive and tasks complete naturally. In a CLI the loop exits with
    # asyncio.run(), killing any pending tasks mid-flight.
    pending = [t for t in pipeline.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

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
