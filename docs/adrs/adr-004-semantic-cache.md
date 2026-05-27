# ADR-004: Semantic Cache — Redis + text-embedding-3-small

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0

---

## Context

LLM calls are the primary operational cost of the system. A significant fraction of queries received in production are semantically equivalent — same intent, slightly different wording. Without a cache, each query triggers an independent model call regardless of having already answered something identical before.

The semantic cache solves this: instead of comparing exact strings (traditional cache), it compares semantic similarity between the new query and previously cached queries. If similarity exceeds a configurable threshold, the cached response is returned without calling the model.

Two relevant constraints for the implementation:
1. Chroma Cloud embeddings (Qwen + Splade) are generated server-side within collection operations — they cannot be invoked independently for use in Redis
2. The cache must be controllable: similarity threshold, TTL, and eviction policy must be auditable and adjustable

---

## Decision

Implement the semantic cache with **Redis Cloud** as the backend and **`text-embedding-3-small`** (OpenAI) as the embedding model for queries.

---

## Rationale

- **Explicit control:** similarity threshold, TTL, and eviction configurable in `config.yaml` — no opaque library behavior
- **Independence from Chroma Cloud:** Qwen + Splade embeddings are server-side and coupled to collection operations (`add()`, `search()`) — a standalone embedding model is required for Redis cache lookups
- **Isolated testability:** cache is a pure function (embed query → Redis lookup → hit or miss) testable without LLM or Chroma dependencies
- **`text-embedding-3-small`** is fast (~1ms), cheap (sub-cent per query), and sufficient for query semantic similarity

---

## Alternatives Considered

### Alternative A — GPTCache

GPTCache is the market standard for semantic caching in LLM applications. It abstracts the backend, embedding model, and similarity calculation into a single integration.

**Perceived advantages:**
- Ready-made abstraction: less code to write and maintain
- Supports multiple backends (Redis, SQLite, etc.) and configurable embedding models
- Direct integration with LangChain, LlamaIndex, and other frameworks
- **Ideal when:** the team needs a functional semantic cache quickly, without the need for fine control over threshold or eviction — early-stage product, small team, or when the maintenance cost of a custom solution outweighs the benefit of control

**Reasons to reject for this project:**
- Opaque abstraction: the similarity threshold, embedding model, and eviction policy are controlled internally by the library, not by `config.yaml`
- For a system where the threshold directly affects cost and response quality, losing this control is unacceptable
- **Ideal for custom (this choice) when:** the system operates at scale where small threshold variations have measurable cost impact, or when the cache invalidation policy needs to be coupled to business events (e.g. document re-ingestion)

### Alternative B — Reuse Chroma Cloud embeddings

Use the same Qwen or Splade models to embed cache queries.

**Perceived advantages:**
- Consistency: same vector space for retrieval and cache
- No additional embedding API dependency

**Reasons to reject:**
- Chroma Cloud embeddings are server-side and generated only within collection operations (`add()`, `search()`) — there is no endpoint to invoke the model independently
- A dummy collection in Chroma would be needed just to generate cache embeddings, which is an unsustainable architectural hack

---

## Consequences

**Expected benefits:**
- Cost reduction proportional to cache hit rate — semantically repeated queries never reach the LLM
- p95 latency of cached queries drops to ~5ms (Redis lookup) instead of 3–8s (LLM call)

**Trade-offs and accepted risks:**
- Every query reaching the system makes an OpenAI API call to generate the cache embedding — dependency on OpenAI availability even when the primary LLM provider is Anthropic
- If OpenAI is unavailable, the cache fails and all queries pass directly to Chroma + LLM Gateway (graceful degradation, no service interruption)
- Threshold `0.92` is a starting point — too low generates false hits (wrong responses for different queries), too high makes the cache ineffective

**Required actions:**
- Redis Cloud must be available as a service — credentials configured in `.env`
- `OPENAI_API_KEY` is required even if the LLM Gateway uses another provider as primary
- Similarity threshold must be calibrated with real traffic data before being used as a definitive criterion

**Triggers for reassessment:**
- Cache false positive rate (incorrect responses due to high similarity between distinct queries) exceeding product tolerance
- Emergence of a standalone embedding model compatible with Chroma Cloud that eliminates the OpenAI dependency for the cache
- Cost of the embedding call for the cache exceeding the cost saved by hits
