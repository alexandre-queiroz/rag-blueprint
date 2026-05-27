# Architecture — rag-production

## Overview

Six-layer pipeline. Each layer is independently testable and has a single responsibility.

![Architecture](architecture.png)

> To update the diagram: open [`architecture.excalidraw`](architecture.excalidraw) at [excalidraw.com](https://excalidraw.com) and export as PNG.

## Layers

### 1. Ingestion Pipeline

**Responsibility:** transform raw documents into indexed, searchable chunks.

Steps:
1. Load document (PDF, TXT, MD)
2. Apply chunking strategy — fixed-size or hierarchical (parent + child chunks), configured per collection. Chroma Cloud enforces a 16 KiB limit per document; chunking is mandatory for longer documents.
3. Call `collection.add()` — Chroma Cloud generates dense (Qwen) and sparse (Splade) embeddings server-side. No external embedding API call required.
4. Store with metadata: `source` (document ID) and `chunk_index` — required for GroupBy deduplication in search results.

The chunking strategy is the single most impactful decision for retrieval quality. Hierarchical chunking stores a large parent chunk for context and smaller child chunks for precise matching — retrieval finds the child, returns the parent.

#### Parent-Child Denormalization Trade-off

Currently, the system uses a **denormalized storage approach** in Chroma Cloud: the `parent_text` is stored directly inside the metadata of each child chunk.

*   **Pros:**
    *   **Ultra-low Latency (1 Network Call):** A single search query in Chroma Cloud retrieves the child matches along with their fully populated `parent_text` metadata, requiring exactly one remote HTTP round-trip in the critical serving path.
    *   **No Unnecessary Vectors:** Storing parents in metadata avoids creating a second vector collection in Chroma. If parents were stored in a separate collection, Chroma Cloud would unnecessarily generate embeddings (Qwen/Splade) and maintain HNSW vector indexes for parent chunks that are never searched semantically.
    *   **Transactional Simplicity:** Avoiding a secondary collection eliminates synchronization overhead (rollbacks, deletion cascades, and transactional drift) during ingest or purge operations.
*   **Cons:**
    *   **Storage Redundancy:** Replicating the parent text across all its children results in a ~4x to 5x increase in text metadata storage footprint. At massive scale (millions of documents), this increases metadata storage costs.

#### Scaling Roadmap (Escalation Path)

If database size or metadata storage costs in Chroma Cloud eventually scale to a point of concern, the escalation path is **not** to normalize into two Chroma collections. Instead, the architecture will evolve as follows:
1.  **Chroma Cloud (Vectors & IDs only):** Index the child chunks in Chroma Cloud, storing only `parent_id` (and not `parent_text`) in metadata.
2.  **PostgreSQL/Redis (Normalized Storage):** Store the mapping of `parent_id` to `parent_text` in a relational table (PostgreSQL) or highly optimized in-memory store (Redis), which are already native components of the project's technology stack.
3.  **In-Memory Resolution:** During retrieval, execute a single query to Chroma Cloud to fetch the best child chunk IDs and `parent_id`s, then run a fast bulk lookup (`SELECT WHERE IN` or Redis MGET) to resolve the parent texts. Since relational/KV storage is significantly cheaper than vector DB metadata memory, this normalizes storage footprint without introducing secondary vector indexing overhead, maintaining sub-millisecond local joins.


**Offline evaluation** (RAGAS) runs after ingestion against a fixed Q&A dataset before any serving starts. See ADR-008.

### 2. API Gateway

**Responsibility:** request validation, rate limiting, token budget enforcement.

- Rejects requests that exceed `token_budget.max_tokens_per_request` before any downstream call
- Entry point for all observability metadata (request ID, timestamp, session ID)
- Token budget is enforced per request only — see ADR-009 for the escalation path to org/department/user

### 3. Classifier

**Responsibility:** assign a complexity label (`simple` / `medium` / `complex`) to the incoming query.

Pipeline (in order, stops at first confident result):
1. Heuristics — keyword matching, query length, entity count (~70% of queries resolved here, zero cost)
2. Embedding similarity — cosine distance against labeled canonical examples (~20% of remaining)
3. LLM call — Gemini 2.5 Flash Lite with structured output for ambiguous cases (~10%)

The complexity label is passed downstream to the LLM Gateway to select the correct fallback chain. See ADR-002.

### 4. DB Search

**Responsibility:** retrieve the most relevant chunks for the query.

Steps:
1. Check semantic cache (Redis) — if similarity ≥ `semantic_cache.similarity_threshold`, return cached response immediately and skip steps 2–3
2. Execute hybrid search via Chroma Cloud native RRF — dense (Qwen) and sparse (Splade) rankings fused with weights `dense: 2.0 / sparse: 1.0`, returning top `vector_db.top_k` chunks
3. Return chunks to LLM Gateway as context

There is no manual BM25 implementation or cross-encoder reranker. The Chroma Cloud RRF handles fusion natively. If `context_precision` degrades in production (measured via RAGAS), adding a cross-encoder reranker after step 2 is the escalation path without structural changes. See ADR-005.

### 5. LLM Gateway

**Responsibility:** model selection, token accounting, prompt assembly, response return.

Flow:
1. Receive query + retrieved chunks + complexity label
2. Count estimated tokens — reject if over `token_budget.max_tokens_per_request`
3. Call primary model (Claude Sonnet)
4. On failure → consult circuit breaker state for each provider in the fallback list for the given complexity tier
5. Call first available fallback; if all fail → return graceful degradation response
6. Record: tokens consumed, cost estimate, model used, latency

Provider configuration and fallback chains per complexity tier live in `config.yaml`. The Gateway is the only component that knows about providers — all other layers call `gateway.complete(prompt, complexity)`. See ADR-002.

### 6. Circuit Breaker

**Responsibility:** track per-provider health and prevent calls to failing providers.

States per provider:
- **Closed** — normal operation, all calls pass through
- **Open** — opened after `circuit_breaker.failure_threshold` consecutive failures; calls are immediately rejected and the next provider in the fallback chain is tried
- **Half-open** — after `circuit_breaker.recovery_timeout_ms`, allows `circuit_breaker.half_open_max_calls` test call(s); success closes the circuit, failure re-opens it

Circuit breakers are provider-scoped, not model-scoped. A provider flagged as open is skipped across all complexity tiers.

### 7. Monitoring (Online)

**Responsibility:** continuously measure RAG quality on live traffic.

- `evaluation.sample_rate` (default 10%) of requests are asynchronously evaluated with RAGAS after response is returned to the caller
- Metrics collected: faithfulness, answer_relevancy, context_precision
- Alert emitted when any metric drops below `evaluation.min_score_threshold`
- Chunk hit rate tracked per document — chunks never retrieved indicate ingestion or chunking problems

## Key Architectural Constraints

1. **LLM Gateway is the only provider-aware component.** No other layer imports a provider SDK directly.
2. **Complexity label flows downstream, never upstream.** Classifier → Gateway only.
3. **Semantic cache is checked before DB Search** — a cache hit skips Chroma Cloud entirely.
4. **Ingestion is offline.** It runs as a separate process, not in the serving path.
5. **RAGAS evaluation is async in production.** It must not block the response to the caller.
6. **Chroma Cloud handles all vector DB embeddings.** No external embedding API is called in the ingestion or search path. `text-embedding-3-small` is used only by the semantic cache (Redis).

## Known Limitations & Escalation Paths

These trade-offs are accepted for the current POC scope. Each has a documented escalation path that requires no structural change to the interface it affects.

### 1. Gateway — silent provider failures (observability)

**Location:** `src/rag/gateway/gateway.py` — `LLMGateway.complete()`

**Issue:** The original `except Exception: continue` pattern swallowed all provider error information, making it impossible to distinguish a network timeout from an invalid API key from a model overload in logs or traces.

**Fix applied:** Exceptions are now logged at `WARNING` level with provider name, model, exception type, and full traceback before the fallback chain continues. The behavior (try next provider) is unchanged.

**Residual gap:** The log event is unstructured `stderr` — invisible in Axiom until the OpenTelemetry integration described in ADR-010 is wired up. Until then, errors are observable locally but not queryable in production dashboards.

---

### 2. Semantic Cache — O(n) lookup latency

**Location:** `src/rag/cache/semantic_cache.py` — `SemanticCache.get()`

**Issue:** Every cache lookup performs a full Redis SCAN of all `sc:*` keys followed by a cosine similarity comparison in Python. Round-trip count and Python-side compute grow linearly with the number of cached entries.

**Acceptable at POC scale:** below ~10 000 entries, SCAN completes in < 5 ms. Beyond that, added latency degrades the serving path.

**Escalation path:** Enable RediSearch and create a vector index on the `embedding` field. Replace the SCAN loop with a single `FT.SEARCH … KNN` query — O(log n) approximate nearest-neighbour, one network round-trip. The `SemanticCache` interface (`get`/`set`) remains unchanged.

---

### 3. Pipeline — fire-and-forget task lifetime in serverless

**Location:** `src/rag/serve/pipeline.py` — `RAGPipeline.query()`

**Issue:** `asyncio.create_task()` schedules cache writes and RAGAS monitor calls after the response is returned. In long-lived processes (uvicorn, gunicorn) the event loop continues running after the handler returns, so tasks complete normally. In ephemeral environments (AWS Lambda, Cloud Run scale-to-zero, Vercel) the process may be frozen or terminated before background tasks finish, causing silent cache misses and lost monitoring events.

**Escalation path:** Replace `create_task()` with a synchronous write to a durable queue (Redis Streams, SQS, Cloud Tasks). The queue write adds < 2 ms to the request; a separate worker process drains the queue and performs cache writes and RAGAS evaluations outside the request lifecycle.

---

### 4. Circuit Breaker — in-memory state, no fleet-wide coordination

**Location:** `src/rag/circuit_breaker/breaker.py` — `CircuitBreakerRegistry`

**Issue:** Breaker state (open / closed / half-open) is stored per-process. In a horizontally scaled deployment each instance maintains independent state — a provider can be Open in one pod and Closed in another simultaneously. The effective failure threshold before a provider is blocked fleet-wide is `failure_threshold × instance_count`, not `failure_threshold`.

**Acceptable at POC scale:** single-process deployment has no fleet coordination problem.

**Escalation path:** Store breaker counters and state in Redis (already in the stack) using atomic `INCR` and `EXPIRE` commands. A state change in any instance propagates to all others within the next request cycle. The `CircuitBreakerRegistry` interface (`is_available`, `get`, `state`) remains unchanged — only the storage backend changes.

---

## ADR Index

| ADR | Decision |
|---|---|
| [ADR-001](adrs/adr-001-vector-db.md) | Vector Database — Chroma Cloud |
| [ADR-002](adrs/adr-002-chroma-cloud-embeddings.md) | Embeddings — Chroma Cloud Native (Qwen + Splade) |
| [ADR-003](adrs/adr-003-no-framework.md) | Orchestration — No RAG Framework (LangChain / LlamaIndex) |
| [ADR-004](adrs/adr-004-semantic-cache.md) | Semantic Cache — Redis + text-embedding-3-small |
| [ADR-005](adrs/adr-005-llm-gateway-litellm.md) | LLM Gateway — LiteLLM |
| [ADR-006](adrs/adr-006-llm-gateway-routing.md) | LLM Gateway — Complexity-Aware Routing |
| [ADR-007](adrs/adr-007-circuit-breaker-pybreaker.md) | Circuit Breaker — pybreaker |
| [ADR-008](adrs/adr-008-evaluation.md) | Evaluation Framework — RAGAS |
| [ADR-009](adrs/adr-009-token-budget.md) | Token Budget — Per Request |
| [ADR-010](adrs/adr-010-observability-axiom.md) | Observability and Monitoring — Axiom |
