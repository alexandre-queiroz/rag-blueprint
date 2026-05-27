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

**Offline evaluation** (RAGAS) runs after ingestion against a fixed Q&A dataset before any serving starts. See ADR-004.

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
