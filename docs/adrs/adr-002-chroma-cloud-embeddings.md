# ADR-002: Embeddings — Chroma Cloud Native (Qwen + Splade)

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0
- **Complements:** ADR-001 (Vector Database — Chroma Cloud)

---

## Context

The original architecture planned the following retrieval pipeline:

1. Generate query embedding via `text-embedding-3-small` (OpenAI API)
2. Semantic search in Chroma using the generated embedding
3. Parallel lexical search via BM25 implemented in the application
4. Merge the two rankings via RRF implemented in the application
5. Refine with a cross-encoder reranker

With the adoption of Chroma Cloud (ADR-001), the service provides its own embedding models executed server-side and native RRF as a search primitive. This fundamentally changes the trade-off.

Hybrid search — combining vector search (semantic) with sparse search (keyword matching) — is today the standard for quality retrieval in production RAG systems. Pure vector search loses on queries with specific technical terms; pure lexical search loses on conceptual queries. RRF merges the rankings robustly without requiring score tuning.

---

## Decision

Use **Chroma Cloud native embeddings** for indexing and search:
- **Dense (semantic):** `ChromaCloudQwenEmbeddingFunction` — model `QWEN3_EMBEDDING_0p6B`, task `retrieval`
- **Sparse (keyword):** `ChromaCloudSpladeEmbeddingFunction` — model `SPLADE_PP_EN_V1`
- **Fusion:** Chroma Cloud native `Rrf` with weights `dense: 2.0 / sparse: 1.0`

---

## Rationale

- **Elimination of responsibilities:** the pipeline goes from 5 steps to 1 call — `collection.search()` executes embedding, vector search, sparse search, and RRF fusion internally
- **No external call in the critical path:** no embedding API is called by the application during ingestion or search — reduces latency and removes a failure point
- **Qwen and Splade are validated models:** Qwen3 is a high-quality multilingual embedding model; Splade is a reference in sparse retrieval
- **RRF with `dense: 2.0` weight:** semantics prevail over keyword matching, configurable without code changes

> No comparative retrieval quality benchmark was conducted between this approach and the alternative with `text-embedding-3-small` + manual BM25. The decision is pragmatic for the scope of a POC.

---

## Alternatives Considered

### Alternative A — `text-embedding-3-small` (OpenAI) + manual BM25 + application-side RRF

Keep the original pipeline: generate embeddings via OpenAI, implement BM25 in the application, merge with custom RRF.

**Perceived advantages:**
- No coupling to Chroma Cloud — works with any vector database
- Full control over the embedding model, BM25, and RRF weights
- `text-embedding-3-small` has public benchmarks and well-documented behavior

**Reasons to reject:**
- Requires maintaining BM25 and RRF as application code — failure surface and maintenance without differentiated value
- Adds an OpenAI API call on every query — extra cost, latency, and dependency
- The Qwen + Splade + native RRF combination delivers equivalent results with fewer moving parts

### Alternative B — Cross-encoder reranker after hybrid search

Keep Chroma's native hybrid search and add a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to refine the top-K results before sending to the LLM.

**Perceived advantages:**
- Cross-encoders are more precise than bi-encoders for reranking — can improve `context_precision`
- Complements RRF without replacing it

**Reasons to reject:**
- Adds latency (~200ms for lightweight models) in the critical serving path
- For this POC, RRF with calibrated weights delivers sufficient quality
- This is the natural escalation if `context_precision` degrades in production — no need to include it now

---

## Consequences

**Expected benefits:**
- DB Search layer reduced from 5 steps to 3 (cache check → `collection.search()` → return chunks)
- Less code to maintain: no BM25, no manual RRF, no external embedding call
- Hybrid search out-of-the-box without implementation

**Trade-offs and accepted risks:**
- **Coupling to Chroma Cloud:** migrating to another vector DB requires reimplementing the embedding pipeline and hybrid search
- **Opaque embedding cost:** Chroma Cloud charges per indexed document and per query — no long-term cost benchmark
- **`text-embedding-3-small`** remains in use only for the semantic cache (Redis) — two embedding models in the system with distinct purposes
- **No separate reranking step:** `rerank_top_k` was removed from `config.yaml`

**Required actions:**
- Collection must be created with a `Schema` that defines the sparse index (Splade) — without this, only dense search works
- Metadata `source` and `chunk_index` must be included in every indexed document to enable GroupBy and deduplication

**Triggers for reassessment:**
- `context_precision` (RAGAS) consistently below threshold in production → add cross-encoder reranker (Alternative B)
- Chroma Cloud embedding cost at scale exceeds cost of `text-embedding-3-small` + custom BM25
- Migration to a different vector DB → reimplementing the embedding pipeline is required
