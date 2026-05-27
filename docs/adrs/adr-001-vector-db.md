# ADR-001: Vector Database — Chroma Cloud

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0

---

## Context

The system needs a vector database to store document embeddings and perform semantic search (FR-01, FR-05). The choice directly impacts infrastructure complexity, cost model, retrieval quality, and ease of future migration.

This project is a reference POC — the focus must be on the RAG architecture, not on infrastructure operations. Any time spent configuring and maintaining a database is time subtracted from learning about the layers that actually matter.

> No formal FinOps or performance benchmark was conducted between the options. Cost comparisons are directional, not measured.

---

## Decision

Adopt **Chroma Cloud** (trychroma.com) as the project's vector database.

---

## Rationale

- **Zero infrastructure setup:** no Docker, no cluster, no persistent volumes to manage
- **Native server-side embeddings:** Chroma Cloud generates dense (Qwen) and sparse (Splade) embeddings on its own servers during `add()` and `search()` — no external embedding API call is needed in the ingestion or search path
- **Native hybrid search via RRF:** eliminates the need to implement BM25 manually and merge rankings in the application
- **Same API as open-source Chroma:** no lock-in at the code level — migrating to self-hosted means replacing the client initialization
- **$5 free tier:** sufficient to validate all project functionality at no cost
- **SOC 2 Type II:** relevant to demonstrate that the choice would be defensible in production

---

## Alternatives Considered

### Alternative A — Self-Hosted Chroma (Docker / Kubernetes)

Run the open-source Chroma server on own infrastructure.

**Perceived advantages:**
- Full data sovereignty — nothing leaves the internal network
- Predictable cost (infrastructure only, no usage-based billing)
- Full control over version and configuration
- Ability to use custom embedding models

**Reasons to reject:**
- Requires infrastructure setup and maintenance (Docker Compose, persistent volumes, backups)
- Chroma Cloud's native embeddings (Qwen, Splade) **are not available** in self-hosted — an external embedding pipeline would be needed (e.g. OpenAI API or local model via Ollama), adding dependency and complexity
- Horizontal scaling is non-trivial; Chroma's distributed mode requires operational expertise
- Diverts POC focus toward infrastructure problems

### Alternative B — Alternative providers (Pinecone, Qdrant Cloud, pgvector)

Other managed vector databases available in the market.

**Perceived advantages:**
- Pinecone and Qdrant Cloud have public performance benchmarks and a longer production track record
- pgvector reuses existing Postgres infrastructure if the team already operates that database
- Mature ecosystems with established SDKs and documentation

**Reasons to reject:**
- Different APIs — migrating from Chroma to Pinecone requires rewriting `src/rag/vector_db/`, which is the only coupling point with Chroma in the codebase. The rest of the system consumes `hybrid_search()` without knowing the provider
- pgvector degrades on pure vector search workloads at scale compared to specialized databases
- None of these options offer native embeddings equivalent to Chroma Cloud's Qwen + Splade without additional integration
- For a POC, setup cost exceeds any unmeasured quality gain

---

## Consequences

**Expected benefits:**
- Setup in minutes: create account, copy credentials to `.env`, connect
- Simplified DB Search layer: hybrid search + RRF in a single Chroma Cloud call
- No external embedding API dependency in the critical search path

**Trade-offs and accepted risks:**
- Dependency on trychroma.com availability — if the service goes down, retrieval stops
- Data transits through Chroma Cloud infrastructure (mitigated by SOC 2 and VPC options, but relevant for regulated data)
- Native embeddings (Qwen + Splade) create coupling — migrating to another provider requires reimplementing the embedding pipeline and hybrid search
- Cost at scale is unknown without a benchmark

**Required actions:**
- `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE` variables must be configured (see `.env.example`)
- The collection must be created with a `Schema` that defines the sparse index (Splade) — specific to Chroma Cloud
- To run locally without Chroma Cloud: replace `CloudClient` with the embedded client, remove the `Schema`, and add a custom embedding pipeline

**Triggers for reassessment:**
- Data sovereignty requirement (data cannot leave own infrastructure)
- Chroma Cloud cost at real scale exceeds alternatives after benchmarking
- Need for features unavailable in Chroma Cloud (e.g. advanced metadata filtering with SQL)
- Before any production deploy, a new ADR must be written based on real load and FinOps data
