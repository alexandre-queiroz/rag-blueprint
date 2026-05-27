# ADR-003: Orchestration — No RAG Framework (LangChain / LlamaIndex)

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0

---

## Context

Frameworks like LangChain and LlamaIndex emerged to abstract recurring patterns in LLM applications: document loading, chunking, embedding generation, vector store integration, retrieval, and chain orchestration. They significantly reduce time to a working prototype.

The architectural question is whether this abstraction serves or hides the system design.

**LlamaIndex** is focused on the data layer: ingestion, indexing strategies, retrieval pipelines, and query engines. It covers exactly the Ingestion and DB Search layers of this system.

**LangChain** is focused on orchestration: chains, agents, memory, and tool use. It covers the orchestration layer between the Classifier, Gateway, and retrieval layers.

Both are production-grade and widely adopted. The decision not to use them requires justification.

---

## Decision

**Do not use** LangChain or LlamaIndex. Each system layer is implemented explicitly with its responsibilities and connections documented.

---

## Rationale

This project is a reference implementation — the primary deliverable is the architecture, not the system itself. Each layer (ingestion, classifier, DB search, LLM gateway, circuit breaker, monitoring) exists to be understood independently, with explicit documentation of why it exists and how it connects to the others.

RAG frameworks abstract exactly the concepts this project aims to make visible. When LlamaIndex's `RetrieverQueryEngine` executes, the chunking strategy, retrieval logic, reranking, and prompt assembly are opaque to the caller. That opacity is valuable for product teams that need to ship fast — it is counterproductive in a reference implementation whose goal is to teach the design.

Additionally, Chroma Cloud's native hybrid search (RRF with Qwen + Splade) has no direct mapping in LlamaIndex's retrieval abstractions, which were designed for the traditional embed-then-search model. Forcing this integration would add complexity without benefit.

---

## Alternatives Considered

### Alternative A — LlamaIndex for data layers

Use LlamaIndex for Ingestion (`SimpleDirectoryReader`, `HierarchicalNodeParser`) and DB Search (`VectorStoreIndex`, `RetrieverQueryEngine`).

**Perceived advantages:**
- Ready-made and tested chunking strategies (including hierarchical)
- Chroma integration via `ChromaVectorStore`
- Reduces boilerplate code in the data layers

**Reasons to reject:**
- Hides the chunking and retrieval logic that is central to the proposed learning
- The integration with Chroma Cloud (schema with Splade, native RRF) is not supported out-of-the-box by LlamaIndex's `ChromaVectorStore` — it would require significant overrides, negating the benefit of abstraction
- Makes it harder for readers to understand what happens at each step

### Alternative B — LangChain for orchestration

Use LangChain to connect Classifier → DB Search → LLM Gateway via chains and agents.

**Perceived advantages:**
- Prompt management and conversation memory abstraction
- Large ecosystem of integrations and examples

**Reasons to reject:**
- LangChain has a history of abstractions that increase complexity rather than reduce it — the debugging curve for nested chains is steep
- The orchestration in this system is simple enough to not need a chains framework
- Hides the data flow between layers that should be explicit

---

## Consequences

**Expected benefits:**
- Each layer is readable, testable, and documented independently
- Data flow between layers is explicit in the code — no framework magic
- Contributors understand the system by reading the code, without needing to learn a framework's abstractions

**Trade-offs and accepted risks:**
- Every layer this project implements explicitly is a layer a team using LlamaIndex gets for free — including edge case handling, active maintenance, and community support
- More code to maintain in the long run
- Explicit implementation prioritizes readability over delivery speed

**Required actions:**
- None — the absence of a framework is the decision

**Triggers for reassessment:**
- **The project evolving from case study to real product:** LangChain and LlamaIndex are market standards with broad adoption, mature ecosystems, and active maintenance. In a product that needs delivery speed, reduced boilerplate, and community support, the absence of a framework becomes a liability. In that scenario, LlamaIndex for the data layer (ingestion + retrieval) and LangChain for orchestration are the recommended choices — not using frameworks in production requires solid technical justification, not just preference
- LlamaIndex releasing native support for Chroma Cloud schema with Splade, making the integration trivial without losing control over retrieval
