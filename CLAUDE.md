# CLAUDE.md — rag-production

This file provides context for AI coding assistants (Claude, Gemini, etc.) working on this project.

## What this project is

A production-grade RAG (Retrieval-Augmented Generation) reference implementation covering the full system design: from document ingestion to multi-model fallback. Built as a learning resource — every architectural decision is documented and testable in isolation.

## Architecture

Full diagram and layer descriptions: [`docs/architecture.md`](docs/architecture.md).

| Layer | Responsibility |
|---|---|
| Ingestion | Chunking, then `collection.add()` — Chroma Cloud generates dense + sparse embeddings server-side |
| API Gateway | Request validation, token budget enforcement per request (ADR-009) |
| Classifier | Assigns complexity (`simple` / `medium` / `complex`) — three-stage pipeline: heuristics → embedding similarity → LLM |
| Semantic Cache | Redis similarity lookup before DB search — hit skips Chroma + LLM entirely |
| DB Search | Hybrid search via Chroma Cloud native RRF (Qwen dense + Splade sparse) |
| LLM Gateway | Model selection by complexity tier, fallback orchestration, token accounting (ADR-006) |
| Circuit Breaker | Per-provider failure tracking — open circuit skips to next provider in chain |
| Monitoring | Async RAGAS sampling on 10% of live traffic |

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ (strict typing — no `Any`) |
| Vector DB | Chroma Cloud — embeddings e hybrid search gerenciados server-side, transparentes para a aplicação |
| Semantic cache | Redis Cloud + `gemini/gemini-embedding-001` (Google) |
| Evaluation | RAGAS |
| LLM — simple | Gemini 2.5 Flash Lite (primary) → Claude Haiku (fallback) |
| LLM — medium | Gemini 2.5 Flash Lite (primary) → Claude Sonnet → GPT-4o-mini (fallback) |
| LLM — complex | Gemini 2.5 Flash (primary) → Claude Sonnet → GPT-4o (fallback) |
| Classifier LLM | Gemini 2.5 Flash Lite |

## Configuration

All provider, circuit breaker, budget, and cache settings live in [`config.yaml`](config.yaml) at the project root. Read it before touching any provider-related code.

In production, these values should be managed by a config service (AWS AppConfig, LaunchDarkly, etc.) to allow runtime changes without redeploy.

Copy `.env.example` to `.env` and fill in your credentials before running anything.

## Code Conventions

- **Typing**: strict — no `Any`, no untyped parameters
- **Naming**: files and folders in `lowercase-kebab-case`; Python modules use `snake_case` (hyphens are not valid identifiers)
- **Code and schemas**: English
- **Documentation**: English
- **Comments**: only when the *why* is non-obvious — never explain what the code does
- **Abstractions**: prefer readable code over clever abstractions (AHA over DRY)
- **Error handling**: only at system boundaries (user input, external APIs) — trust internal contracts

## Project Structure

```
rag-production/
├── src/rag/
│   ├── ingestion/         # Chunking strategies + Chroma Cloud indexing
│   ├── classifier/        # Three-stage complexity classification pipeline
│   ├── cache/             # Semantic cache (Redis + gemini/text-embedding-004)
│   ├── gateway/           # LLM Gateway — model selection, fallback, token accounting
│   ├── circuit_breaker/   # Per-provider circuit breaker (pybreaker)
│   ├── monitoring/        # Async RAGAS sampling on live traffic
│   ├── vector_db/         # Chroma Cloud client and hybrid search
│   ├── config.py          # Typed config loader (reads config.yaml)
│   └── types.py           # Shared types (ComplexityLabel, RAGResponse, etc.)
├── docs/
│   ├── adrs/              # Architecture Decision Records — read before implementing
│   ├── architecture.md    # Full system design documentation
│   ├── architecture.png   # Architecture diagram
│   └── architecture.excalidraw
├── config.yaml            # System configuration (project root)
├── .env.example           # Required environment variables
├── pyproject.toml
└── CLAUDE.md
```

## Before Implementing Anything

1. Read `docs/adrs/` — decisions already made should not be reversed without a new ADR
2. Read `config.yaml` — understand the provider and complexity-tier configuration
3. Check `requirements.md` for the FR/NFR the feature must satisfy

## Adding a New Provider or Changing Model Tiers

1. Update `config.yaml` with the new provider entry under the relevant complexity tier
2. Create an ADR in `docs/adrs/` explaining the reason for the change
3. The LLM Gateway must remain the single point of provider knowledge — no other layer imports a provider SDK directly

## Running Evaluations

RAGAS is used for both offline evaluation (pre-deploy) and online sampling (production monitoring). Minimum score thresholds and sample rate are defined in `config.yaml` under `evaluation`.
