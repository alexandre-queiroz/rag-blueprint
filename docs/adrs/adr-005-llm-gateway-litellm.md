# ADR-005: LLM Gateway — LiteLLM

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0

---

## Context

The system uses multiple LLM providers (Anthropic, Google, OpenAI) with complexity-aware routing and automatic fallback (FR-06, FR-08, ADR-006). Without an abstraction layer, every component that needs to call an LLM would be coupled to the specific provider SDK — swapping providers, adding fallback, or monitoring cost would become a cross-cutting change across the entire codebase.

The LLM Gateway is the single point of provider knowledge in the system. It needs to:
- **Abstract providers:** one interface, N providers
- **Model tiering:** route by complexity as configured in `config.yaml`
- **Fallback:** if a provider fails, try the next in the chain
- **Token accounting:** count tokens before and after each call to enforce budgets and calculate cost
- **Observability:** record which model responded, latency, and cost per request

The question is: build this gateway from scratch or use a library that already solves these problems?

---

## Decision

Use **LiteLLM** as the foundation of the LLM Gateway layer.

---

## Rationale

- **Unified interface:** LiteLLM exposes a single `completion()` function compatible with 100+ providers — the provider is selected by the model string (e.g. `"anthropic/claude-sonnet-4-6"`, `"gemini/gemini-2.5-flash-lite"`) without code changes
- **Native Router:** `litellm.Router` supports fallback lists per model, per-provider cooldowns, and health checks — the complexity-aware fallback logic (ADR-006) is implemented by configuring separate lists in the Router
- **Automatic token tracking:** every response includes `usage.prompt_tokens`, `usage.completion_tokens`, and `usage.total_tokens` — no manual response parsing needed
- **Battle-tested:** used in production by teams with high LLM call volume; provider edge cases (rate limits, timeouts, different error formats) have already been encountered and resolved by the library

LiteLLM provides the routing primitive; the logic for selecting the tier by complexity is ours.

---

## Alternatives Considered

### Alternative A — Custom gateway (no library)

Implement the gateway from scratch: per-provider SDK adapters, retry logic, fallback chain, token counter.

**Perceived advantages:**
- Full control over every aspect of routing and observability
- No dependency on a library with strong opinions about request/response format
- Ability to use provider-specific features (e.g. Anthropic prompt caching, Google grounding)

**Reasons to reject:**
- Maintaining adapters for 3+ provider SDKs is ongoing work — each provider changes its API periodically
- Retry logic with exponential backoff, circuit breaker, and fallback have edge cases that a custom implementation rediscovers in production
- LiteLLM already solves these problems; rewriting is reinventing without differentiated value
- Provider-specific features (prompt caching, grounding) can be accessed directly via the provider SDK when needed, without compromising the gateway

---

## Consequences

**Expected benefits:**
- Adding a new provider is adding an entry to `config.yaml` — no new adapter code
- Token accounting and estimated cost out-of-the-box on every response
- Complexity-aware fallback works via list configuration in `litellm.Router`

**Trade-offs and accepted risks:**
- LiteLLM abstracts provider-specific features — if Anthropic prompt caching or Google grounding are needed in the gateway, it is necessary to either extend LiteLLM or call the provider SDK directly for those cases
- LiteLLM versions can break compatibility — pinning the version in `pyproject.toml` is required for reproducible builds
- The abstraction adds an indirection layer — debugging provider failures requires understanding what LiteLLM does internally

**Required actions:**
- `litellm>=1.40.0` added to `pyproject.toml`
- `config.yaml` must maintain the model string format compatible with LiteLLM (`provider/model-name`)

**Triggers for reassessment:**
- Need for provider-specific features that LiteLLM does not expose or exposes with limitations
- LiteLLM ceasing active maintenance
- Air-gap or zero external dependency requirement in the gateway
