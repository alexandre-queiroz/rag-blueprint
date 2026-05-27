# ADR-006: LLM Gateway — Complexity-Aware Routing

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0

---

## Context

The fallback mechanism between providers is LiteLLM's responsibility (ADR-005). The question this ADR answers is different: **how should the LLM Gateway select which model to use for each query — both on the happy path and in fallback?**

Language models have very different quality and cost depending on the task type. Gemini 2.5 Flash Lite performs well on simple queries but degrades significantly on queries requiring deeper reasoning. Claude Sonnet delivers superior quality at any complexity, but at disproportionate cost for a trivial query.

The Classifier (the layer before the Gateway) already assigns a complexity label (`simple` / `medium` / `complex`) to each query before any model call. This label is available at routing time at no additional cost — the decision is whether to use it or ignore it.

---

## Decision

Implement **complexity-aware routing at two levels**: primary and fallback selected by tier (`simple`, `medium`, `complex`), configured in `config.yaml` under `llm_gateway.primary` and `llm_gateway.fallback`, selected based on the label assigned by the Classifier.

```
simple  → primary: Claude Haiku           → fallback: Gemini 2.5 Flash Lite
medium  → primary: Claude Sonnet          → fallback: GPT-4o-mini → Gemini 2.5 Flash
complex → primary: Claude Sonnet          → fallback: GPT-4o      → Gemini 2.5 Flash
```

---

## Rationale

- **Cost optimized on the happy path:** simple queries use the cheapest available model from the very first call — not only in fallback
- **The complexity label already exists in the pipeline** — using it is a config lookup, not additional inference
- **Avoids silent degradation:** complex queries never fall to an underpowered model, neither in primary nor in fallback
- **Declarative and auditable configuration:** changing the model for any tier requires no code — only `config.yaml`

---

## Alternatives Considered

### Alternative A — Flat fallback chain

A linear list of providers: primary → secondary → tertiary, independent of query complexity.

**Reasons to reject:**
- Forces an impossible choice for the secondary: either strong enough for complex queries (expensive) or cheap enough for simple queries (underpowered for the others)
- A single Gemini 2.5 Flash Lite as secondary would silently degrade medium and complex queries
- Wastes the complexity label that the Classifier already computed

### Alternative B — Single primary for all complexities, complexity-aware fallback only

Fixed primary at Claude Sonnet for all tiers; only the fallback is selected by complexity.

**Perceived advantages:**
- Simpler gateway implementation — primary is always the same call
- Maximum quality guaranteed on the happy path

**Reasons to reject:**
- Cost optimization only occurs in fallback — the happy path (majority of calls) remains expensive for simple queries
- A query like `"what is RAG?"` calls Claude Sonnet when Claude Haiku would resolve it at lower cost and latency
- The complexity label is already available — ignoring it in the primary leaves useful information on the table

---

## Consequences

**Expected benefits:**
- Cost reduction proportional to the volume of simple queries in real traffic
- Response quality proportional to complexity across all tiers, including the primary
- Auditable configuration changeable without deploy

**Trade-offs and accepted risks:**
- The LLM Gateway must receive the complexity label from the Classifier — data coupling between the two layers
- If the Classifier misclassifies (complex query labeled as `simple`), the primary will be underpowered; the fallback does not correct this error because the wrong label also selects the wrong fallback
- Adding a new tier requires updating both Classifier and `config.yaml`
- Circuit breakers remain per provider, not per tier — if a provider goes down, it goes down across all tiers

**Required actions:**
- LLM Gateway must receive and use the `complexity` field on every call, for selecting both primary and fallback
- `config.yaml` must maintain the structure `llm_gateway.primary.{simple,medium,complex}` and `llm_gateway.fallback.{simple,medium,complex}`

**Triggers for reassessment:**
- Empirical evidence that the Classifier misclassifies frequently (many complex queries going to Claude Haiku as primary)
- Emergence of a complexity tier not covered by the current taxonomy
- Market consolidation into few providers, making tier distinctions less relevant
