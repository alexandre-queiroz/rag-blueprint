# ADR-011: Data Governance — Including Query and Answer in Axiom Spans

- **Status:** Accepted (supersedes initial draft that excluded these fields)
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 2.0

---

## Context

The RAGAS monitoring layer emits one OpenTelemetry span to Axiom per sampled request. The span captures quality scores (faithfulness, answer_relevancy, context_precision) and operational metadata (model, latency, cost).

During live testing, one span was emitted without `rag.query` and `rag.answer`. That span returned `answer_relevancy: 0` and `context_precision: 0` alongside `faithfulness: 1.0` — a combination that signals evaluator instability rather than a real system failure. Without the query in the span, it was impossible to confirm the hypothesis: we could not see which query triggered the evaluation, what the answer contained, or reproduce the exact input.

Debugging RAGAS false positives requires seeing the query and the answer. Without them, a low-score alert demands manual reproduction from scratch — slow and impractical for sampled traffic.

### LGPD consideration

Axiom is a US-based SaaS. Storing user query content in a foreign system without a Data Processing Agreement (DPA) and without explicit user consent creates a compliance risk under **LGPD Art. 33**, which governs international transfers of personal data.

This concern is real and applies in production. It does not apply to this system in its current state: the indexed corpus is architecture documentation, queries are about the system itself, and there are no real users or personal data in the pipeline.

---

## Decision

**Include `rag.query` and `rag.answer` in the Axiom span.**

| Field | Value |
|---|---|
| `rag.query` | The user's raw query string |
| `rag.answer` | The generated response text |

---

## Rationale

**Debuggability is not optional for a monitoring system.** An alert without the query that triggered it requires manual reproduction. For sampled traffic (10% of requests), reproducing the exact input after the fact is often impossible. The monitoring layer exists to diagnose problems — removing the primary diagnostic input defeats that purpose.

**This system has no PII in its query corpus.** Queries target architecture documentation. The LGPD concern does not apply at this scope.

**Calibrating RAGAS requires content.** As demonstrated during live testing, evaluator instability (Gemini Flash Lite generating off-target questions for multi-part queries) only becomes diagnosable when the answer is visible in the span. Score alone is insufficient.

---

## Alternatives Considered

### Alternative A — Exclude query and answer (original approach)

Addresses LGPD risk preemptively by never sending content to Axiom.

**Reason to reject:** Debugging RAGAS failures becomes impossible without content. Demonstrated in live testing — a span with `faithfulness: 1.0`, `answer_relevancy: 0`, `context_precision: 0` cannot be diagnosed without knowing what was asked and what was answered.

### Alternative B — Store query hash + separate compliant query log

Store `sha256(query)` in the span; maintain a separate encrypted query store joinable by hash.

**Reason to reject for this POC:** Requires building and operating a second storage system. The hash is useless without the store. This is the correct production escalation — not the right investment for a POC with no real user data.

---

## Consequences

**Expected benefits:**
- Alert investigation is possible directly from the Axiom span — no manual reproduction needed
- RAGAS evaluator failures (false positives from multi-part queries) are diagnosable from the span alone
- Threshold calibration is grounded in real query content, not inferred from scores

**Trade-offs and accepted risks:**
- If this pattern is copied into a production system with real users without modification, it creates an LGPD Art. 33 violation
- The span payload is larger

**Triggers for reassessment:**
- Real user traffic enters the system → remove `rag.query` and `rag.answer` from the span and implement hash-based correlation with a compliant query store
- Axiom DPA is signed and user consent mechanism is in place → fields can be re-added under a valid legal basis
