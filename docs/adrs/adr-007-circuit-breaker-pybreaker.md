# ADR-007: Circuit Breaker — pybreaker

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0

---

## Context

The system depends on external LLM providers (Anthropic, Google, OpenAI) that may have partial or total unavailability. Without protection, a prolonged provider failure causes a cascade problem: each request waits the full timeout before trying the fallback, threads accumulate waiting on unresponsive providers, and the system degrades under load.

The **Circuit Breaker** pattern (Martin Fowler, 2014) solves this problem. It wraps calls to an external service and tracks failures. When the number of consecutive failures exceeds a threshold, the circuit "opens" and subsequent calls are rejected immediately — without touching the failing provider — until a recovery timeout elapses.

Three states:

```
         failure_threshold reached
[Closed] ──────────────────────────► [Open]
   ▲                                    │
   │ successful test call               │ recovery_timeout elapsed
   │                                    ▼
   └──────────────── [Half-Open] ◄──────┘
                  (allows 1 test call)
```

- **Closed:** normal operation, all calls pass through
- **Open:** calls rejected immediately, next provider in the chain is used
- **Half-Open:** after the timeout, one test call is allowed; success closes the circuit, failure reopens it

It is important to distinguish Circuit Breaker from **Retry**: retry handles transient errors ("try again after backoff"); circuit breaker handles sustained failures ("stop trying for a while"). Using only retry under a prolonged failure means each request burns N attempts before giving up, multiplying latency and cost.

---

## Decision

Use **pybreaker** for the circuit breaker implementation, with one instance per provider maintained in the LLM Gateway.

---

## Rationale

- **Correct pattern implementation:** pybreaker implements the three states (Closed/Open/Half-Open) with thread-safe transitions
- **Direct mapping to `config.yaml`:** `fail_max` → `circuit_breaker.failure_threshold`, `reset_timeout` → `circuit_breaker.recovery_timeout_ms`
- **Listeners for observability:** supports hooks that fire when the circuit changes state — emitting metrics (e.g. "Anthropic provider circuit opened") is trivial
- **Not retry:** intentional choice — retry (tenacity) exists in parallel for transient errors; circuit breaker exists for sustained failures

---

## Alternatives Considered

### Alternative A — tenacity (retry with backoff)

Python library for retry with exponential backoff, jitter, and configurable stop conditions.

**Perceived advantages:**
- Widely adopted, well documented
- Handles transient errors (500ms timeout, momentary rate limit) elegantly

**Reasons to reject:**
- tenacity is retry, not circuit breaker — different patterns for different problems
- Under prolonged provider failure, tenacity with 3 retries means each request waits 3× the timeout before going to the fallback
- Has no concept of "open" state that blocks calls preemptively — the cascade problem persists

> tenacity can (and should) be used in parallel with pybreaker to handle transient errors before the circuit breaker counts a failure.

### Alternative B — Custom implementation

Write the circuit breaker from scratch: state machine, thread-safe failure counter, recovery timeout.

**Perceived advantages:**
- Full control over behavior in each state
- No additional dependency

**Reasons to reject:**
- Concurrent state transitions (multiple threads trying to open the circuit simultaneously) have subtle race conditions
- pybreaker has already solved these edge cases — rewriting is reinventing without differentiated value
- For a POC, the complexity of maintaining a correct thread-safe state machine is not justified

---

## Consequences

**Expected benefits:**
- Provider failures stop propagating after `failure_threshold` — the system fails fast and moves to the next provider
- Providers in recovery get a chance to recover without being bombarded by requests (Half-Open)
- Observability: listeners emit auditable events when circuits change state

**Trade-offs and accepted risks:**
- **In-process, in-memory state:** in a multi-instance deploy (pods), each instance has its own circuit state — a provider may be "open" in one pod and "closed" in another. For true distributed circuit breaking, shared state in Redis would be required
- For this POC running as a single process, in-process behavior is sufficient

**Required actions:**
- `pybreaker>=1.2.0` added to `pyproject.toml`
- LLM Gateway must maintain a registry of `CircuitBreaker` instances per provider, initialized with values from `config.yaml`

**Triggers for reassessment:**
- Multi-instance deploy (pods) — in-memory circuit breaker loses effectiveness and shared state in Redis becomes necessary
- SLA requirement with faster recovery than supported by current parameters
