# ADR-009: Token Budget — Per Request

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0

---

## Context

Production RAG systems need cost control over LLM calls. The concrete risk: a single request with a malformed context (large chunk + long history + extended query) can consume the budget of an expensive call before any validation. Without per-request control, this cost goes unnoticed until the bill arrives.

The market standard for cost control in multi-tenant systems is a hierarchical cascading model:

1. **Request** — per-request limit; immediate ceiling against runaway requests
2. **Organization** — total organization budget; absolute ceiling
3. **Department** — sub-quota per cost center within the organization
4. **User** — individual limit within the department

Each level must be validated in cascade: a request is only accepted if organization, department, and user all have remaining budget.

---

## Decision

Implement token budget **exclusively per request** (`max_tokens_per_request`). The hierarchical cascading model is out of scope for this system.

---

## Rationale

- **Per request solves the immediate pain:** prevents runaway requests without needing external state. Stateless, testable, no Redis or DB dependency.
- **The cascading model is the correct standard, but requires an identity layer:** to validate budget per organization, department, and user, the system needs to know who the authenticated caller is and which hierarchy they belong to. Without auth + directory service (LDAP, Okta, etc.), hierarchical grouping has no semantics. Implementing rate limiting without a trusted caller identity has no value.

---

## Alternatives Considered

### Alternative A — Per request ✅ chosen

Fixed limit on the size of a single prompt/context. The request is rejected before any provider call if the estimated tokens exceed `max_tokens_per_request`.

**Advantages:**
- Stateless — no Redis or DB dependency, no state between requests
- Testable in isolation — validation is a pure function over prompt size
- Solves the immediate pain: prevents a single malformed request from consuming disproportionate cost
- Does not require caller identity — works without an auth layer

---

### Alternative B — Hierarchical cascading budget (organization → department → user)

Each request consumes from the user's budget, which consumes from the department's, which consumes from the organization's. A request is rejected if any level of the hierarchy is exhausted.

```
Organization: 10,000,000 tokens/month
  └─ Department A: 3,000,000 tokens/month
       └─ User 1: 50,000 tokens/day
       └─ User 2: 50,000 tokens/day
  └─ Department B: 2,000,000 tokens/month
       └─ ...
```

**Perceived advantages:**
- Real cost control — maps directly to the billing model of LLM providers
- Fair usage between tenants — a department or user cannot exhaust others' budgets
- Operational granularity — data teams and product teams can have independent quotas

**Reasons to reject (in this scope):**
- Requires authenticated caller identity on every request — without an auth layer, hierarchical grouping has no semantics
- Requires real-time resolution of the organization → department → user hierarchy, which implies an integrated directory service (LDAP, Okta, etc.)
- Requires persistent state per level: three counters in Redis with TTL aligned to the billing cycle (day/month) per entity
- Not a definitive rejection — this is the correct escalation when the system has an authentication layer and multi-tenancy

---

## Consequences

**Expected benefits:**
- Immediate cost control against runaway requests, without external state
- Stateless implementation — testable in isolation

**Trade-offs and accepted risks:**
- No accumulation control: a caller can make N requests within the unit limit without aggregate restriction
- Acceptable in this scope — the system has no multi-tenancy or trusted caller identity

**Natural escalation:**
- When the system has an auth layer, Redis (already in the stack for the semantic cache) is the direct backend for hierarchical counters — no structural change in the gateway, only the addition of a rate limiting middleware with identity resolution before the pipeline entry point

**Required actions:**
- `max_tokens_per_session` removed from `config.yaml`
- Escalation path to the hierarchical model documented in `config.yaml`
