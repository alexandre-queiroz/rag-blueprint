# ADR-010: Observability and Monitoring — Axiom

- **Status:** Accepted
- **Date:** 2026-05-27
- **Author(s):** Alexandre Queiroz
- **Version:** 1.0

---

## Context

The system produces two distinct categories of data that need a destination:

**Observability** — operational data emitted on every request:
- Request trace through the pipeline (Classifier → Cache → DB Search → LLM Gateway → Circuit Breaker)
- Latency per layer
- Model used, complexity tier, tokens consumed, estimated cost
- Cache hit/miss

**Monitoring** — RAG quality data from async RAGAS sampling (10% of traffic):
- Faithfulness, answer relevancy, context precision scores
- Alerts when any metric falls below `evaluation.min_score_threshold`
- Chunk hit rate per document

Without a defined destination, this data exists only as `print()` statements — useful locally, invisible in any real environment.

The question is where this data should go and whether observability and monitoring need separate tools.

---

## Decision

Use **Axiom** as the single destination for both observability events and monitoring data, instrumented via **OpenTelemetry**.

All structured events (operational and quality) are sent to Axiom via the **OTLP HTTP exporter** (`opentelemetry-exporter-otlp-proto-http`) pointed at `https://api.axiom.co/v1/traces`. No Axiom-specific SDK is used — Axiom accepts standard OTLP natively, so the exporter is provider-agnostic.

```
OpenTelemetry SDK
    ├── Traces  → Axiom  (request flow per layer, latency breakdown)
    ├── Metrics → Axiom  (cost, tokens, cache hit rate, circuit breaker state)
    └── Events  → Axiom  (RAGAS scores, quality alerts)
```

---

## Rationale

- **Single platform for both concerns:** observability and monitoring land in the same query interface — correlating a quality drop with a latency spike or a circuit breaker event requires no context switching between tools
- **Native OpenTelemetry support:** Axiom accepts OTLP (OpenTelemetry Protocol) natively — instrumentation is provider-agnostic; replacing Axiom with any other OTel-compatible backend requires only changing the exporter endpoint
- **Generous free tier:** 500 GB/month ingestion, 30-day retention, permanent — covers substantial real traffic without cost for a POC or early validation stage
- **Event-based model fits RAG monitoring:** RAGAS scores are discrete events, not time-series counters. Axiom's log analytics model (ingest everything, query later) is a better fit than Prometheus's pull-based metrics model for this data shape

---

## Alternatives Considered

### Alternative A — Prometheus + Grafana

Pull-based metrics scraping from an instrumented `/metrics` endpoint, visualized in Grafana.

**Perceived advantages:**
- Industry standard for infrastructure metrics
- Open source, self-hostable
- Rich ecosystem of exporters and dashboards

**Reasons to reject:**
- Pull-based model is a poor fit for discrete events (RAGAS scores, circuit breaker state changes) — they need to be modeled as counters or gauges, losing event-level detail
- Requires running two services (Prometheus + Grafana) instead of one
- No native log analytics — a separate aggregator (ELK, Loki) would still be needed for structured events
- More infrastructure to configure and maintain for the same outcome

### Alternative B — Datadog

Full observability suite: metrics, logs, traces, APM, alerts, dashboards.

**Perceived advantages:**
- Industry standard at enterprise scale
- Unified platform with the most comprehensive feature set
- Native OpenTelemetry ingestion

**Reasons to reject:**
- Pricing is prohibitive for a POC — no permanent free tier; costs escalate sharply with volume
- Feature set far exceeds what this system needs
- The operational overhead of configuring Datadog agents outweighs the benefit at this scale

### Alternative C — Structured logging to stdout only

Emit structured JSON logs to stdout and let the deployment infrastructure handle routing.

**Perceived advantages:**
- Zero additional dependency
- Works in any environment — cloud providers ingest stdout natively (CloudWatch, Cloud Logging, etc.)
- Correct pattern for twelve-factor applications

**Reasons to reject:**
- Stdout has no query interface, no dashboards, and no alerting without additional tooling on top
- Quality monitoring alerts (RAGAS below threshold) would require configuring log-based alerts in whatever infrastructure handles stdout — more setup than Axiom with less functionality
- Acceptable as a fallback; not acceptable as a primary monitoring strategy

---

## Consequences

**Expected benefits:**
- Operational and quality data are queryable, dashboardable, and alertable from day one
- OpenTelemetry instrumentation is provider-agnostic — Axiom can be replaced without changing application code, only the exporter endpoint
- Single free-tier account covers both observability and monitoring for POC and early validation

**Trade-offs and accepted risks:**
- **Axiom does not replace infrastructure monitoring** (pod CPU, memory, disk, network) — if the system runs on Kubernetes or a managed cloud, infrastructure metrics still require the platform's native tooling (CloudWatch, Cloud Monitoring, etc.)
- **Free tier has retention limits:** 30 days — historical quality trend analysis beyond that window requires a paid plan or data export
- Axiom as a vendor introduces a dependency — if it becomes unavailable, observability stops until the exporter is reconfigured

**Required actions:**
- `opentelemetry-sdk` and `opentelemetry-exporter-otlp-proto-http` added to `pyproject.toml` under the `monitoring` extra
- `AXIOM_API_KEY` and `AXIOM_DATASET` added to `.env.example`
- LLM Gateway and monitoring layer emit structured events to Axiom on every request and every RAGAS evaluation
- Circuit breaker listeners emit state change events to Axiom

**Triggers for reassessment:**
- System deployed to production with real multi-tenant traffic → evaluate Datadog or self-hosted OTel collector with long-term storage
- Retention beyond 30 days required for compliance or trend analysis → Axiom paid plan or export pipeline to S3/BigQuery
- Infrastructure metrics become a requirement → add platform-native monitoring alongside Axiom
