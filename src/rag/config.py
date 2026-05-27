from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key_env: str
    timeout_ms: int
    max_retries: int


@dataclass(frozen=True)
class PrimaryConfig:
    simple: ProviderConfig
    medium: ProviderConfig
    complex: ProviderConfig


@dataclass(frozen=True)
class FallbackConfig:
    simple: tuple[ProviderConfig, ...]
    medium: tuple[ProviderConfig, ...]
    complex: tuple[ProviderConfig, ...]


@dataclass(frozen=True)
class LLMGatewayConfig:
    primary: PrimaryConfig
    fallback: FallbackConfig


@dataclass(frozen=True)
class ClassifierConfig:
    provider: str
    model: str
    api_key_env: str
    max_tokens: int


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int
    recovery_timeout_ms: int
    half_open_max_calls: int


@dataclass(frozen=True)
class TokenBudgetConfig:
    max_tokens_per_request: int
    classifier_max_tokens: int


@dataclass(frozen=True)
class SemanticCacheConfig:
    provider: str
    similarity_threshold: float
    ttl_seconds: int
    embedding_model: str


@dataclass(frozen=True)
class VectorDBConfig:
    provider: str
    collection: str
    rrf_dense_weight: float
    rrf_sparse_weight: float
    top_k: int


@dataclass(frozen=True)
class IngestionConfig:
    strategy: str  # "fixed" | "hierarchical"
    chunk_size: int        # child chunk size for hierarchical; single chunk size for fixed
    chunk_overlap: int
    parent_chunk_size: int  # only used for "hierarchical"


@dataclass(frozen=True)
class EvaluationConfig:
    framework: str
    metrics: tuple[str, ...]
    min_score_threshold: float
    sample_rate: float


@dataclass(frozen=True)
class Config:
    llm_gateway: LLMGatewayConfig
    classifier: ClassifierConfig
    circuit_breaker: CircuitBreakerConfig
    token_budget: TokenBudgetConfig
    semantic_cache: SemanticCacheConfig
    vector_db: VectorDBConfig
    ingestion: IngestionConfig
    evaluation: EvaluationConfig


def _parse_provider(raw: object) -> ProviderConfig:
    if not isinstance(raw, dict):
        raise TypeError("Expected dictionary for provider config")
    return ProviderConfig(
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        api_key_env=str(raw["api_key_env"]),
        timeout_ms=int(raw.get("timeout_ms", 30000)),
        max_retries=int(raw.get("max_retries", 2)),
    )


def _to_provider_tuple(raw: object) -> tuple[ProviderConfig, ...]:
    if not isinstance(raw, list):
        raise TypeError("Expected list of providers")
    return tuple(_parse_provider(p) for p in raw)


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Config:
    """
    Loads and caches the system configuration from config.yaml.
    Call with path=None to use the default (project root).
    Pass a path string to override (useful in tests).
    """
    config_path = Path(path) if path else Path(__file__).parents[2] / "config.yaml"
    raw_yaml = yaml.safe_load(config_path.read_text())
    if not isinstance(raw_yaml, dict):
        raise TypeError("Expected root configuration to be a dictionary")

    raw: dict[str, object] = {str(k): v for k, v in raw_yaml.items()}

    gw = raw["llm_gateway"]
    if not isinstance(gw, dict):
        raise TypeError("Expected dictionary for llm_gateway")

    primary = gw["primary"]
    if not isinstance(primary, dict):
        raise TypeError("Expected dictionary for llm_gateway.primary")

    fb = gw["fallback"]
    if not isinstance(fb, dict):
        raise TypeError("Expected dictionary for llm_gateway.fallback")

    classifier = raw["classifier"]
    if not isinstance(classifier, dict):
        raise TypeError("Expected dictionary for classifier")

    circuit_breaker = raw["circuit_breaker"]
    if not isinstance(circuit_breaker, dict):
        raise TypeError("Expected dictionary for circuit_breaker")

    token_budget = raw["token_budget"]
    if not isinstance(token_budget, dict):
        raise TypeError("Expected dictionary for token_budget")

    semantic_cache = raw["semantic_cache"]
    if not isinstance(semantic_cache, dict):
        raise TypeError("Expected dictionary for semantic_cache")

    vector_db = raw["vector_db"]
    if not isinstance(vector_db, dict):
        raise TypeError("Expected dictionary for vector_db")

    ingestion = raw["ingestion"]
    if not isinstance(ingestion, dict):
        raise TypeError("Expected dictionary for ingestion")

    evaluation = raw["evaluation"]
    if not isinstance(evaluation, dict):
        raise TypeError("Expected dictionary for evaluation")

    metrics_list = evaluation["metrics"]
    if not isinstance(metrics_list, list):
        raise TypeError("Expected list for evaluation.metrics")

    return Config(
        llm_gateway=LLMGatewayConfig(
            primary=PrimaryConfig(
                simple=_parse_provider(primary["simple"]),
                medium=_parse_provider(primary["medium"]),
                complex=_parse_provider(primary["complex"]),
            ),
            fallback=FallbackConfig(
                simple=_to_provider_tuple(fb["simple"]),
                medium=_to_provider_tuple(fb["medium"]),
                complex=_to_provider_tuple(fb["complex"]),
            ),
        ),
        classifier=ClassifierConfig(
            provider=str(classifier["provider"]),
            model=str(classifier["model"]),
            api_key_env=str(classifier["api_key_env"]),
            max_tokens=int(classifier["max_tokens"]),
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=int(circuit_breaker["failure_threshold"]),
            recovery_timeout_ms=int(circuit_breaker["recovery_timeout_ms"]),
            half_open_max_calls=int(circuit_breaker["half_open_max_calls"]),
        ),
        token_budget=TokenBudgetConfig(
            max_tokens_per_request=int(token_budget["max_tokens_per_request"]),
            classifier_max_tokens=int(token_budget["classifier_max_tokens"]),
        ),
        semantic_cache=SemanticCacheConfig(
            provider=str(semantic_cache["provider"]),
            similarity_threshold=float(semantic_cache["similarity_threshold"]),
            ttl_seconds=int(semantic_cache["ttl_seconds"]),
            embedding_model=str(semantic_cache["embedding_model"]),
        ),
        vector_db=VectorDBConfig(
            provider=str(vector_db["provider"]),
            collection=str(vector_db["collection"]),
            rrf_dense_weight=float(vector_db["rrf_dense_weight"]),
            rrf_sparse_weight=float(vector_db["rrf_sparse_weight"]),
            top_k=int(vector_db["top_k"]),
        ),
        ingestion=IngestionConfig(
            strategy=str(ingestion["strategy"]),
            chunk_size=int(ingestion["chunk_size"]),
            chunk_overlap=int(ingestion["chunk_overlap"]),
            parent_chunk_size=int(ingestion["parent_chunk_size"]),
        ),
        evaluation=EvaluationConfig(
            framework=str(evaluation["framework"]),
            metrics=tuple(str(m) for m in metrics_list),
            min_score_threshold=float(evaluation["min_score_threshold"]),
            sample_rate=float(evaluation["sample_rate"]),
        ),
    )
