from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComplexityLabel = Literal["simple", "medium", "complex"]


@dataclass
class RetrievedChunk:
    document: str
    score: float
    source: str
    chunk_index: int


@dataclass
class CompletionRecord:
    """Metadata recorded after each LLM call — used for cost tracking and monitoring."""

    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    complexity: ComplexityLabel
    record: CompletionRecord | None  # None for cache hits
    cached: bool = False
