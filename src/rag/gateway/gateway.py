from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import cast

import litellm
import pybreaker

_logger = logging.getLogger(__name__)

from rag.circuit_breaker.breaker import CircuitBreakerRegistry
from rag.config import LLMGatewayConfig, ProviderConfig, TokenBudgetConfig
from rag.types import ComplexityLabel, CompletionRecord, RAGResponse, RetrievedChunk

# Provider name → LiteLLM model prefix
_LITELLM_PREFIX: dict[str, str] = {
    "anthropic": "anthropic",
    "google": "gemini",   # Google AI Studio (Gemini API)
    "openai": "openai",
}

_SYSTEM_PROMPT = (
    "Answer the question using only the information in the context below. "
    "Be concise and accurate. "
    "If the context does not contain enough information to answer, "
    "say so clearly rather than speculating."
)

_DEGRADATION_ANSWER = (
    "I'm unable to process your request right now. "
    "All AI providers are temporarily unavailable. Please try again later."
)


class TokenBudgetExceededError(Exception):
    """Raised when the estimated prompt token count exceeds the configured limit."""

    def __init__(self, estimated: int, limit: int) -> None:
        super().__init__(
            f"Token budget exceeded: {estimated} estimated tokens > {limit} limit"
        )
        self.estimated = estimated
        self.limit = limit


class LLMGateway:
    """Complexity-aware LLM Gateway.

    Single entry point for all LLM calls in the serving path.
    No other component imports a provider SDK directly.

    Flow per request:
        1. Assemble prompt (system + context chunks + query)
        2. Estimate token count — reject if over budget (ADR-009)
        3. Try primary model for the complexity tier
        4. On failure or open circuit → try each fallback in order (ADR-006)
        5. If all providers exhausted → return graceful degradation response
        6. Record: model, provider, tokens, cost, latency
    """

    def __init__(
        self,
        config: LLMGatewayConfig,
        token_budget: TokenBudgetConfig,
        circuit_breakers: CircuitBreakerRegistry,
    ) -> None:
        self._config = config
        self._token_budget = token_budget
        self._circuit_breakers = circuit_breakers

    async def complete(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        complexity: ComplexityLabel,
    ) -> RAGResponse:
        """Run the query through the provider chain for the given complexity tier.

        Raises TokenBudgetExceededError before any provider call if the prompt
        exceeds the configured token limit.
        """
        messages = _build_messages(query, chunks)
        chain = _provider_chain(self._config, complexity)

        # Token budget check using the primary model's tokenizer
        primary_model = _litellm_model(chain[0])
        estimated = _estimate_tokens(primary_model, messages)
        if estimated > self._token_budget.max_tokens_per_request:
            raise TokenBudgetExceededError(estimated, self._token_budget.max_tokens_per_request)

        for provider in chain:
            if not self._circuit_breakers.is_available(provider.provider):
                continue

            model = _litellm_model(provider)
            breaker = self._circuit_breakers.get(provider.provider)
            started_at = time.monotonic()

            try:
                with breaker.calling():
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        timeout=provider.timeout_ms / 1000,
                        num_retries=provider.max_retries,
                        api_key=os.environ.get(provider.api_key_env, ""),
                    )
                latency_ms = (time.monotonic() - started_at) * 1000

                return RAGResponse(
                    answer=str(response.choices[0].message.content),
                    sources=list(dict.fromkeys(c.source for c in chunks)),
                    complexity=complexity,
                    record=_build_record(response, model, provider.provider, latency_ms),
                    cached=False,
                )

            except pybreaker.CircuitBreakerError:
                _logger.debug("circuit open for provider %s — skipping", provider.provider)
                continue

            except Exception as exc:
                # Log at WARNING so the failure is visible in any log aggregator
                # without stopping the fallback chain. pybreaker records the
                # failure on its end; we record the context here.
                _logger.warning(
                    "provider %s model %s failed: %s: %s",
                    provider.provider,
                    model,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                continue

        return _degradation_response(complexity)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _provider_chain(
    config: LLMGatewayConfig,
    complexity: ComplexityLabel,
) -> list[ProviderConfig]:
    """Return [primary] + fallback list for the given complexity tier."""
    primary = getattr(config.primary, complexity)
    fallbacks = list(getattr(config.fallback, complexity))
    return [primary] + fallbacks


def _litellm_model(provider: ProviderConfig) -> str:
    """Build the LiteLLM model string from a ProviderConfig."""
    prefix = _LITELLM_PREFIX.get(provider.provider, provider.provider)
    return f"{prefix}/{provider.model}"


def _build_messages(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """Assemble the prompt as a LiteLLM-compatible messages list."""
    context = "\n\n".join(
        f"[Source: {chunk.source}, chunk {chunk.chunk_index}]\n{chunk.document}"
        for chunk in chunks
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def _estimate_tokens(model: str, messages: list[dict[str, str]]) -> int:
    """Estimate prompt token count using LiteLLM's token counter.

    Falls back to a character-based estimate (4 chars ≈ 1 token) when the
    model's tokenizer is not available locally.
    """
    try:
        return litellm.token_counter(model=model, messages=messages)
    except Exception:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4


def _build_record(
    response: object,
    model: str,
    provider: str,
    latency_ms: float,
) -> CompletionRecord:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
    completion_tokens = int(getattr(usage, "completion_tokens", 0))

    try:
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception:
        cost = 0.0

    return CompletionRecord(
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost,
    )


def _degradation_response(complexity: ComplexityLabel) -> RAGResponse:
    """Return a safe fallback response when all providers are unavailable."""
    return RAGResponse(
        answer=_DEGRADATION_ANSWER,
        sources=[],
        complexity=complexity,
        record=None,
        cached=False,
    )
