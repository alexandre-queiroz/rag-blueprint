from __future__ import annotations

import pytest

from rag.config import FallbackConfig, LLMGatewayConfig, PrimaryConfig, ProviderConfig
from rag.gateway.gateway import (
    TokenBudgetExceededError,
    _build_messages,
    _degradation_response,
    _litellm_model,
    _provider_chain,
)
from rag.types import RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _provider(provider: str, model: str) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model=model,
        api_key_env=f"{provider.upper()}_API_KEY",
        timeout_ms=15000,
        max_retries=2,
    )


def _make_gateway_config() -> LLMGatewayConfig:
    haiku = _provider("anthropic", "claude-3-5-haiku-20241022")
    sonnet = _provider("anthropic", "claude-sonnet-4-6")
    flash_lite = _provider("google", "gemini-2.5-flash-lite")
    gpt4o_mini = _provider("openai", "gpt-4o-mini")
    gpt4o = _provider("openai", "gpt-4o")
    flash = _provider("google", "gemini-2.5-flash")

    return LLMGatewayConfig(
        primary=PrimaryConfig(simple=haiku, medium=sonnet, complex=sonnet),
        fallback=FallbackConfig(
            simple=(flash_lite,),
            medium=(gpt4o_mini, flash),
            complex=(gpt4o, flash),
        ),
    )


def _make_chunk(source: str = "doc.txt", index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        document="Relevant context about the topic.",
        score=0.95,
        source=source,
        chunk_index=index,
    )


# ── _provider_chain ───────────────────────────────────────────────────────────


class TestProviderChain:
    def test_simple_chain_starts_with_haiku(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "simple")
        assert chain[0].model == "claude-3-5-haiku-20241022"

    def test_medium_chain_starts_with_sonnet(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "medium")
        assert chain[0].model == "claude-sonnet-4-6"

    def test_complex_chain_starts_with_sonnet(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "complex")
        assert chain[0].model == "claude-sonnet-4-6"

    def test_simple_fallback_is_flash_lite(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "simple")
        assert chain[1].model == "gemini-2.5-flash-lite"

    def test_medium_chain_has_three_providers(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "medium")
        assert len(chain) == 3

    def test_complex_chain_has_three_providers(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "complex")
        assert len(chain) == 3

    def test_simple_chain_has_two_providers(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "simple")
        assert len(chain) == 2

    def test_medium_fallback_order(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "medium")
        assert chain[1].model == "gpt-4o-mini"
        assert chain[2].model == "gemini-2.5-flash"

    def test_complex_fallback_order(self) -> None:
        config = _make_gateway_config()
        chain = _provider_chain(config, "complex")
        assert chain[1].model == "gpt-4o"
        assert chain[2].model == "gemini-2.5-flash"


# ── _litellm_model ────────────────────────────────────────────────────────────


class TestLitellmModel:
    def test_anthropic_prefix(self) -> None:
        provider = _provider("anthropic", "claude-sonnet-4-6")
        assert _litellm_model(provider) == "anthropic/claude-sonnet-4-6"

    def test_google_maps_to_gemini_prefix(self) -> None:
        provider = _provider("google", "gemini-2.5-flash-lite")
        assert _litellm_model(provider) == "gemini/gemini-2.5-flash-lite"

    def test_openai_prefix(self) -> None:
        provider = _provider("openai", "gpt-4o-mini")
        assert _litellm_model(provider) == "openai/gpt-4o-mini"

    def test_unknown_provider_uses_name_as_prefix(self) -> None:
        provider = _provider("cohere", "command-r")
        assert _litellm_model(provider) == "cohere/command-r"


# ── _build_messages ───────────────────────────────────────────────────────────


class TestBuildMessages:
    def test_returns_two_messages(self) -> None:
        messages = _build_messages("What is RAG?", [_make_chunk()])
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        messages = _build_messages("What is RAG?", [_make_chunk()])
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self) -> None:
        messages = _build_messages("What is RAG?", [_make_chunk()])
        assert messages[1]["role"] == "user"

    def test_user_message_contains_query(self) -> None:
        messages = _build_messages("What is RAG?", [_make_chunk()])
        assert "What is RAG?" in messages[1]["content"]

    def test_user_message_contains_chunk_document(self) -> None:
        chunk = _make_chunk()
        messages = _build_messages("query", [chunk])
        assert chunk.document in messages[1]["content"]

    def test_user_message_contains_chunk_source(self) -> None:
        chunk = _make_chunk(source="architecture.md")
        messages = _build_messages("query", [chunk])
        assert "architecture.md" in messages[1]["content"]

    def test_multiple_chunks_all_included(self) -> None:
        chunks = [_make_chunk("doc_a.txt", 0), _make_chunk("doc_b.txt", 1)]
        messages = _build_messages("query", chunks)
        assert "doc_a.txt" in messages[1]["content"]
        assert "doc_b.txt" in messages[1]["content"]

    def test_empty_chunks_still_returns_two_messages(self) -> None:
        messages = _build_messages("What is RAG?", [])
        assert len(messages) == 2


# ── TokenBudgetExceededError ──────────────────────────────────────────────────


class TestTokenBudgetExceededError:
    def test_is_exception(self) -> None:
        err = TokenBudgetExceededError(5000, 4000)
        assert isinstance(err, Exception)

    def test_stores_estimated(self) -> None:
        err = TokenBudgetExceededError(5000, 4000)
        assert err.estimated == 5000

    def test_stores_limit(self) -> None:
        err = TokenBudgetExceededError(5000, 4000)
        assert err.limit == 4000

    def test_message_contains_values(self) -> None:
        err = TokenBudgetExceededError(5000, 4000)
        assert "5000" in str(err)
        assert "4000" in str(err)


# ── _degradation_response ─────────────────────────────────────────────────────


class TestDegradationResponse:
    def test_returns_rag_response(self) -> None:
        from rag.types import RAGResponse
        response = _degradation_response("simple")
        assert isinstance(response, RAGResponse)

    def test_complexity_preserved(self) -> None:
        for complexity in ("simple", "medium", "complex"):
            response = _degradation_response(complexity)  # type: ignore[arg-type]
            assert response.complexity == complexity

    def test_sources_is_empty(self) -> None:
        assert _degradation_response("medium").sources == []

    def test_record_is_none(self) -> None:
        assert _degradation_response("medium").record is None

    def test_cached_is_false(self) -> None:
        assert _degradation_response("medium").cached is False

    def test_answer_is_not_empty(self) -> None:
        assert len(_degradation_response("medium").answer) > 0
