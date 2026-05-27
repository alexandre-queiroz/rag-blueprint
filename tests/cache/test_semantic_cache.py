from __future__ import annotations

import math

import pytest

from rag.cache.semantic_cache import (
    _cosine_similarity,
    _deserialize_response,
    _query_key,
    _serialize_response,
)
from rag.types import CompletionRecord, RAGResponse


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_response(answer: str = "Test answer") -> RAGResponse:
    return RAGResponse(
        answer=answer,
        sources=["doc_a.txt", "doc_b.txt"],
        complexity="medium",
        record=CompletionRecord(
            model="claude-sonnet-4-6",
            provider="anthropic",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=980.0,
            cost_usd=0.0012,
        ),
        cached=False,
    )


# ── _cosine_similarity ────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a_returns_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_b_returns_zero(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_scaled_vectors_return_one(self) -> None:
        # Cosine similarity is scale-invariant
        a = [2.0, 4.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == pytest.approx(1.0)

    def test_known_angle(self) -> None:
        # 45° angle → cos(45°) = √2/2 ≈ 0.707
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)


# ── _query_key ────────────────────────────────────────────────────────────────


class TestQueryKey:
    def test_same_query_produces_same_key(self) -> None:
        assert _query_key("What is RAG?") == _query_key("What is RAG?")

    def test_different_queries_produce_different_keys(self) -> None:
        assert _query_key("What is RAG?") != _query_key("What is a vector database?")

    def test_key_is_16_characters(self) -> None:
        assert len(_query_key("any query")) == 16

    def test_normalization_lowercases(self) -> None:
        # Same content, different case → same key
        assert _query_key("What is RAG?") == _query_key("what is rag?")

    def test_normalization_strips_whitespace(self) -> None:
        assert _query_key("What is RAG?") == _query_key("  What is RAG?  ")


# ── serialize / deserialize roundtrip ────────────────────────────────────────


class TestSerializeDeserialize:
    def test_roundtrip_preserves_answer(self) -> None:
        response = _make_response(answer="The answer is 42.")
        raw = {"response": __import__("json").dumps(_serialize_response(response)), "embedding": "[]", "query": "q"}
        result = _deserialize_response(raw)
        assert result.answer == "The answer is 42."

    def test_roundtrip_preserves_sources(self) -> None:
        response = _make_response()
        raw = {"response": __import__("json").dumps(_serialize_response(response)), "embedding": "[]", "query": "q"}
        result = _deserialize_response(raw)
        assert result.sources == ["doc_a.txt", "doc_b.txt"]

    def test_roundtrip_preserves_complexity(self) -> None:
        response = _make_response()
        raw = {"response": __import__("json").dumps(_serialize_response(response)), "embedding": "[]", "query": "q"}
        result = _deserialize_response(raw)
        assert result.complexity == "medium"

    def test_deserialized_response_is_marked_cached(self) -> None:
        response = _make_response()
        raw = {"response": __import__("json").dumps(_serialize_response(response)), "embedding": "[]", "query": "q"}
        result = _deserialize_response(raw)
        assert result.cached is True

    def test_deserialized_response_has_no_record(self) -> None:
        # Cache hits have no CompletionRecord — no LLM was called
        response = _make_response()
        raw = {"response": __import__("json").dumps(_serialize_response(response)), "embedding": "[]", "query": "q"}
        result = _deserialize_response(raw)
        assert result.record is None

    def test_serialize_does_not_include_record(self) -> None:
        response = _make_response()
        serialized = _serialize_response(response)
        assert "record" not in serialized
