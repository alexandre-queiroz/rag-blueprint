from __future__ import annotations

import pytest

from rag.serve.pipeline import _validate_query


# ── _validate_query ───────────────────────────────────────────────────────────
#
# RAGPipeline.query() itself is pure orchestration — all methods depend on
# external services (Redis, Chroma, LiteLLM). Testing it in isolation would
# require mocking every layer, which contradicts the project's testing philosophy
# (pure functions only; no mocks for external services). Integration tests with
# real infrastructure are the appropriate vehicle for the orchestration logic.


class TestValidateQuery:
    def test_valid_query_returned_unchanged(self) -> None:
        assert _validate_query("What is RAG?") == "What is RAG?"

    def test_leading_whitespace_stripped(self) -> None:
        assert _validate_query("  What is RAG?") == "What is RAG?"

    def test_trailing_whitespace_stripped(self) -> None:
        assert _validate_query("What is RAG?  ") == "What is RAG?"

    def test_both_sides_stripped(self) -> None:
        assert _validate_query("  What is RAG?  ") == "What is RAG?"

    def test_internal_whitespace_preserved(self) -> None:
        assert _validate_query("  hello   world  ") == "hello   world"

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_query("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_query("   ")

    def test_tab_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_query("\t")

    def test_newline_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_query("\n")

    def test_returns_str(self) -> None:
        assert isinstance(_validate_query("hello"), str)
