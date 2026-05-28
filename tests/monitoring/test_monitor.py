from __future__ import annotations

import pytest

from rag.monitoring.monitor import (
    _build_ragas_dataset,
    _format_axiom_event,
    _scores_below_threshold,
    _should_sample,
)
from rag.types import CompletionRecord, RAGResponse, RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_response(
    answer: str = "RAG stands for Retrieval-Augmented Generation.",
    complexity: str = "simple",
    cached: bool = False,
    with_record: bool = True,
) -> RAGResponse:
    record = (
        CompletionRecord(
            model="claude-3-5-haiku-20241022",
            provider="anthropic",
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
            latency_ms=450.0,
            cost_usd=0.0003,
        )
        if with_record
        else None
    )
    return RAGResponse(
        answer=answer,
        sources=["doc.txt"],
        complexity=complexity,  # type: ignore[arg-type]
        record=record,
        cached=cached,
    )


def _make_chunk(document: str = "RAG uses retrieval to augment generation.") -> RetrievedChunk:
    return RetrievedChunk(
        document=document,
        score=0.91,
        source="doc.txt",
        chunk_index=0,
    )


# ── _should_sample ────────────────────────────────────────────────────────────


class TestShouldSample:
    def test_samples_when_value_below_rate(self) -> None:
        assert _should_sample(0.5, 0.3) is True

    def test_does_not_sample_when_value_equals_rate(self) -> None:
        # strictly less-than: value == rate → not sampled
        assert _should_sample(0.5, 0.5) is False

    def test_does_not_sample_when_value_above_rate(self) -> None:
        assert _should_sample(0.1, 0.9) is False

    def test_always_samples_at_rate_one(self) -> None:
        assert _should_sample(1.0, 0.9999) is True

    def test_never_samples_at_rate_zero(self) -> None:
        assert _should_sample(0.0, 0.0) is False

    def test_boundary_just_below_rate(self) -> None:
        assert _should_sample(0.1, 0.0999) is True

    def test_boundary_just_above_rate(self) -> None:
        assert _should_sample(0.1, 0.1001) is False


# ── _build_ragas_dataset ──────────────────────────────────────────────────────


class TestBuildRagasDataset:
    def test_user_input_contains_query(self) -> None:
        result = _build_ragas_dataset("What is RAG?", _make_response(), [_make_chunk()])
        assert result["user_input"] == ["What is RAG?"]

    def test_response_contains_answer(self) -> None:
        response = _make_response(answer="RAG augments LLMs with retrieval.")
        result = _build_ragas_dataset("query", response, [_make_chunk()])
        assert result["response"] == ["RAG augments LLMs with retrieval."]

    def test_retrieved_contexts_wraps_chunk_documents(self) -> None:
        chunk = _make_chunk(document="Some context text.")
        result = _build_ragas_dataset("query", _make_response(), [chunk])
        assert result["retrieved_contexts"] == [["Some context text."]]

    def test_single_sample_list_length(self) -> None:
        result = _build_ragas_dataset("query", _make_response(), [_make_chunk()])
        assert len(result["user_input"]) == 1  # type: ignore[arg-type]
        assert len(result["response"]) == 1  # type: ignore[arg-type]
        assert len(result["retrieved_contexts"]) == 1  # type: ignore[arg-type]

    def test_no_reference_key(self) -> None:
        result = _build_ragas_dataset("query", _make_response(), [_make_chunk()])
        assert "reference" not in result

    def test_multiple_chunks_all_included(self) -> None:
        chunks = [_make_chunk("Context A."), _make_chunk("Context B.")]
        result = _build_ragas_dataset("query", _make_response(), chunks)
        assert result["retrieved_contexts"] == [["Context A.", "Context B."]]

    def test_empty_chunks_produces_empty_context(self) -> None:
        result = _build_ragas_dataset("query", _make_response(), [])
        assert result["retrieved_contexts"] == [[]]

    def test_returns_dict(self) -> None:
        result = _build_ragas_dataset("query", _make_response(), [_make_chunk()])
        assert isinstance(result, dict)


# ── _scores_below_threshold ───────────────────────────────────────────────────


class TestScoresBelowThreshold:
    def test_all_above_returns_empty(self) -> None:
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.85, "llm_context_precision_without_reference": 0.80}
        assert _scores_below_threshold(scores, 0.75) == []

    def test_all_below_returns_all_metrics(self) -> None:
        scores = {"faithfulness": 0.5, "answer_relevancy": 0.6, "llm_context_precision_without_reference": 0.4}
        result = _scores_below_threshold(scores, 0.75)
        assert set(result) == {"faithfulness", "answer_relevancy", "llm_context_precision_without_reference"}

    def test_exactly_at_threshold_not_failing(self) -> None:
        # threshold is exclusive: score == threshold → not below
        scores = {"faithfulness": 0.75}
        assert _scores_below_threshold(scores, 0.75) == []

    def test_partial_failure_returns_only_failing(self) -> None:
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.5}
        result = _scores_below_threshold(scores, 0.75)
        assert result == ["answer_relevancy"]

    def test_empty_scores_returns_empty(self) -> None:
        assert _scores_below_threshold({}, 0.75) == []

    def test_single_failing_metric(self) -> None:
        scores = {"faithfulness": 0.4}
        assert _scores_below_threshold(scores, 0.75) == ["faithfulness"]

    def test_returns_list(self) -> None:
        assert isinstance(_scores_below_threshold({}, 0.5), list)


# ── _format_axiom_event ───────────────────────────────────────────────────────


class TestFormatAxiomEvent:
    def test_query_included(self) -> None:
        event = _format_axiom_event({}, "What is RAG?", _make_response(), 0.75)
        assert event["query"] == "What is RAG?"

    def test_answer_included(self) -> None:
        response = _make_response(answer="RAG stands for Retrieval-Augmented Generation.")
        event = _format_axiom_event({}, "q", response, 0.75)
        assert event["answer"] == "RAG stands for Retrieval-Augmented Generation."

    def test_complexity_included(self) -> None:
        event = _format_axiom_event({}, "q", _make_response(complexity="complex"), 0.75)
        assert event["complexity"] == "complex"

    def test_cached_false_propagated(self) -> None:
        event = _format_axiom_event({}, "q", _make_response(cached=False), 0.75)
        assert event["cached"] is False

    def test_cached_true_propagated(self) -> None:
        event = _format_axiom_event({}, "q", _make_response(cached=True), 0.75)
        assert event["cached"] is True

    def test_alert_false_when_all_above_threshold(self) -> None:
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.85}
        event = _format_axiom_event(scores, "q", _make_response(), 0.75)
        assert event["alert"] is False

    def test_alert_true_when_any_below_threshold(self) -> None:
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.5}
        event = _format_axiom_event(scores, "q", _make_response(), 0.75)
        assert event["alert"] is True

    def test_below_threshold_lists_failing_metrics(self) -> None:
        scores = {"faithfulness": 0.4, "answer_relevancy": 0.9}
        event = _format_axiom_event(scores, "q", _make_response(), 0.75)
        assert event["below_threshold"] == ["faithfulness"]

    def test_below_threshold_empty_when_all_pass(self) -> None:
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.85}
        event = _format_axiom_event(scores, "q", _make_response(), 0.75)
        assert event["below_threshold"] == []

    def test_scores_merged_into_event(self) -> None:
        scores = {"faithfulness": 0.88, "answer_relevancy": 0.76}
        event = _format_axiom_event(scores, "q", _make_response(), 0.75)
        assert event["faithfulness"] == pytest.approx(0.88)
        assert event["answer_relevancy"] == pytest.approx(0.76)

    def test_time_field_present(self) -> None:
        event = _format_axiom_event({}, "q", _make_response(), 0.75)
        assert "_time" in event
        assert isinstance(event["_time"], str)

    def test_record_fields_included_when_present(self) -> None:
        response = _make_response(with_record=True)
        event = _format_axiom_event({}, "q", response, 0.75)
        assert event["model"] == "claude-3-5-haiku-20241022"
        assert event["provider"] == "anthropic"
        assert event["total_tokens"] == 160
        assert event["cost_usd"] == pytest.approx(0.0003)
        assert event["latency_ms"] == pytest.approx(450.0)

    def test_record_fields_absent_when_no_record(self) -> None:
        response = _make_response(with_record=False)
        event = _format_axiom_event({}, "q", response, 0.75)
        assert "model" not in event
        assert "provider" not in event
        assert "total_tokens" not in event

    def test_returns_dict(self) -> None:
        event = _format_axiom_event({}, "q", _make_response(), 0.75)
        assert isinstance(event, dict)
