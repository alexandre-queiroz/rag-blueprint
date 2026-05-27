from __future__ import annotations

import pytest

from rag.classifier.heuristics import _rough_entity_count, heuristic_classify


class TestHeuristicClassifySimple:
    def test_simple_opener_what_is(self) -> None:
        assert heuristic_classify("What is RAG?") == "simple"

    def test_simple_opener_define(self) -> None:
        assert heuristic_classify("Define vector database.") == "simple"

    def test_simple_opener_who_is(self) -> None:
        assert heuristic_classify("Who is Alan Turing?") == "simple"

    def test_simple_opener_when(self) -> None:
        assert heuristic_classify("When was GPT-3 released?") == "simple"

    def test_simple_very_short_no_signals(self) -> None:
        # 4 words, no medium or complex signals
        assert heuristic_classify("list the main components") == "simple"

    def test_simple_opener_ignored_when_too_long(self) -> None:
        # "What is" opener but 12 words → does not meet the _SIMPLE_OPENER_MAX_WORDS limit
        query = "What is the difference between dense and sparse retrieval in RAG?"
        # contains "difference between" → complex takes priority anyway
        assert heuristic_classify(query) == "complex"


class TestHeuristicClassifyComplex:
    def test_complex_compare_signal(self) -> None:
        assert heuristic_classify("Compare BM25 and dense retrieval.") == "complex"

    def test_complex_analyze_signal(self) -> None:
        assert heuristic_classify("Analyze the trade-offs of vector databases.") == "complex"

    def test_complex_evaluate_signal(self) -> None:
        assert heuristic_classify("Evaluate the pros and cons of LangChain.") == "complex"

    def test_complex_difference_between(self) -> None:
        assert heuristic_classify("What is the difference between HNSW and IVF indexes?") == "complex"

    def test_complex_very_long_query(self) -> None:
        # 31 words — crosses _COMPLEX_MIN_WORDS threshold
        query = " ".join(["word"] * 31)
        assert heuristic_classify(query) == "complex"

    def test_complex_entity_count_threshold(self) -> None:
        # 3 named entities → _COMPLEX_ENTITY_THRESHOLD
        assert heuristic_classify("Compare OpenAI, Anthropic, and Google.") == "complex"

    def test_complex_signal_overrides_short_length(self) -> None:
        # Only 5 words but contains "compare"
        assert heuristic_classify("Compare Claude and GPT-4.") == "complex"

    def test_complex_signal_overrides_simple_opener(self) -> None:
        # Starts with "What" but contains "difference between" — complex wins
        assert heuristic_classify("What is the difference between Claude and GPT-4?") == "complex"


class TestHeuristicClassifyMedium:
    def test_medium_how_to(self) -> None:
        assert heuristic_classify("How to implement semantic search in Python?") == "medium"

    def test_medium_how_does(self) -> None:
        assert heuristic_classify("How does a transformer model process text sequences?") == "medium"

    def test_medium_explain(self) -> None:
        assert heuristic_classify("Explain the attention mechanism in transformers.") == "medium"

    def test_medium_describe(self) -> None:
        assert heuristic_classify("Describe the steps to fine-tune a language model.") == "medium"


class TestHeuristicClassifyNone:
    def test_none_ambiguous_mid_length(self) -> None:
        # 9 words, no signals — should escalate to stage 2
        result = heuristic_classify("The benefits of using embeddings in modern applications")
        assert result is None

    def test_none_yes_no_question(self) -> None:
        result = heuristic_classify("Is semantic search better than keyword search?")
        assert result is None

    def test_none_neutral_statement(self) -> None:
        result = heuristic_classify("Large language models have changed the industry significantly")
        assert result is None


class TestRoughEntityCount:
    def test_no_entities(self) -> None:
        assert _rough_entity_count("what is the capital of france") == 0

    def test_single_entity(self) -> None:
        assert _rough_entity_count("What is Google?") == 1

    def test_three_entities(self) -> None:
        assert _rough_entity_count("Compare OpenAI, Anthropic, and Google.") == 3

    def test_punctuation_attached_to_entity(self) -> None:
        # OpenAI at position 0 is excluded (first-word rule); Anthropic and Google count
        # "The" is the first word, so all three companies are counted
        assert _rough_entity_count("The companies OpenAI, Anthropic, and Google compete.") == 3

    def test_first_word_not_counted(self) -> None:
        # "Compare" is the first word and should be excluded
        assert _rough_entity_count("Compare Google and Microsoft.") == 2

    def test_hyphenated_token_not_counted(self) -> None:
        # "GPT-4" contains a hyphen → isalpha() fails even after rstrip
        assert _rough_entity_count("Compare GPT-4 and Claude.") == 1  # only Claude

    def test_all_lowercase(self) -> None:
        assert _rough_entity_count("how does semantic search work?") == 0
