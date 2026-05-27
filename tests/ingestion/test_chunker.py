from __future__ import annotations

import pytest

from rag.ingestion.chunker import (
    ChunkConfig,
    HierarchicalChunk,
    _paragraph_aware_split,
    _split_with_overlap,
    chunk_fixed,
    chunk_hierarchical,
)


class TestChunkConfig:
    def test_overlap_equal_to_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(strategy="fixed", chunk_size=100, chunk_overlap=100, parent_chunk_size=500)

    def test_overlap_exceeds_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(strategy="fixed", chunk_size=100, chunk_overlap=150, parent_chunk_size=500)

    def test_hierarchical_parent_smaller_than_child_raises(self) -> None:
        with pytest.raises(ValueError, match="parent_chunk_size"):
            ChunkConfig(strategy="hierarchical", chunk_size=400, chunk_overlap=50, parent_chunk_size=300)

    def test_hierarchical_parent_equal_to_child_raises(self) -> None:
        with pytest.raises(ValueError, match="parent_chunk_size"):
            ChunkConfig(strategy="hierarchical", chunk_size=400, chunk_overlap=50, parent_chunk_size=400)

    def test_valid_fixed_config(self) -> None:
        config = ChunkConfig(strategy="fixed", chunk_size=100, chunk_overlap=20, parent_chunk_size=0)
        assert config.chunk_size == 100
        assert config.chunk_overlap == 20

    def test_valid_hierarchical_config(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=400, chunk_overlap=50, parent_chunk_size=1500)
        assert config.parent_chunk_size == 1500


class TestSplitWithOverlap:
    def test_empty_string(self) -> None:
        assert _split_with_overlap("", 100, 20) == []

    def test_shorter_than_chunk_size(self) -> None:
        result = _split_with_overlap("hello world", 100, 20)
        assert result == ["hello world"]

    def test_text_shorter_than_step_returns_single_chunk(self) -> None:
        # step = chunk_size - overlap = 80; text of 79 chars → only start=0 fires
        text = "a" * 79
        result = _split_with_overlap(text, 100, 20)
        assert len(result) == 1

    def test_overlap_produces_correct_chunk_count(self) -> None:
        # 180 chars, chunk_size=100, step=80 → starts at 0, 80, 160 → 3 chunks
        text = "a" * 180
        result = _split_with_overlap(text, 100, 20)
        assert len(result) == 3

    def test_overlap_shared_content(self) -> None:
        # The last 20 chars of chunk 0 must equal the first 20 chars of chunk 1
        text = "0123456789" * 20  # 200 deterministic chars
        result = _split_with_overlap(text, 100, 20)
        assert len(result) >= 2
        assert result[0][-20:] == result[1][:20]

    def test_all_chunks_within_size(self) -> None:
        text = "x" * 500
        result = _split_with_overlap(text, 100, 30)
        assert all(len(chunk) <= 100 for chunk in result)


class TestParagraphAwareSplit:
    def test_single_short_paragraph(self) -> None:
        text = "This is a short paragraph."
        result = _paragraph_aware_split(text, 1000)
        assert result == [text]

    def test_two_paragraphs_fit_in_one_chunk(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        result = _paragraph_aware_split(text, 1000)
        assert len(result) == 1
        assert "First paragraph." in result[0]
        assert "Second paragraph." in result[0]

    def test_paragraphs_overflow_into_separate_chunks(self) -> None:
        # Two paragraphs of 600 chars each — cannot fit in a 1000-char chunk together
        para_a = "A" * 600
        para_b = "B" * 600
        text = f"{para_a}\n\n{para_b}"
        result = _paragraph_aware_split(text, 1000)
        assert len(result) == 2
        assert result[0] == para_a
        assert result[1] == para_b

    def test_oversized_single_paragraph_hard_split(self) -> None:
        # 2500 chars with chunk_size=1000 → 3 chunks (1000 + 1000 + 500)
        big_para = "X" * 2500
        result = _paragraph_aware_split(big_para, 1000)
        assert len(result) == 3
        assert all(len(r) <= 1000 for r in result)

    def test_empty_paragraphs_ignored(self) -> None:
        text = "First.\n\n\n\nSecond."
        result = _paragraph_aware_split(text, 1000)
        assert len(result) == 1  # both fit and empty lines are skipped


class TestChunkFixed:
    def test_empty_text(self) -> None:
        config = ChunkConfig(strategy="fixed", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        assert chunk_fixed("", config) == []

    def test_whitespace_only(self) -> None:
        config = ChunkConfig(strategy="fixed", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        assert chunk_fixed("   \n  ", config) == []

    def test_single_chunk_for_short_text(self) -> None:
        config = ChunkConfig(strategy="fixed", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        result = chunk_fixed("hello world", config)
        assert len(result) == 1
        assert result[0] == "hello world"

    def test_chunk_count_with_overlap(self) -> None:
        config = ChunkConfig(strategy="fixed", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        # step=80: starts at 0, 80, 160, 240 → 4 chunks for 250-char text
        result = chunk_fixed("a" * 250, config)
        assert len(result) == 4


class TestChunkHierarchical:
    def test_empty_text(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        assert chunk_hierarchical("", config, source="doc.txt") == []

    def test_whitespace_only(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        assert chunk_hierarchical("   ", config, source="doc.txt") == []

    def test_returns_hierarchical_chunk_instances(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        result = chunk_hierarchical("Hello world. " * 40, config, source="doc.txt")
        assert len(result) > 0
        assert all(isinstance(r, HierarchicalChunk) for r in result)

    def test_parent_text_preserved_exactly(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        text = "A short document for testing."
        result = chunk_hierarchical(text, config, source="test.txt")
        assert len(result) == 1
        assert result[0].parent_text == text

    def test_each_pair_has_at_least_one_child(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        result = chunk_hierarchical("Hello world. " * 40, config, source="doc.txt")
        assert all(len(r.children) >= 1 for r in result)

    def test_children_fit_within_chunk_size(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        result = chunk_hierarchical("Hello world. " * 100, config, source="doc.txt")
        for pair in result:
            assert all(len(child) <= config.chunk_size for child in pair.children)

    def test_parent_ids_are_unique_within_document(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        # Enough content to generate multiple parents
        text = "\n\n".join(["Paragraph. " * 60] * 5)
        result = chunk_hierarchical(text, config, source="doc.txt")
        ids = [r.parent_id for r in result]
        assert len(ids) == len(set(ids))

    def test_different_sources_produce_different_parent_ids(self) -> None:
        config = ChunkConfig(strategy="hierarchical", chunk_size=100, chunk_overlap=20, parent_chunk_size=500)
        text = "Same content for both sources."
        result_a = chunk_hierarchical(text, config, source="doc_a.txt")
        result_b = chunk_hierarchical(text, config, source="doc_b.txt")
        assert result_a[0].parent_id != result_b[0].parent_id
