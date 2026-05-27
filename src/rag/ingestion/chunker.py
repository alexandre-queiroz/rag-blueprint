from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

ChunkStrategy = Literal["fixed", "hierarchical"]


@dataclass(frozen=True)
class ChunkConfig:
    strategy: ChunkStrategy
    chunk_size: int         # characters — child size for hierarchical
    chunk_overlap: int      # characters — child overlap within a parent
    parent_chunk_size: int  # characters — only used for "hierarchical"

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.strategy == "hierarchical" and self.parent_chunk_size <= self.chunk_size:
            raise ValueError(
                f"parent_chunk_size ({self.parent_chunk_size}) must be greater than "
                f"chunk_size ({self.chunk_size}) for hierarchical strategy"
            )


@dataclass
class HierarchicalChunk:
    """One parent chunk and its derived children.

    Children are indexed in Chroma for precise vector search.
    Parent text is stored in child metadata and returned to the LLM as context.
    """

    parent_id: str
    parent_text: str
    children: list[str] = field(default_factory=list)


def chunk_fixed(text: str, config: ChunkConfig) -> list[str]:
    """Sliding-window split with overlap — ignores paragraph boundaries."""
    if not text.strip():
        return []
    return _split_with_overlap(text, config.chunk_size, config.chunk_overlap)


def chunk_hierarchical(text: str, config: ChunkConfig, source: str) -> list[HierarchicalChunk]:
    """Parent-child chunking.

    1. Splits text into large parent chunks (paragraph-aware where possible).
    2. Further splits each parent into smaller child chunks with overlap.

    Retrieval finds the child; parent text is returned to the LLM for context.
    """
    if not text.strip():
        return []

    parent_texts = _paragraph_aware_split(text, config.parent_chunk_size)
    result: list[HierarchicalChunk] = []

    for idx, parent_text in enumerate(parent_texts):
        parent_id = _chunk_id(f"parent:{source}", idx)
        children = _split_with_overlap(parent_text, config.chunk_size, config.chunk_overlap)
        result.append(
            HierarchicalChunk(parent_id=parent_id, parent_text=parent_text, children=children)
        )

    return result


def _split_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Sliding-window character split with overlap."""
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _paragraph_aware_split(text: str, chunk_size: int) -> list[str]:
    """Non-overlapping split that respects paragraph boundaries where possible.

    Used to generate parent chunks — overlapping parents would duplicate context
    and inflate the metadata stored alongside each child chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            # Single paragraph exceeds limit — hard split, no overlap at parent level
            for i in range(0, len(para), chunk_size):
                chunk = para[i : i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
        elif current and len(current) + 2 + len(para) > chunk_size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}".strip() if current else para

    if current:
        chunks.append(current)

    return chunks


def _chunk_id(source: str, index: int) -> str:
    """Deterministic 16-character ID derived from source path and chunk index."""
    raw = f"{source}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
