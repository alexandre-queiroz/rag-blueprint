from __future__ import annotations

from pathlib import Path
from typing import cast

import chromadb

from rag.config import IngestionConfig
from rag.ingestion.chunker import (
    ChunkConfig,
    ChunkStrategy,
    _chunk_id,
    chunk_fixed,
    chunk_hierarchical,
)


def config_to_chunk_config(ingestion: IngestionConfig) -> ChunkConfig:
    """Convert the YAML-sourced IngestionConfig into a validated ChunkConfig."""
    return ChunkConfig(
        strategy=cast(ChunkStrategy, ingestion.strategy),
        chunk_size=ingestion.chunk_size,
        chunk_overlap=ingestion.chunk_overlap,
        parent_chunk_size=ingestion.parent_chunk_size,
    )


def ingest_file(
    collection: chromadb.Collection,
    file_path: str | Path,
    config: ChunkConfig,
) -> int:
    """Read a UTF-8 text file, chunk it, and index all chunks into the collection.

    Returns the number of child chunks added.
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return ingest_text(collection, text, source=path.name, config=config)


def ingest_text(
    collection: chromadb.Collection,
    text: str,
    source: str,
    config: ChunkConfig,
) -> int:
    """Chunk raw text and index all chunks into the collection.

    Returns the number of child chunks added.
    """
    if config.strategy == "fixed":
        return _ingest_flat(collection, text, source, config)
    return _ingest_hierarchical(collection, text, source, config)


def _ingest_flat(
    collection: chromadb.Collection,
    text: str,
    source: str,
    config: ChunkConfig,
) -> int:
    chunks = chunk_fixed(text, config)
    if not chunks:
        return 0

    ids = [_chunk_id(source, i) for i in range(len(chunks))]
    metadatas: list[dict[str, str | int]] = [
        {"source": source, "chunk_index": i} for i in range(len(chunks))
    ]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def _ingest_hierarchical(
    collection: chromadb.Collection,
    text: str,
    source: str,
    config: ChunkConfig,
) -> int:
    """Index child chunks with parent metadata.

    Chroma Cloud generates embeddings for each child chunk server-side.
    Parent text is stored in child metadata — hybrid_search returns the parent
    to the LLM instead of the smaller child text.
    """
    pairs = chunk_hierarchical(text, config, source=source)
    if not pairs:
        return 0

    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    child_counter = 0
    for pair in pairs:
        for child_text in pair.children:
            documents.append(child_text)
            ids.append(_chunk_id(source, child_counter))
            metadatas.append({
                "source": source,
                "chunk_index": child_counter,
                "parent_id": pair.parent_id,
                "parent_text": pair.parent_text,
            })
            child_counter += 1

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    return child_counter
