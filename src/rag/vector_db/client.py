from __future__ import annotations

import chromadb
from chromadb import K, Knn, Rrf, Schema, Search, SparseVectorIndexConfig
from chromadb.utils.embedding_functions import (
    ChromaCloudQwenEmbeddingFunction,
    ChromaCloudQwenEmbeddingModel,
    ChromaCloudSpladeEmbeddingFunction,
    ChromaCloudSpladeEmbeddingModel,
)

from rag.types import RetrievedChunk


# CloudClient reads CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE from env
def get_client() -> chromadb.CloudClient:
    return chromadb.CloudClient()


def _build_schema() -> Schema:
    sparse_ef = ChromaCloudSpladeEmbeddingFunction(
        model=ChromaCloudSpladeEmbeddingModel.SPLADE_PP_EN_V1,
    )
    schema = Schema()
    schema.create_index(
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT,
            embedding_function=sparse_ef,
        ),
        key="sparse_embedding",
    )
    return schema


def get_collection(
    client: chromadb.CloudClient,
    name: str,
) -> chromadb.Collection:
    dense_ef = ChromaCloudQwenEmbeddingFunction(
        model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
        # "retrieval" optimizes for document semantic search
        task="retrieval",
    )
    return client.get_or_create_collection(
        name=name,
        embedding_function=dense_ef,
        schema=_build_schema(),
    )


def hybrid_search(
    collection: chromadb.Collection,
    query: str,
    limit: int = 10,
) -> list[RetrievedChunk]:
    # limit=300 on internal ranks ensures wide coverage before RRF fusion
    dense_rank = Knn(
        query=query,
        key="#embedding",
        return_rank=True,
        limit=300,
    )
    sparse_rank = Knn(
        query=query,
        key="sparse_embedding",
        return_rank=True,
        limit=300,
    )
    # Dense weight 2x — semantics prevail over keyword matching
    rrf = Rrf(
        ranks=[dense_rank, sparse_rank],
        weights=[2.0, 1.0],
        k=60,
    )
    search = (
        Search()
        .rank(rrf)
        .limit(limit)
        .select(K.DOCUMENT, K.SCORE, "source", "chunk_index", "parent_text", "parent_id")
    )
    raw_results: list[dict[str, object]] = collection.search(search)

    chunks = [
        RetrievedChunk(
            # parent_text takes precedence — LLM receives the larger context window
            document=str(r["parent_text"]) if r.get("parent_text") else str(r.get(K.DOCUMENT, "")),
            score=float(r.get(K.SCORE, 0.0)),
            source=str(r.get("source", "")),
            chunk_index=int(r.get("chunk_index", 0)),
            parent_id=str(r["parent_id"]) if r.get("parent_id") else None,
        )
        for r in raw_results
    ]

    # Deduplicate by parent_id — multiple children of the same parent would send
    # duplicate context to the LLM. Results are already sorted by score descending,
    # so the first occurrence per parent is the highest-scoring match.
    seen_parents: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.parent_id is None:
            deduped.append(chunk)
        elif chunk.parent_id not in seen_parents:
            seen_parents.add(chunk.parent_id)
            deduped.append(chunk)

    return deduped[:limit]
