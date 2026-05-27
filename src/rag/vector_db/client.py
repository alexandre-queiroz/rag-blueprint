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
        .select(K.DOCUMENT, K.SCORE, "source", "chunk_index")
    )
    raw_results: list[dict[str, object]] = collection.search(search)
    return [
        RetrievedChunk(
            document=str(r.get(K.DOCUMENT, "")),
            score=float(r.get(K.SCORE, 0.0)),
            source=str(r.get("source", "")),
            chunk_index=int(r.get("chunk_index", 0)),
        )
        for r in raw_results
    ]
