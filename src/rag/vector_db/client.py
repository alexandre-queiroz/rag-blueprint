from __future__ import annotations

import os

import chromadb
from chromadb import K, Knn, Rrf, Schema, Search, SparseVectorIndexConfig, VectorIndexConfig
from chromadb.utils.embedding_functions import (
    ChromaCloudQwenEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import (
    ChromaCloudQwenEmbeddingModel,
)
from chromadb.utils.embedding_functions.chroma_cloud_splade_embedding_function import (
    ChromaCloudSpladeEmbeddingModel,
)

from rag.types import RetrievedChunk


def get_client() -> chromadb.CloudClient:
    """Create a Chroma Cloud client from environment variables.

    Reads CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE explicitly —
    CloudClient() only auto-reads CHROMA_API_KEY; tenant and database must
    be passed as arguments.
    """
    return chromadb.CloudClient(
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
        api_key=os.environ["CHROMA_API_KEY"],
    )


def _build_schema() -> Schema:
    """Build a hybrid search schema with Qwen dense + Splade sparse indexes.

    Both embedding functions are configured inside the schema so that
    get_or_create_collection receives only schema= (no embedding_function=).
    Passing both simultaneously causes a server-side conflict in chromadb 1.5.x.
    """
    dense_ef = ChromaCloudQwenEmbeddingFunction(
        model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
        task="retrieval",
    )
    sparse_ef = ChromaCloudSpladeEmbeddingFunction(
        model=ChromaCloudSpladeEmbeddingModel.SPLADE_PP_EN_V1,
    )
    schema = Schema()
    # Dense vector index — Qwen3 via Chroma Cloud
    schema.create_index(config=VectorIndexConfig(embedding_function=dense_ef))
    # Sparse vector index — Splade PP EN v1 via Chroma Cloud
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
    """Get or create a Chroma collection with Qwen dense + Splade sparse hybrid search.

    embedding_function is omitted from get_or_create_collection — the dense EF
    is embedded inside the schema via VectorIndexConfig to avoid the server-side
    conflict that occurs when both schema= and embedding_function= are provided.
    """
    return client.get_or_create_collection(
        name=name,
        schema=_build_schema(),
        embedding_function=None,
    )


def hybrid_search(
    collection: chromadb.Collection,
    query: str,
    limit: int = 10,
) -> list[RetrievedChunk]:
    """Run RRF hybrid search and return deduplicated RetrievedChunks.

    The SearchResult returned by collection.search() uses a columnar format:
    results["documents"][0], results["scores"][0], results["metadatas"][0]
    are parallel lists for the single query. Scores are negated RRF values
    (more negative = more relevant); abs() normalises to [0, ∞).
    """
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
    result = collection.search(search)

    # Unpack columnar SearchResult into per-row dicts
    documents: list[str] = result["documents"][0] or []
    scores: list[float] = result["scores"][0] or []
    metadatas: list[dict[str, object]] = result["metadatas"][0] or []

    chunks = [
        RetrievedChunk(
            # parent_text takes precedence — LLM receives the larger context window
            document=str(meta.get("parent_text") or doc),
            score=abs(float(score)),   # Chroma returns negated RRF; abs() → [0, ∞)
            source=str(meta.get("source", "")),
            chunk_index=int(meta.get("chunk_index", 0)),
            parent_id=str(meta["parent_id"]) if meta.get("parent_id") else None,
        )
        for doc, score, meta in zip(documents, scores, metadatas)
    ]

    # Deduplicate by parent_id — multiple children of the same parent would send
    # duplicate context to the LLM. Results are already sorted by relevance,
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
