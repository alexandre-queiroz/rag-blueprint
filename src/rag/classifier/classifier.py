from __future__ import annotations

import asyncio
import math
import os
from typing import cast

import litellm

from rag.classifier.heuristics import heuristic_classify
from rag.config import ClassifierConfig
from rag.types import ComplexityLabel

# Minimum cosine similarity to accept an embedding-based classification.
# Queries below this threshold escalate to the LLM stage.
_SIMILARITY_THRESHOLD = 0.82

# Embedding model used for stage 2 — same as the semantic cache for consistency.
# Reads OPENAI_API_KEY from env at call time.
_EMBED_MODEL = "text-embedding-3-small"

# Canonical labeled queries that anchor the embedding similarity space.
# These are embedded once on first use and cached in memory.
_CANONICAL: dict[ComplexityLabel, list[str]] = {
    "simple": [
        "What is RAG?",
        "Who invented the transformer architecture?",
        "What does LLM stand for?",
        "When was GPT-3 released?",
        "Define vector database.",
        "What is a context window?",
    ],
    "medium": [
        "How does retrieval-augmented generation work?",
        "Explain the attention mechanism in transformers.",
        "How do I implement semantic search?",
        "Describe the difference between BM25 and dense retrieval.",
        "What are the steps to fine-tune a language model?",
        "How does a circuit breaker pattern work in software?",
    ],
    "complex": [
        "Compare the trade-offs between sparse and dense retrieval for production RAG.",
        "Analyze the implications of using a smaller embedding model for semantic cache versus retrieval quality.",
        "Evaluate the pros and cons of different chunking strategies for hierarchical document retrieval.",
        "Explain why circuit breakers are necessary in multi-provider LLM architectures and what failure modes they prevent.",
        "What is the relationship between context window size, chunk size, and answer quality in RAG systems?",
        "How do cost, latency, and quality trade-offs differ between Claude Haiku, Sonnet, and GPT-4o for production workloads?",
    ],
}

# Module-level cache populated on first embedding-stage call.
_canonical_embeddings: dict[ComplexityLabel, list[list[float]]] | None = None
_canonical_lock = asyncio.Lock()


# LiteLLM provider prefix map — classifier is the only non-gateway component
# that calls an LLM directly (dedicated budget, no fallback, no circuit breaker).
_LITELLM_PREFIX: dict[str, str] = {
    "anthropic": "anthropic",
    "google": "gemini",  # Google AI Studio (Gemini API), not Vertex AI
    "openai": "openai",
}


async def classify(query: str, config: ClassifierConfig) -> ComplexityLabel:
    """Three-stage complexity classifier.

    Stage 1 — heuristics: keyword and length rules, zero cost (~70% of queries).
    Stage 2 — embedding similarity: cosine distance to canonical examples (~20%).
    Stage 3 — LLM call: Gemini 2.5 Flash Lite with structured output (~10%).
    """
    result = heuristic_classify(query)
    if result is not None:
        return result

    result = await _embedding_classify(query)
    if result is not None:
        return result

    return await _llm_classify(query, config)


async def _embedding_classify(query: str) -> ComplexityLabel | None:
    """Stage 2: cosine similarity against labeled canonical queries.

    Returns None when the top similarity is below _SIMILARITY_THRESHOLD,
    escalating to the LLM stage.
    """
    global _canonical_embeddings
    if _canonical_embeddings is None:
        async with _canonical_lock:
            if _canonical_embeddings is None:
                _canonical_embeddings = await _build_canonical_embeddings()


    query_vec = await _embed(query)

    best_label: ComplexityLabel = "medium"
    best_score = -1.0

    for label, vecs in _canonical_embeddings.items():
        score = max(_cosine_similarity(query_vec, v) for v in vecs)
        if score > best_score:
            best_score = score
            best_label = label

    if best_score >= _SIMILARITY_THRESHOLD:
        return best_label
    return None


async def _llm_classify(query: str, config: ClassifierConfig) -> ComplexityLabel:
    """Stage 3: LLM call with a single-word structured prompt.

    Falls back to 'medium' if the model returns an unexpected response.
    """
    prefix = _LITELLM_PREFIX.get(config.provider, config.provider)
    model = f"{prefix}/{config.model}"
    prompt = (
        "Classify the query into exactly one complexity label.\n\n"
        "Labels:\n"
        "  simple  — factual lookups, definitions, single-concept questions\n"
        "  medium  — explanations, how-to questions, multi-step reasoning\n"
        "  complex — comparative analysis, trade-off evaluation, multi-concept synthesis\n\n"
        f"Query: {query}\n\n"
        "Reply with exactly one word: simple, medium, or complex."
    )
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.max_tokens,
        api_key=os.environ.get(config.api_key_env, ""),
    )
    raw = str(response.choices[0].message.content).strip().lower()
    if raw not in ("simple", "medium", "complex"):
        return "medium"
    return cast(ComplexityLabel, raw)


async def _build_canonical_embeddings() -> dict[ComplexityLabel, list[list[float]]]:
    """Batch-embed all canonical examples in a single API call and cache in memory."""
    labels: list[ComplexityLabel] = []
    texts: list[str] = []
    for label, examples in _CANONICAL.items():
        for text in examples:
            labels.append(label)
            texts.append(text)

    response = await litellm.aembedding(
        model=_EMBED_MODEL,
        input=texts,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    result: dict[ComplexityLabel, list[list[float]]] = {
        "simple": [],
        "medium": [],
        "complex": [],
    }
    for i, label in enumerate(labels):
        result[label].append([float(x) for x in response.data[i].embedding])

    return result


async def _embed(text: str) -> list[float]:
    """Embed a single string using the stage-2 embedding model."""
    response = await litellm.aembedding(
        model=_EMBED_MODEL,
        input=[text],
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )
    return [float(x) for x in response.data[0].embedding]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
