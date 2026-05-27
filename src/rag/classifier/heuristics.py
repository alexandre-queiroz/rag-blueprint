from __future__ import annotations

from rag.types import ComplexityLabel

# Openers that strongly signal a simple factual lookup
_SIMPLE_OPENERS: tuple[str, ...] = (
    "what is ",
    "what are ",
    "what was ",
    "what were ",
    "who is ",
    "who was ",
    "who are ",
    "when was ",
    "when did ",
    "when is ",
    "where is ",
    "where was ",
    "define ",
    "list ",
    "name ",
)

# Keywords that strongly signal multi-concept or comparative reasoning
_COMPLEX_SIGNALS: tuple[str, ...] = (
    "compare",
    "contrast",
    "analyze",
    "analyse",
    "evaluate",
    "trade-off",
    "trade off",
    "pros and cons",
    "implications of",
    "explain why",
    "difference between",
    "differences between",
    "relationship between",
    "similarities and differences",
)

# Keywords that signal procedural or explanatory queries
_MEDIUM_SIGNALS: tuple[str, ...] = (
    "how to",
    "how do",
    "how does",
    "how can",
    "explain ",
    "describe ",
    "walk me through",
    "what are the steps",
    "what are the differences",
)

_SIMPLE_MAX_WORDS = 5
_SIMPLE_OPENER_MAX_WORDS = 10
_COMPLEX_MIN_WORDS = 30
_COMPLEX_ENTITY_THRESHOLD = 3


def heuristic_classify(query: str) -> ComplexityLabel | None:
    """Stage 1 classifier — zero-cost keyword and length rules.

    Covers ~70% of queries. Returns None when the query is ambiguous
    and must escalate to embedding similarity (stage 2).
    """
    lower = query.lower().strip()
    word_count = len(lower.split())

    # Complex signals take priority — they override all length heuristics
    if any(signal in lower for signal in _COMPLEX_SIGNALS):
        return "complex"

    # Multiple named entities suggest a multi-concept or comparative query
    if _rough_entity_count(query) >= _COMPLEX_ENTITY_THRESHOLD:
        return "complex"

    # Very long queries almost always require multi-concept reasoning
    if word_count > _COMPLEX_MIN_WORDS:
        return "complex"

    # Simple opener + short length = confident factual lookup
    if any(lower.startswith(opener) for opener in _SIMPLE_OPENERS) and word_count <= _SIMPLE_OPENER_MAX_WORDS:
        return "simple"

    # Very short queries with no escalation signal are simple
    if word_count <= _SIMPLE_MAX_WORDS:
        return "simple"

    # Procedural or explanatory queries
    if any(signal in lower for signal in _MEDIUM_SIGNALS):
        return "medium"

    # Inconclusive — escalate to embedding similarity
    return None


def _rough_entity_count(query: str) -> int:
    """Count mid-sentence capitalized alphabetic words as a proxy for named entities."""
    words = query.split()
    return sum(1 for w in words[1:] if w and w[0].isupper() and w.rstrip(".,?!:;()").isalpha())

