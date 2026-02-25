"""
reranker.py — CrossEncoder reranker using BAAI/bge-reranker-base.

Matches the local pipeline (src/rag/pipeline_e2e.py) exactly.
HuggingFace Spaces provides ~2GB RAM — enough for the ~278MB model.
"""

_reranker = None


def _get_reranker():
    """Return the singleton CrossEncoder model (lazy-loaded)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 5,
    api_key: str | None = None,
) -> list[dict]:
    """
    Rerank chunks using CrossEncoder and return top_n.

    Each chunk dict must have a 'content' key.
    Returns chunks sorted by cross-encoder relevance (best first).
    api_key is accepted but unused (backward compat with app.py).
    """
    if not chunks:
        return []

    if len(chunks) <= top_n:
        return chunks

    reranker = _get_reranker()

    # Build query-chunk pairs (mirrors src/rag/pipeline_e2e.py lines 61-62)
    pairs = [[query, chunk["content"]] for chunk in chunks]
    scores = reranker.predict(pairs)

    # Attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    # Sort by rerank score descending, take top_n
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
    return reranked
