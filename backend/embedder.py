"""
embedder.py — Singleton embedding model for the web backend.

Loads sentence-transformers/all-MiniLM-L12-v2 once at startup (~130MB RAM).
"""

from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    """Return the singleton SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    return _model


def embed_query(text: str) -> list[float]:
    """Embed a single query string, returns normalized 384-dim vector."""
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()
