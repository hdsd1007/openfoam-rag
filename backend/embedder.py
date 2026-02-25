"""
embedder.py — Singleton embedding model for the web backend.

Uses sentence-transformers with all-MiniLM-L12-v2 (384-dim).
HuggingFace Spaces provides ~2GB RAM — enough for PyTorch models.
"""

_model = None


def get_model():
    """Return the singleton SentenceTransformer model (lazy-loaded)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    return _model


def embed_query(text: str) -> list[float]:
    """Embed a single query string, returns normalized 384-dim vector."""
    model = get_model()
    vector = model.encode([text], normalize_embeddings=True)[0]
    return vector.tolist()
