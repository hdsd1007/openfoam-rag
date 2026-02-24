"""
supabase_client.py — Supabase pgvector search wrapper.

Calls the match_chunks RPC function for similarity search.
"""

import os
from supabase import create_client, Client

_client: Client | None = None


def get_client() -> Client:
    """Return the singleton Supabase client."""
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
    return _client


def search_chunks(
    query_embedding: list[float],
    k: int = 10,
    parser: str = "marker",
) -> list[dict]:
    """
    Search for similar chunks using Supabase pgvector.

    Returns list of dicts with: content, section, subsection,
    subsubsection, page, source, similarity.
    """
    client = get_client()
    response = client.rpc(
        "match_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter_parser": parser,
        },
    ).execute()

    return response.data or []
