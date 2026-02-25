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


def insert_feedback(
    query: str,
    answer_hash: str,
    mode: str,
    vote: str,
    comment: str | None = None,
    answer_text: str | None = None,
) -> dict:
    """Insert a feedback row into Supabase."""
    client = get_client()
    row = {
        "query": query,
        "answer_hash": answer_hash,
        "mode": mode,
        "vote": vote,
    }
    if comment:
        row["comment"] = comment
    if answer_text:
        row["answer_text"] = answer_text
    response = client.table("feedback").insert(row).execute()
    return response.data[0] if response.data else {}


def fetch_feedback(
    vote_filter: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Fetch feedback rows from Supabase.

    Args:
        vote_filter: If set, only return rows with this vote ('up' or 'down').
        limit: Max rows to return (default 500).

    Returns:
        List of feedback row dicts.
    """
    client = get_client()
    query = client.table("feedback").select("*").order("created_at", desc=True).limit(limit)
    if vote_filter:
        query = query.eq("vote", vote_filter)
    response = query.execute()
    return response.data or []
