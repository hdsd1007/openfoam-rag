
"""
migrate_chroma_to_supabase.py — One-time migration from ChromaDB to Supabase pgvector.

Uses chromadb directly (no LangChain, no embedding model needed —
embeddings are already stored in ChromaDB).

Usage:
    python scripts/migrate_chroma_to_supabase.py

Requires environment variables:
    SUPABASE_URL  — your Supabase project URL
    SUPABASE_KEY  — your Supabase service_role key
"""

import os
import sys
import chromadb
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "marker_db")
BATCH_SIZE = 100


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Set SUPABASE_URL and SUPABASE_KEY environment variables.")
        sys.exit(1)

    # Load ChromaDB directly (no embedding model needed)
    print("Loading ChromaDB from", os.path.abspath(CHROMA_DB_PATH))
    client = chromadb.PersistentClient(path=os.path.abspath(CHROMA_DB_PATH))
    collections = client.list_collections()
    print(f"Collections found: {[c.name for c in collections]}")

    if not collections:
        print("ERROR: No collections found in ChromaDB.")
        sys.exit(1)

    collection = collections[0]
    print(f"Using collection: {collection.name}")

    # Extract everything (embeddings are already stored)
    result = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = result["ids"]
    documents = result["documents"]
    metadatas = result["metadatas"]
    embeddings_list = result["embeddings"]

    total = len(ids)
    print(f"Found {total} chunks in ChromaDB")

    if total == 0:
        print("Nothing to migrate.")
        return

    # Verify embeddings exist
    if embeddings_list is None or len(embeddings_list) == 0:
        print("ERROR: No embeddings found in ChromaDB. Cannot migrate.")
        sys.exit(1)

    print(f"Embedding dimension: {len(embeddings_list[0])}")

    # Connect to Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    inserted = 0
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = []
        for i in range(start, end):
            meta = metadatas[i] or {}
            row = {
                "content": documents[i],
                "section": meta.get("section", "N/A"),
                "subsection": meta.get("subsection", "N/A"),
                "subsubsection": meta.get("subsubsection", "N/A"),
                "page": str(meta.get("page", "N/A")),
                "source": meta.get("source", "Unknown"),
                "parser": "marker",
                "word_count": meta.get("word_count"),
                "token_count": meta.get("token_count"),
                "embedding": embeddings_list[i],
            }
            batch.append(row)

        supabase.table("chunks").insert(batch).execute()
        inserted += len(batch)
        print(f"  Inserted {inserted}/{total}")

    print(f"\nMigration complete. {inserted} rows inserted into Supabase.")

    # Verify
    count_resp = supabase.table("chunks").select("id", count="exact").execute()
    print(f"Supabase row count: {count_resp.count}")
    if count_resp.count == total:
        print("Row counts match — migration verified.")
    else:
        print(f"WARNING: ChromaDB has {total} chunks, Supabase has {count_resp.count}.")


if __name__ == "__main__":
    main()
