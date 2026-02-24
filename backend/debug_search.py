"""Quick debug script to see what dense search returns."""
from dotenv import load_dotenv
load_dotenv()

from embedder import embed_query
from supabase_client import search_chunks

import sys
query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Explain fvSchemes"
vec = embed_query(query)
results = search_chunks(vec, k=15)

print(f"Query: {query}")
print(f"Results: {len(results)}\n")
for i, r in enumerate(results):
    section = (r.get("section") or "?")[:60]
    content = r["content"][:80].replace("\n", " ")
    sim = r.get("similarity", 0)
    print(f"{i+1:2d}. sim={sim:.4f} | {section}")
    print(f"    {content}...")
    print()
