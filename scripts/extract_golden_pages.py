"""
extract_golden_pages.py — Retrieval-only page extraction for golden set prompts.

Runs decompose + retrieve (NO generation) for each golden set prompt,
extracts source:page pairs from retrieved chunks, and writes a consolidated
pages_summary.json for auto-populating relevant_pages in golden_set.json.

Only costs 1 Gemini API call per prompt (for decomposition).
No report generation = fast + cheap.

Usage (on Kaggle, after setup):
    python scripts/extract_golden_pages.py --parser marker

Output:
    eval_results/pages_summary.json
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from src.vectorstore.load_db import load_vector_db
from src.llm.load_generator_llm import load_generator_llm
from src.rag.report_pipeline import decompose_query, retrieve_for_report
from sentence_transformers import CrossEncoder


GOLDEN_SET_PATH = Path("src/eval/golden_set.json")


def extract_pages_with_scores(chunks):
    """Extract unique source:page pairs with best rerank score."""
    pages = {}
    for c in chunks:
        source = c.get("source", "Unknown")
        page = c.get("page", "N/A")
        if page == "N/A":
            continue
        key = f"{source}:{page}"
        score = c.get("rerank_score", 0)
        if key not in pages or score > pages[key]:
            pages[key] = round(score, 4)
    return pages


def main():
    parser = argparse.ArgumentParser(
        description="Extract relevant pages for golden set prompts (retrieval-only)"
    )
    parser.add_argument("--parser", required=True, choices=["docling", "marker", "pymupdf"])
    parser.add_argument("--k", type=int, default=20, help="Dense retrieval k per sub-question")
    parser.add_argument("--top-n", type=int, default=7, help="Rerank top-n per sub-question")
    parser.add_argument("--max-chunks", type=int, default=25, help="Max unique chunks after dedup")
    parser.add_argument("--entries", nargs="*", default=None,
                        help="Specific entry IDs to run (e.g. R02 R03). Default: all.")
    args = parser.parse_args()

    # Load golden set
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    # Filter entries if specified
    if args.entries:
        golden_set = [e for e in golden_set if e["id"] in args.entries]
        print(f"Running {len(golden_set)} selected entries: {[e['id'] for e in golden_set]}")
    else:
        print(f"Running all {len(golden_set)} entries")

    # Load resources
    print(f"\nLoading vector DB for parser: {args.parser}")
    vector_db = load_vector_db(args.parser)

    print("Loading generator LLM (for decomposition only)...")
    llm = load_generator_llm(max_tokens=1024)  # small, only for decomposition

    print("Loading reranker...")
    reranker = CrossEncoder('BAAI/bge-reranker-base', device='cpu')

    # Process each entry
    summary = {}

    for entry in golden_set:
        entry_id = entry["id"]
        prompt = entry["prompt"]
        print(f"\n{'='*60}")
        print(f"[{entry_id}] {prompt[:80]}...")
        print(f"{'='*60}")

        # Step 1: Decompose into sub-questions
        print(f"  Decomposing into sub-questions...")
        sub_questions = decompose_query(prompt, llm)
        print(f"  Sub-questions ({len(sub_questions)}):")
        for i, sq in enumerate(sub_questions, 1):
            print(f"    {i}. {sq}")

        # Step 2: Retrieve + rerank (NO generation)
        print(f"  Retrieving (k={args.k}, top_n={args.top_n}, max_chunks={args.max_chunks})...")
        merged_docs, retrieval_metadata = retrieve_for_report(
            sub_questions, vector_db, reranker,
            k_per_query=args.k,
            top_n_per_query=args.top_n,
            max_unique_chunks=args.max_chunks,
        )

        # Also do a direct retrieval on the original prompt for extra coverage
        print(f"  Also retrieving on original prompt for extra coverage...")
        direct_docs = vector_db.similarity_search_with_score(prompt, k=args.k)
        direct_list = []
        for doc, score in direct_docs:
            doc.metadata['similarity_score'] = round(float(score), 4)
            direct_list.append(doc)

        # Rerank direct results
        pairs = [[prompt, doc.page_content] for doc in direct_list]
        rerank_scores = reranker.predict(pairs)
        for doc, score in zip(direct_list, rerank_scores):
            doc.metadata['rerank_score'] = round(float(score), 4)

        # Build chunk data from merged docs
        chunks = []
        for i, doc in enumerate(merged_docs, 1):
            chunks.append({
                "rank": i,
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "section": doc.metadata.get('section', 'N/A'),
                "rerank_score": doc.metadata.get('rerank_score'),
                "origin": "decomposed",
            })

        # Add direct retrieval chunks
        direct_ranked = sorted(direct_list, key=lambda x: x.metadata['rerank_score'], reverse=True)[:args.top_n]
        for i, doc in enumerate(direct_ranked, len(chunks) + 1):
            chunks.append({
                "rank": i,
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "section": doc.metadata.get('section', 'N/A'),
                "rerank_score": doc.metadata.get('rerank_score'),
                "origin": "direct",
            })

        # Extract unique pages with best scores
        pages_with_scores = extract_pages_with_scores(chunks)

        # Sort by score descending
        sorted_pages = sorted(pages_with_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Retrieved {len(sorted_pages)} unique pages:")
        for page, score in sorted_pages:
            marker = " ***" if score >= 0.7 else " **" if score >= 0.4 else ""
            print(f"    {page} (score={score:.4f}){marker}")

        summary[entry_id] = {
            "prompt": prompt[:100],
            "sub_questions": sub_questions,
            "retrieval_metadata": retrieval_metadata,
            "pages": {page: score for page, score in sorted_pages},
            "pages_list": [page for page, score in sorted_pages],
            "high_confidence": [page for page, score in sorted_pages if score >= 0.4],
        }

    # Save summary
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "pages_summary.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Pages summary saved to: {output_path}")
    print(f"{'='*60}")

    # Print quick overview
    print(f"\nOverview:")
    for entry_id, data in summary.items():
        total = len(data["pages"])
        high = len(data["high_confidence"])
        print(f"  {entry_id}: {total} total pages, {high} high-confidence (score >= 0.4)")


if __name__ == "__main__":
    main()
