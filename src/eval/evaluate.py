"""
evaluate.py — Run report-level evaluation using golden_set.json.

Loads golden set prompts, runs the report pipeline (and optionally the
single-query baseline), judges outputs, computes retrieval metrics, and
saves comprehensive results.

Usage:
    python -m src.eval.evaluate --parser marker
    python -m src.eval.evaluate --parser marker --baseline
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.vectorstore.load_db import load_vector_db
from src.llm.load_judge_llm import load_judge_llm
from src.llm.load_generator_llm import load_generator_llm
from src.rag.pipeline_e2e import ask_openfoam, format_context_with_metadata
from src.rag.report_pipeline import run_report_pipeline
from src.eval.judge import judge_answer, judge_report
from src.eval.retrieval_metrics import compute_all_retrieval_metrics
from sentence_transformers import CrossEncoder


GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def load_golden_set():
    """Load golden set prompts from JSON file."""
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_pages_from_chunks(chunks):
    """Extract ordered list of page numbers from chunk data."""
    pages = []
    for chunk in chunks:
        page = chunk.get('page')
        if page and page != 'N/A':
            try:
                pages.append(int(page))
            except (ValueError, TypeError):
                pass
    return pages


def evaluate_report(entry, vector_db, generator_llm, judge_llm, reranker):
    """
    Run report pipeline and evaluate a single golden set entry.

    Returns dict with report result and evaluation scores.
    """
    prompt = entry["prompt"]
    entry_id = entry["id"]

    print(f"\n  [{entry_id}] Generating report...")

    # Run report pipeline
    result = run_report_pipeline(prompt, vector_db, generator_llm, reranker)

    # Build context docs for judging (reconstruct Document-like objects)
    from langchain_core.documents import Document
    context_docs = []
    for chunk in result["chunks"]:
        doc = Document(
            page_content=chunk["content_preview"],
            metadata={
                "section": chunk.get("section", "N/A"),
                "subsection": chunk.get("subsection"),
                "page": chunk.get("page", "N/A"),
                "source": chunk.get("source", "Unknown"),
                "similarity_score": chunk.get("similarity_score"),
                "rerank_score": chunk.get("rerank_score"),
            }
        )
        context_docs.append(doc)

    # Judge the report
    print(f"  [{entry_id}] Judging report...")
    reference_checklist = {
        "expected_sections": entry.get("expected_sections", []),
        "must_include_facts": entry.get("must_include_facts", []),
    }
    judge_result = judge_report(
        question=prompt,
        report=result["report"],
        context=context_docs,
        reference_checklist=reference_checklist,
        llm=judge_llm,
    )

    # Compute retrieval metrics (if relevant_pages available)
    retrieved_pages = extract_pages_from_chunks(result["chunks"])
    relevant_pages = entry.get("relevant_pages", [])
    retrieval_metrics = {}
    if relevant_pages:
        retrieval_metrics = compute_all_retrieval_metrics(retrieved_pages, relevant_pages)

    if "error" in judge_result:
        print(f"  [{entry_id}] Judge error: {judge_result['error']}")
    else:
        print(f"  [{entry_id}] Overall score: {judge_result.get('overall_score', 'N/A')}")

    return {
        "id": entry_id,
        "prompt": prompt,
        "mode": "report",
        "report": result["report"],
        "sub_questions": result["sub_questions"],
        "chunks": result["chunks"],
        "retrieval_metadata": result["retrieval_metadata"],
        "evaluation": judge_result,
        "retrieval_metrics": retrieval_metrics,
        "timestamp": datetime.now().isoformat(),
    }


def evaluate_baseline(entry, vector_db, generator_llm, judge_llm, reranker):
    """
    Run single-query baseline and evaluate a golden set entry.

    Returns dict with baseline answer and evaluation scores.
    """
    prompt = entry["prompt"]
    entry_id = entry["id"]

    print(f"  [{entry_id}] Running baseline (single-query)...")

    response, context_docs = ask_openfoam(
        prompt, vector_db, generator_llm, reranker, return_context=True
    )

    # Judge the answer using existing judge_answer
    print(f"  [{entry_id}] Judging baseline...")
    baseline_judge = judge_answer(
        question=prompt,
        answer=response,
        context=context_docs,
        llm=judge_llm,
    )

    # Build chunk data
    chunks = []
    for i, doc in enumerate(context_docs, 1):
        chunks.append({
            "rank": i,
            "section": doc.metadata.get('section', 'N/A'),
            "subsection": doc.metadata.get('subsection'),
            "page": doc.metadata.get('page', 'N/A'),
            "source": doc.metadata.get('source', 'Unknown'),
            "similarity_score": doc.metadata.get('similarity_score'),
            "rerank_score": doc.metadata.get('rerank_score'),
            "content_preview": doc.page_content[:],
        })

    # Retrieval metrics
    retrieved_pages = extract_pages_from_chunks(chunks)
    relevant_pages = entry.get("relevant_pages", [])
    retrieval_metrics = {}
    if relevant_pages:
        retrieval_metrics = compute_all_retrieval_metrics(retrieved_pages, relevant_pages)

    if "error" in baseline_judge:
        print(f"  [{entry_id}] Baseline judge error: {baseline_judge['error']}")
    else:
        print(f"  [{entry_id}] Baseline score: {baseline_judge.get('overall_score', 'N/A')}")

    return {
        "id": entry_id,
        "prompt": prompt,
        "mode": "baseline",
        "answer": response,
        "chunks": chunks,
        "evaluation": baseline_judge,
        "retrieval_metrics": retrieval_metrics,
        "timestamp": datetime.now().isoformat(),
    }


def evaluate_parser(parser_name, run_baseline=False):
    """
    Run full evaluation on all golden set entries.

    Args:
        parser_name: Which parser's vector DB to use.
        run_baseline: If True, also run single-query baseline for comparison.

    Returns:
        List of evaluation result dicts.
    """
    print(f"\nLoading golden set from {GOLDEN_SET_PATH}...")
    golden_set = load_golden_set()
    print(f"Loaded {len(golden_set)} report prompts")

    print(f"\nLoading resources for parser: {parser_name}")
    vector_db = load_vector_db(parser_name)
    generator_llm = load_generator_llm(max_tokens=8192)
    baseline_llm = load_generator_llm(max_tokens=2048) if run_baseline else None
    judge_llm = load_judge_llm()
    reranker = CrossEncoder('BAAI/bge-reranker-base', device='cpu')

    results = []

    for entry in golden_set:
        entry_id = entry["id"]
        print(f"\n{'='*60}")
        print(f"Processing {entry_id}: {entry['prompt'][:80]}...")
        print(f"{'='*60}")

        # Report evaluation
        report_result = evaluate_report(
            entry, vector_db, generator_llm, judge_llm, reranker
        )

        entry_result = {
            "id": entry_id,
            "prompt": entry["prompt"],
            "expected_sections": entry.get("expected_sections", []),
            "must_include_facts": entry.get("must_include_facts", []),
            "report": report_result,
        }

        # Baseline evaluation (optional)
        if run_baseline:
            baseline_result = evaluate_baseline(
                entry, vector_db, baseline_llm, judge_llm, reranker
            )
            entry_result["baseline"] = baseline_result

        results.append(entry_result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate report pipeline using golden set"
    )
    parser.add_argument(
        "--parser",
        required=True,
        choices=["docling", "marker", "pymupdf"],
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run single-query baseline for side-by-side comparison",
    )
    args = parser.parse_args()

    results = evaluate_parser(args.parser, run_baseline=args.baseline)

    # Save results
    output_path = Path("eval_results")
    output_path.mkdir(exist_ok=True)

    file_path = output_path / f"{args.parser}_report_evaluation.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Evaluation saved to {file_path}")
    print(f"{'='*60}")

    # Print summary
    report_scores = []
    for r in results:
        eval_data = r["report"]["evaluation"]
        if "error" not in eval_data:
            report_scores.append(eval_data.get("overall_score", 0))

    if report_scores:
        avg_score = sum(report_scores) / len(report_scores)
        print(f"\nReport Pipeline Summary:")
        print(f"  Prompts evaluated: {len(results)}")
        print(f"  Successful judges: {len(report_scores)}/{len(results)}")
        print(f"  Avg overall score: {avg_score:.2f}")

    if args.baseline:
        baseline_scores = []
        for r in results:
            if "baseline" in r:
                eval_data = r["baseline"]["evaluation"]
                if "error" not in eval_data:
                    baseline_scores.append(eval_data.get("overall_score", 0))
        if baseline_scores:
            avg_baseline = sum(baseline_scores) / len(baseline_scores)
            print(f"\nBaseline Summary:")
            print(f"  Successful judges: {len(baseline_scores)}/{len(results)}")
            print(f"  Avg overall score: {avg_baseline:.2f}")


if __name__ == "__main__":
    main()

# TO RUN EVALUATION USE THE FOLLOWING SCRIPT
# python -m src.eval.evaluate --parser marker
# python -m src.eval.evaluate --parser marker --baseline
