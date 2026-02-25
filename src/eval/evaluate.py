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
from src.llm.token_tracker import tracker
from src.rag.pipeline_e2e import ask_openfoam, format_context_with_metadata
from src.rag.report_pipeline import run_report_pipeline, generate_report
from src.eval.judge import judge_answer, judge_report
from src.eval.retrieval_metrics import compute_all_retrieval_metrics
from sentence_transformers import CrossEncoder


GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def load_golden_set():
    """Load golden set prompts from JSON file."""
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_pages_from_chunks(chunks):
    """Extract ordered list of source:page identifiers from chunk data.

    Returns strings like 'UserGuide.pdf:75' to disambiguate pages
    across different PDFs. Matches the format used in golden_set.json
    relevant_pages.
    """
    pages = []
    for chunk in chunks:
        page = chunk.get('page')
        source = chunk.get('source', 'Unknown')
        if page and page != 'N/A':
            pages.append(f"{source}:{page}")
    return pages


def evaluate_report(entry, vector_db, generator_llm, judge_llm, reranker):
    """
    Run report pipeline and evaluate a single golden set entry.

    Returns dict with report result and evaluation scores.
    """
    prompt = entry["prompt"]
    entry_id = entry["id"]

    print(f"\n  [{entry_id}] Generating report...")

    # Run report pipeline (with token tracking)
    result = run_report_pipeline(
        prompt, vector_db, generator_llm, reranker,
        track_tokens_prefix=f"{entry_id}_report",
    )

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

    # Judge the report (with token tracking)
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
        track_tokens=f"{entry_id}_report_judge",
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


def evaluate_baseline(entry, vector_db, generator_llm, judge_llm, reranker,
                      k_dense=35, top_n=20):
    """
    Run single-query baseline and evaluate a golden set entry.

    Fair comparison with report mode: same chunk budget (top_n=20),
    same max_tokens (8192), same report prompt, same 6-dim judge.
    Only difference: single-query retrieval vs decomposed multi-retrieval.

    Args:
        entry: Golden set entry dict.
        vector_db: ChromaDB vector store.
        generator_llm: LLM for report generation (max_tokens=8192).
        judge_llm: LLM for judging.
        reranker: CrossEncoder reranker.
        k_dense: Number of chunks for dense retrieval (default 35).
        top_n: Number of chunks after reranking (default 20, matches report mode).
    """
    from langchain_core.documents import Document

    prompt = entry["prompt"]
    entry_id = entry["id"]

    print(f"  [{entry_id}] Running baseline (single-query, k={k_dense}, top_n={top_n})...")

    # Stage 1: Dense retrieval — single query, higher k
    docs_with_scores = vector_db.similarity_search_with_score(prompt, k=k_dense)
    initial_docs = []
    for doc, score in docs_with_scores:
        doc.metadata['similarity_score'] = round(float(score), 4)
        initial_docs.append(doc)

    # Stage 2: Cross-encoder reranking
    pairs = [[prompt, doc.page_content] for doc in initial_docs]
    rerank_scores = reranker.predict(pairs)
    for doc, score in zip(initial_docs, rerank_scores):
        doc.metadata['rerank_score'] = round(float(score), 4)

    # Take top_n after reranking (same chunk budget as report mode)
    final_docs = sorted(initial_docs,
                        key=lambda x: x.metadata['rerank_score'],
                        reverse=True)[:top_n]

    # Generate report using the SAME prompt template as report mode (with token tracking)
    print(f"  [{entry_id}] Generating baseline report ({len(final_docs)} chunks)...")
    report = generate_report(
        prompt, final_docs, generator_llm,
        track_tokens=f"{entry_id}_baseline_generate",
    )

    # Judge with the SAME 6-dim judge as report mode (with token tracking)
    print(f"  [{entry_id}] Judging baseline...")
    reference_checklist = {
        "expected_sections": entry.get("expected_sections", []),
        "must_include_facts": entry.get("must_include_facts", []),
    }
    baseline_judge = judge_report(
        question=prompt,
        report=report,
        context=final_docs,
        reference_checklist=reference_checklist,
        llm=judge_llm,
        track_tokens=f"{entry_id}_baseline_judge",
    )

    # Build chunk data
    chunks = []
    for i, doc in enumerate(final_docs, 1):
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
        "report": report,
        "chunks": chunks,
        "evaluation": baseline_judge,
        "retrieval_metrics": retrieval_metrics,
        "timestamp": datetime.now().isoformat(),
    }


def evaluate_parser(parser_name, run_baseline=False, start=None, end=None):
    """
    Run full evaluation on golden set entries.

    Args:
        parser_name: Which parser's vector DB to use.
        run_baseline: If True, also run single-query baseline for comparison.
        start: 1-indexed start position (inclusive). None = from beginning.
        end: 1-indexed end position (inclusive). None = to end.

    Returns:
        List of evaluation result dicts.
    """
    print(f"\nLoading golden set from {GOLDEN_SET_PATH}...")
    golden_set = load_golden_set()
    print(f"Loaded {len(golden_set)} report prompts")

    # Slice the golden set if start/end specified
    if start is not None or end is not None:
        s = (start - 1) if start else 0
        e = end if end else len(golden_set)
        golden_set = golden_set[s:e]
        print(f"Running subset: entries {s+1} to {s+len(golden_set)} "
              f"({len(golden_set)} prompts)")

    print(f"\nLoading resources for parser: {parser_name}")
    vector_db = load_vector_db(parser_name)
    generator_llm = load_generator_llm(max_tokens=8192)
    judge_llm = load_judge_llm()
    reranker = CrossEncoder('BAAI/bge-reranker-base', device='cpu')

    tracker.reset()
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
        # Uses same LLM, same chunk budget (20), same prompt, same judge
        # Only difference: single-query retrieval (k=35) vs decomposed multi-retrieval
        if run_baseline:
            baseline_result = evaluate_baseline(
                entry, vector_db, generator_llm, judge_llm, reranker
            )
            entry_result["baseline"] = baseline_result

        # Token usage for this entry
        entry_tokens = tracker.sum_for_prefix(entry_id)
        entry_result["token_usage"] = entry_tokens
        _t = entry_tokens
        print(f"  [{entry_id}] Tokens: {_t['input_tokens']:,} in, "
              f"{_t['output_tokens']:,} out, {_t['thinking_tokens']:,} think, "
              f"{_t['total_tokens']:,} total ({_t['num_calls']} calls)")

        results.append(entry_result)

    return results


def merge_results(file_paths, output_path):
    """
    Merge multiple partial evaluation result files into one.

    Args:
        file_paths: List of JSON file paths to merge.
        output_path: Output file path for merged results.
    """
    merged = []
    seen_ids = set()

    for fp in file_paths:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:
            if entry["id"] not in seen_ids:
                merged.append(entry)
                seen_ids.add(entry["id"])
            else:
                print(f"  Warning: duplicate {entry['id']} in {fp}, skipping")

    # Sort by ID to maintain R01-R10 order
    merged.sort(key=lambda x: x["id"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(merged)} entries from {len(file_paths)} files → {output_path}")


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
    parser.add_argument(
        "--start", type=int, default=None,
        help="1-indexed start position (inclusive), e.g. --start 1 for R01",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="1-indexed end position (inclusive), e.g. --end 5 for R05",
    )
    parser.add_argument(
        "--merge", nargs="+", default=None,
        help="Merge multiple partial result files, e.g. --merge part1.json part2.json",
    )
    args = parser.parse_args()

    # Merge mode: combine partial results and exit
    if args.merge:
        output_path = Path("eval_results")
        output_path.mkdir(exist_ok=True)
        merged_path = output_path / f"{args.parser}_report_evaluation.json"
        merge_results(args.merge, merged_path)
        return

    results = evaluate_parser(
        args.parser, run_baseline=args.baseline,
        start=args.start, end=args.end,
    )

    # Save results — include range in filename for partial runs
    output_path = Path("eval_results")
    output_path.mkdir(exist_ok=True)

    if args.start is not None or args.end is not None:
        s = args.start or 1
        e = args.end or 10
        file_path = output_path / f"{args.parser}_report_evaluation_R{s:02d}-R{e:02d}.json"
    else:
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

    # Overall token usage summary
    total = tracker.get_summary()
    if total["num_calls"] > 0:
        print(f"\nToken Usage Summary ({total['num_calls']} LLM calls):")
        print(f"  Input:    {total['input_tokens']:,}")
        print(f"  Output:   {total['output_tokens']:,}")
        print(f"  Thinking: {total['thinking_tokens']:,}")
        print(f"  Total:    {total['total_tokens']:,}")


if __name__ == "__main__":
    main()

# TO RUN EVALUATION USE THE FOLLOWING SCRIPT
# python -m src.eval.evaluate --parser marker
# python -m src.eval.evaluate --parser marker --baseline
# python -m src.eval.evaluate --parser marker --baseline --start 1 --end 5
# python -m src.eval.evaluate --parser marker --baseline --start 6 --end 10
# python -m src.eval.evaluate --parser marker --merge eval_results/marker_report_evaluation_R01-R05.json eval_results/marker_report_evaluation_R06-R10.json
