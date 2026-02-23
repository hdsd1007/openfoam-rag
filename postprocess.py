"""
postprocess.py — Analyze saved query outputs and generate a structured JSON report.

The JSON output is designed to be fed directly into an LLM to generate
publication-ready tables for RAG4Report or similar workshops.

Usage:
    python postprocess.py --input run5_marker/*.json --output run5_stats.json
    python postprocess.py --input run4_marker/*.json --output run4_stats.json --label run4_marker
"""

import argparse
import json
import statistics
import re
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_query_files(filepaths):
    queries = []
    for fp in filepaths:
        with open(fp, 'r', encoding='utf-8') as f:
            queries.append(json.load(f))
    return queries


def extract_citations(answer):
    """Return list of all [n] citation numbers found in answer text."""
    return [int(c) for c in re.findall(r'\[(\d+)\]', answer)]


def is_abstained(answer):
    return answer.strip().startswith("This information is not available")


def round2(x):
    return round(x, 2)


# ── Metric Functions ───────────────────────────────────────────────────────────

def compute_retrieval_scores(queries):
    """
    Re-ranker score vs similarity score statistics.
    Answers RQ: Does re-ranking add value over raw vector similarity?
    """
    all_rerank, all_sim, all_gaps = [], [], []
    high_conf_per_query = []

    for q in queries:
        q_rerank, q_sim = [], []
        for chunk in q['chunks']:
            r = chunk.get('rerank_score')
            s = chunk.get('similarity_score')
            if r is not None:
                q_rerank.append(r * 100)
                all_rerank.append(r * 100)
            if s is not None:
                q_sim.append(s * 100)
                all_sim.append(s * 100)

        if len(q_rerank) == len(q_sim):
            all_gaps.extend([r - s for r, s in zip(q_rerank, q_sim)])

        high_conf_per_query.append(sum(1 for s in q_rerank if s > 70))

    def stats(vals):
        if not vals:
            return {}
        return {
            "mean":    round2(statistics.mean(vals)),
            "median":  round2(statistics.median(vals)),
            "std":     round2(statistics.stdev(vals) if len(vals) > 1 else 0),
            "min":     round2(min(vals)),
            "max":     round2(max(vals)),
        }

    return {
        "rerank_score":    stats(all_rerank),
        "similarity_score": stats(all_sim),
        "score_gap": {
            **stats(all_gaps),
            "positive_count": sum(1 for g in all_gaps if g > 0),
            "negative_count": sum(1 for g in all_gaps if g < 0),
            "total_chunks":   len(all_gaps),
            "positive_pct":   round2(sum(1 for g in all_gaps if g > 0) / len(all_gaps) * 100) if all_gaps else 0,
        },
        "high_confidence_gt70": {
            "avg_per_query": round2(statistics.mean(high_conf_per_query)) if high_conf_per_query else 0,
        }
    }


def compute_position_degradation(queries):
    """
    Score degradation from top to bottom of ranked list.
    Works for any k (5 or 10). Bottom always means last 2 chunks.
    Answers RQ: How sharply does re-ranking separate relevant from irrelevant?
    """
    top1, top3, bottom2 = [], [], []

    for q in queries:
        chunks = [c for c in q['chunks'] if c.get('rerank_score') is not None]
        if not chunks:
            continue
        top1.append(chunks[0]['rerank_score'] * 100)
        if len(chunks) >= 3:
            top3.append(statistics.mean(c['rerank_score'] * 100 for c in chunks[:3]))
        if len(chunks) >= 4:
            bottom2.append(statistics.mean(c['rerank_score'] * 100 for c in chunks[-2:]))

    return {
        "top1_avg":    round2(statistics.mean(top1))    if top1    else None,
        "top3_avg":    round2(statistics.mean(top3))    if top3    else None,
        "bottom2_avg": round2(statistics.mean(bottom2)) if bottom2 else None,
        "degradation_top1_to_bottom2": round2(
            statistics.mean(top1) - statistics.mean(bottom2)
        ) if top1 and bottom2 else None,
    }


def compute_citation_coverage(queries):
    """
    How well does the LLM cite what was retrieved?
    Excludes abstained answers from citation stats (they have no citations by design).
    Answers RQ: Is the generator grounded in retrieved context?
    """
    per_query = []

    for q in queries:
        abstained = is_abstained(q['answer'])
        total_chunks = len(q['chunks'])
        citations = extract_citations(q['answer']) if not abstained else []
        unique = set(citations)

        # top3_cited: how many of chunks ranked 1,2,3 were cited
        top3_cited = sum(1 for c in unique if c <= 3)

        # top5_cited: how many of chunks ranked 1-5 were cited (bounded correctly)
        top_k = min(5, total_chunks)
        top5_cited = sum(1 for c in unique if 1 <= c <= top_k)

        per_query.append({
            "question":        q['question'][:80],
            "abstained":       abstained,
            "total_chunks":    total_chunks,
            "total_citations": len(citations),
            "unique_citations": len(unique),
            "top3_cited":      top3_cited,
            "top5_cited":      top5_cited,
            "coverage_rate":   round2(len(unique) / total_chunks) if total_chunks > 0 else 0,
        })

    # Aggregate only non-abstained
    active = [p for p in per_query if not p['abstained']]

    return {
        "total_queries":    len(per_query),
        "abstained_count":  sum(1 for p in per_query if p['abstained']),
        "abstained_pct":    round2(sum(1 for p in per_query if p['abstained']) / len(per_query) * 100) if per_query else 0,
        "avg_coverage_rate":     round2(statistics.mean(p['coverage_rate'] for p in active)) if active else 0,
        "avg_top3_cited":        round2(statistics.mean(p['top3_cited'] for p in active)) if active else 0,
        "avg_top5_cited":        round2(statistics.mean(p['top5_cited'] for p in active)) if active else 0,
        "avg_unique_citations":  round2(statistics.mean(p['unique_citations'] for p in active)) if active else 0,
        "per_query": per_query,
    }


def compute_source_diversity(queries):
    """
    Are retrieved chunks coming from varied sources/pages/sections?
    Answers RQ: Is the retriever fetching diverse evidence or repeating the same passage?
    """
    per_query = []

    for q in queries:
        sources   = set(c['source'] for c in q['chunks'])
        pages     = set(c['page'] for c in q['chunks'])
        sections  = set(c['section'] for c in q['chunks'] if c.get('section') not in (None, 'N/A'))

        per_query.append({
            "question":        q['question'][:80],
            "unique_sources":  len(sources),
            "unique_pages":    len(pages),
            "unique_sections": len(sections),
            "total_chunks":    len(q['chunks']),
            "source_names":    sorted(sources),
        })

    return {
        "avg_unique_sources":  round2(statistics.mean(p['unique_sources']  for p in per_query)),
        "avg_unique_pages":    round2(statistics.mean(p['unique_pages']    for p in per_query)),
        "avg_unique_sections": round2(statistics.mean(p['unique_sections'] for p in per_query)),
        "per_query": per_query,
    }


# ── Report Evaluation Metrics ─────────────────────────────────────────────────

def _safe_stats(vals):
    """Compute mean/median/std/min/max from a list, handling edge cases."""
    if not vals:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean":   round2(statistics.mean(vals)),
        "median": round2(statistics.median(vals)),
        "std":    round2(statistics.stdev(vals) if len(vals) > 1 else 0),
        "min":    round2(min(vals)),
        "max":    round2(max(vals)),
    }


def compute_report_metrics(eval_results):
    """
    Aggregate 6 judge dimensions across all report prompts.

    Args:
        eval_results: List of evaluation result dicts (from evaluate.py output).

    Returns:
        Dict with per-dimension stats (mean, median, std, min, max)
        and overall_score stats.
    """
    dimensions = [
        "groundedness", "technical_accuracy", "citation_correctness",
        "coverage", "factual_recall", "structure",
    ]
    dim_values = {d: [] for d in dimensions}
    overall_scores = []

    # Track section/fact coverage
    total_expected_sections = 0
    total_found_sections = 0
    total_expected_facts = 0
    total_found_facts = 0

    for entry in eval_results:
        report_data = entry.get("report", {})
        eval_data = report_data.get("evaluation", {})

        if "error" in eval_data:
            continue

        for dim in dimensions:
            val = eval_data.get(dim)
            if val is not None:
                dim_values[dim].append(val)

        overall = eval_data.get("overall_score")
        if overall is not None:
            overall_scores.append(overall)

        # Section/fact tracking
        sections_found = eval_data.get("sections_found", [])
        sections_missing = eval_data.get("sections_missing", [])
        facts_found = eval_data.get("facts_found", [])
        facts_missing = eval_data.get("facts_missing", [])

        total_found_sections += len(sections_found)
        total_expected_sections += len(sections_found) + len(sections_missing)
        total_found_facts += len(facts_found)
        total_expected_facts += len(facts_found) + len(facts_missing)

    result = {
        "total_prompts": len(eval_results),
        "successful_judges": len(overall_scores),
    }

    for dim in dimensions:
        result[dim] = _safe_stats(dim_values[dim])

    result["overall_score"] = _safe_stats(overall_scores)

    result["section_coverage"] = {
        "total_expected": total_expected_sections,
        "total_found": total_found_sections,
        "rate": round2(total_found_sections / total_expected_sections) if total_expected_sections > 0 else 0,
    }

    result["fact_coverage"] = {
        "total_expected": total_expected_facts,
        "total_found": total_found_facts,
        "rate": round2(total_found_facts / total_expected_facts) if total_expected_facts > 0 else 0,
    }

    return result


def compute_retrieval_ir_metrics(eval_results):
    """
    Aggregate IR retrieval metrics (Recall@k, MRR, NDCG) across all prompts.

    Args:
        eval_results: List of evaluation result dicts (from evaluate.py output).

    Returns:
        Dict with per-metric stats (mean, median, std, min, max).
    """
    metric_names = [
        "recall@5", "recall@10", "recall@15", "recall@20",
        "MRR", "NDCG@10", "NDCG@20",
    ]
    metric_values = {m: [] for m in metric_names}

    for entry in eval_results:
        report_data = entry.get("report", {})
        ir_metrics = report_data.get("retrieval_metrics", {})

        if not ir_metrics:
            continue

        for m in metric_names:
            val = ir_metrics.get(m)
            if val is not None:
                metric_values[m].append(val)

    result = {
        "prompts_with_relevant_pages": len(metric_values["recall@5"]),
    }

    for m in metric_names:
        result[m] = _safe_stats(metric_values[m])

    return result


# ── Main Report Builder ────────────────────────────────────────────────────────

def build_report(queries, label):
    return {
        "meta": {
            "label":        label,
            "total_queries": len(queries),
            "has_reranker":  queries[0].get('has_reranker', False) if queries else None,
            "parser":        queries[0].get('parser', 'unknown') if queries else None,
            "k_retrieved":   len(queries[0]['chunks']) if queries else None,
        },
        "retrieval_scores":     compute_retrieval_scores(queries),
        "position_degradation": compute_position_degradation(queries),
        "citation_coverage":    compute_citation_coverage(queries),
        "source_diversity":     compute_source_diversity(queries),
    }


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Postprocess query JSONs into structured stats JSON.')
    parser.add_argument('--input',  nargs='+', required=False, help='Query JSON files')
    parser.add_argument('--output', required=True,             help='Output JSON report path')
    parser.add_argument('--label',  default=None,              help='Run label (e.g. run5_marker). Auto-detected from folder if omitted.')
    parser.add_argument('--eval-results', default=None,        help='Evaluation JSON from evaluate.py (for report/IR metrics)')
    args = parser.parse_args()

    if not args.input and not args.eval_results:
        print("Error: Must provide either --input or --eval-results")
        return

    combined_report = {}

    # Standard query postprocessing
    if args.input:
        label = args.label or Path(args.input[0]).parent.name

        print(f"Loading {len(args.input)} query files...")
        queries = load_query_files(args.input)
        print(f"Building report for: {label} ({len(queries)} queries)")

        combined_report = build_report(queries, label)

        # Print a quick summary to stdout
        m  = combined_report['meta']
        rs = combined_report['retrieval_scores']
        pd = combined_report['position_degradation']
        cc = combined_report['citation_coverage']
        sd = combined_report['source_diversity']

        print(f"""
┌─ SUMMARY: {label} ({'reranker' if m['has_reranker'] else 'no reranker'}, k={m['k_retrieved']}) ──────────────
│ Queries: {m['total_queries']}   Abstained: {cc['abstained_count']} ({cc['abstained_pct']}%)
│
│ Retrieval Scores
│   Rerank  mean/median: {rs['rerank_score'].get('mean','N/A')} / {rs['rerank_score'].get('median','N/A')}%
│   Sim     mean/median: {rs['similarity_score'].get('mean','N/A')} / {rs['similarity_score'].get('median','N/A')}%
│   Gap     mean: {rs['score_gap'].get('mean','N/A')}%   positive: {rs['score_gap'].get('positive_pct','N/A')}% of chunks
│
│ Position Degradation
│   Top-1: {pd['top1_avg']}%  Top-3: {pd['top3_avg']}%  Bottom-2: {pd['bottom2_avg']}%
│   Drop top1→bottom2: {pd['degradation_top1_to_bottom2']}%
│
│ Citation Coverage (non-abstained)
│   Avg coverage rate: {cc['avg_coverage_rate']}   Avg top-3 cited: {cc['avg_top3_cited']}/3
│
│ Source Diversity
│   Avg unique sources: {sd['avg_unique_sources']}   pages: {sd['avg_unique_pages']}   sections: {sd['avg_unique_sections']}
└──────────────────────────────────────────────────────────────────────────────
""")

    # Evaluation results postprocessing (report pipeline metrics)
    if args.eval_results:
        print(f"\nLoading evaluation results from {args.eval_results}...")
        with open(args.eval_results, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)

        report_metrics = compute_report_metrics(eval_results)
        ir_metrics = compute_retrieval_ir_metrics(eval_results)

        combined_report["report_evaluation"] = report_metrics
        combined_report["retrieval_ir_metrics"] = ir_metrics

        # Print evaluation summary
        rm = report_metrics
        print(f"""
┌─ REPORT EVALUATION SUMMARY ──────────────────────────────────────────────────
│ Prompts: {rm['total_prompts']}   Successful judges: {rm['successful_judges']}
│
│ Overall Score:   mean={rm['overall_score'].get('mean','N/A')}  median={rm['overall_score'].get('median','N/A')}
│ Groundedness:    mean={rm['groundedness'].get('mean','N/A')}
│ Tech Accuracy:   mean={rm['technical_accuracy'].get('mean','N/A')}
│ Citation:        mean={rm['citation_correctness'].get('mean','N/A')}
│ Coverage:        mean={rm['coverage'].get('mean','N/A')}
│ Factual Recall:  mean={rm['factual_recall'].get('mean','N/A')}
│ Structure:       mean={rm['structure'].get('mean','N/A')}
│
│ Section Coverage: {rm['section_coverage']['total_found']}/{rm['section_coverage']['total_expected']} ({rm['section_coverage']['rate']})
│ Fact Coverage:    {rm['fact_coverage']['total_found']}/{rm['fact_coverage']['total_expected']} ({rm['fact_coverage']['rate']})
└──────────────────────────────────────────────────────────────────────────────
""")

        if ir_metrics.get("prompts_with_relevant_pages", 0) > 0:
            print(f"""
┌─ IR RETRIEVAL METRICS ───────────────────────────────────────────────────────
│ Prompts with relevant_pages: {ir_metrics['prompts_with_relevant_pages']}
│ Recall@5:  mean={ir_metrics['recall@5'].get('mean','N/A')}
│ Recall@10: mean={ir_metrics['recall@10'].get('mean','N/A')}
│ Recall@20: mean={ir_metrics['recall@20'].get('mean','N/A')}
│ MRR:       mean={ir_metrics['MRR'].get('mean','N/A')}
│ NDCG@10:   mean={ir_metrics['NDCG@10'].get('mean','N/A')}
│ NDCG@20:   mean={ir_metrics['NDCG@20'].get('mean','N/A')}
└──────────────────────────────────────────────────────────────────────────────
""")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(combined_report, f, indent=2, ensure_ascii=False)

    print(f"Report saved to: {args.output}")


if __name__ == '__main__':
    main()


# ── Usage ──────────────────────────────────────────────────────────────────────
# Run 4 (k=5 reranker):
#   python postprocess.py --input run4_marker/*.json --output run4_stats.json
#
# Run 5 (k=10 reranker):
#   python postprocess.py --input run5_marker/*.json --output run5_stats.json --label run5_marker
#
# Evaluation results (report pipeline):
#   python postprocess.py --eval-results eval_results/marker_report_evaluation.json --output report_stats.json
#
# Combined (query stats + evaluation):
#   python postprocess.py --input run4_marker/*.json --eval-results eval_results/marker_report_evaluation.json --output combined_stats.json
#
# Then feed both JSONs to an LLM to generate comparison tables:
#   "Here are two RAG pipeline evaluation reports. Generate a LaTeX comparison table."