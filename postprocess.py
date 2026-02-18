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
    parser.add_argument('--input',  nargs='+', required=True, help='Query JSON files')
    parser.add_argument('--output', required=True,            help='Output JSON report path')
    parser.add_argument('--label',  default=None,             help='Run label (e.g. run5_marker). Auto-detected from folder if omitted.')
    args = parser.parse_args()

    # Auto-detect label from parent folder of first input file if not given
    label = args.label or Path(args.input[0]).parent.name

    print(f"Loading {len(args.input)} query files...")
    queries = load_query_files(args.input)
    print(f"Building report for: {label} ({len(queries)} queries)")

    report = build_report(queries, label)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Report saved to: {args.output}")

    # Print a quick summary to stdout
    m  = report['meta']
    rs = report['retrieval_scores']
    pd = report['position_degradation']
    cc = report['citation_coverage']
    sd = report['source_diversity']

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


if __name__ == '__main__':
    main()


# ── Usage ──────────────────────────────────────────────────────────────────────
# Run 4 (k=5 reranker):
#   python postprocess.py --input run4_marker/*.json --output run4_stats.json
#
# Run 5 (k=10 reranker):
#   python postprocess.py --input run5_marker/*.json --output run5_stats.json --label run5_marker
#
# Then feed both JSONs to an LLM to generate comparison tables:
#   "Here are two RAG pipeline evaluation reports. Generate a LaTeX comparison table."