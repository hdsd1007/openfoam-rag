"""
Analyze saved query outputs and generate statistics.

Usage:
    python analyze_queries.py --input query_outputs/*.json
    python analyze_queries.py --input queries/with/*.json --output stats_report.txt
"""

import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict, Counter
import re


def load_query_files(filepaths):
    """Load all query JSON files."""
    queries = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            queries.append(json.load(f))
    return queries


def extract_citations(answer):
    """Extract citation numbers from answer text."""
    # Find all [n] patterns
    citations = re.findall(r'\[(\d+)\]', answer)
    return [int(c) for c in citations]


def calculate_rerank_stats(queries):
    """Calculate re-rank score statistics."""
    all_rerank_scores = []
    all_sim_scores = []
    score_gaps = []
    high_confidence_counts = []
    
    for query in queries:
        rerank_scores = []
        sim_scores = []
        
        for chunk in query['chunks']:
            if chunk.get('rerank_score') is not None:
                rerank_scores.append(chunk['rerank_score'] * 100)
                all_rerank_scores.append(chunk['rerank_score'] * 100)
            
            if chunk.get('similarity_score') is not None:
                sim_scores.append(chunk['similarity_score'] * 100)
                all_sim_scores.append(chunk['similarity_score'] * 100)
        
        # Calculate gaps for this query
        if len(rerank_scores) == len(sim_scores):
            gaps = [r - s for r, s in zip(rerank_scores, sim_scores)]
            score_gaps.extend(gaps)
        
        # High confidence chunks (>70%)
        high_conf = sum(1 for s in rerank_scores if s > 70)
        high_confidence_counts.append(high_conf)
    
    stats = {}
    
    if all_rerank_scores:
        stats['rerank'] = {
            'mean': statistics.mean(all_rerank_scores),
            'median': statistics.median(all_rerank_scores),
            'std_dev': statistics.stdev(all_rerank_scores) if len(all_rerank_scores) > 1 else 0,
            'min': min(all_rerank_scores),
            'max': max(all_rerank_scores)
        }
    
    if all_sim_scores:
        stats['similarity'] = {
            'mean': statistics.mean(all_sim_scores),
            'median': statistics.median(all_sim_scores),
            'std_dev': statistics.stdev(all_sim_scores) if len(all_sim_scores) > 1 else 0,
            'min': min(all_sim_scores),
            'max': max(all_sim_scores)
        }
    
    if score_gaps:
        stats['score_gap'] = {
            'mean': statistics.mean(score_gaps),
            'median': statistics.median(score_gaps),
            'positive_gaps': sum(1 for g in score_gaps if g > 0),
            'negative_gaps': sum(1 for g in score_gaps if g < 0),
            'total': len(score_gaps)
        }
    
    if high_confidence_counts:
        stats['high_confidence'] = {
            'avg_per_query': statistics.mean(high_confidence_counts),
            'total_queries': len(high_confidence_counts)
        }
    
    return stats


def calculate_citation_coverage(queries):
    """Calculate citation usage statistics."""
    citation_stats = []
    
    for query in queries:
        citations = extract_citations(query['answer'])
        unique_citations = set(citations)
        total_chunks = len(query['chunks'])
        
        # How many of top-k chunks were cited
        top3_cited = sum(1 for c in unique_citations if c <= 3)
        top5_cited = len(unique_citations)
        
        citation_stats.append({
            'total_citations': len(citations),
            'unique_citations': len(unique_citations),
            'total_chunks': total_chunks,
            'top3_cited': top3_cited,
            'top5_cited': top5_cited,
            'coverage_rate': len(unique_citations) / total_chunks if total_chunks > 0 else 0
        })
    
    avg_coverage = statistics.mean([s['coverage_rate'] for s in citation_stats])
    avg_top3_cited = statistics.mean([s['top3_cited'] for s in citation_stats])
    avg_unique_citations = statistics.mean([s['unique_citations'] for s in citation_stats])
    
    return {
        'avg_coverage_rate': avg_coverage,
        'avg_top3_cited': avg_top3_cited,
        'avg_unique_citations': avg_unique_citations,
        'per_query': citation_stats
    }


def calculate_diversity_metrics(queries):
    """Calculate chunk diversity metrics."""
    diversity = []
    
    for query in queries:
        sources = set(chunk['source'] for chunk in query['chunks'])
        pages = set(chunk['page'] for chunk in query['chunks'])
        sections = set(chunk['section'] for chunk in query['chunks'] if chunk['section'] != 'N/A')
        
        diversity.append({
            'unique_sources': len(sources),
            'unique_pages': len(pages),
            'unique_sections': len(sections),
            'total_chunks': len(query['chunks'])
        })
    
    return {
        'avg_unique_sources': statistics.mean([d['unique_sources'] for d in diversity]),
        'avg_unique_pages': statistics.mean([d['unique_pages'] for d in diversity]),
        'avg_unique_sections': statistics.mean([d['unique_sections'] for d in diversity]),
        'per_query': diversity
    }


def calculate_position_metrics(queries):
    """Calculate position-based metrics for re-ranked chunks."""
    position_stats = {
        'top1_avg_rerank': [],
        'top3_avg_rerank': [],
        'bottom2_avg_rerank': []
    }
    
    for query in queries:
        chunks_with_rerank = [c for c in query['chunks'] if c.get('rerank_score') is not None]
        
        if chunks_with_rerank:
            # Top 1
            if len(chunks_with_rerank) >= 1:
                position_stats['top1_avg_rerank'].append(chunks_with_rerank[0]['rerank_score'] * 100)
            
            # Top 3
            if len(chunks_with_rerank) >= 3:
                top3_avg = statistics.mean([c['rerank_score'] * 100 for c in chunks_with_rerank[:3]])
                position_stats['top3_avg_rerank'].append(top3_avg)
            
            # Bottom 2
            if len(chunks_with_rerank) >= 5:
                bottom2_avg = statistics.mean([c['rerank_score'] * 100 for c in chunks_with_rerank[-2:]])
                position_stats['bottom2_avg_rerank'].append(bottom2_avg)
    
    results = {}
    if position_stats['top1_avg_rerank']:
        results['top1_avg'] = statistics.mean(position_stats['top1_avg_rerank'])
    if position_stats['top3_avg_rerank']:
        results['top3_avg'] = statistics.mean(position_stats['top3_avg_rerank'])
    if position_stats['bottom2_avg_rerank']:
        results['bottom2_avg'] = statistics.mean(position_stats['bottom2_avg_rerank'])
    
    return results


def generate_report(queries, output_file=None):
    """Generate comprehensive statistics report."""
    
    report = []
    report.append("="*80)
    report.append("QUERY ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"\nTotal Queries Analyzed: {len(queries)}")
    report.append(f"Has Re-Ranker: {queries[0].get('has_reranker', False) if queries else 'Unknown'}")
    report.append("")
    
    # Re-Rank Statistics
    report.append("-" * 80)
    report.append("1. RE-RANK SCORE STATISTICS")
    report.append("-" * 80)
    
    rerank_stats = calculate_rerank_stats(queries)
    
    if 'rerank' in rerank_stats:
        r = rerank_stats['rerank']
        report.append(f"Mean Re-Rank Score:      {r['mean']:.2f}%")
        report.append(f"Median Re-Rank Score:    {r['median']:.2f}%")
        report.append(f"Std Deviation:           {r['std_dev']:.2f}%")
        report.append(f"Range:                   {r['min']:.2f}% - {r['max']:.2f}%")
    
    if 'similarity' in rerank_stats:
        s = rerank_stats['similarity']
        report.append(f"\nMean Similarity Score:   {s['mean']:.2f}%")
        report.append(f"Median Similarity Score: {s['median']:.2f}%")
        report.append(f"Std Deviation:           {s['std_dev']:.2f}%")
        report.append(f"Range:                   {s['min']:.2f}% - {s['max']:.2f}%")
    
    if 'score_gap' in rerank_stats:
        g = rerank_stats['score_gap']
        report.append(f"\nScore Gap (Re-Rank - Similarity):")
        report.append(f"  Mean Gap:              {g['mean']:.2f}%")
        report.append(f"  Median Gap:            {g['median']:.2f}%")
        report.append(f"  Positive Gaps:         {g['positive_gaps']}/{g['total']} ({g['positive_gaps']/g['total']*100:.1f}%)")
        report.append(f"  Negative Gaps:         {g['negative_gaps']}/{g['total']} ({g['negative_gaps']/g['total']*100:.1f}%)")
    
    if 'high_confidence' in rerank_stats:
        h = rerank_stats['high_confidence']
        report.append(f"\nHigh Confidence Chunks (>70%):")
        report.append(f"  Avg per Query:         {h['avg_per_query']:.2f} chunks")
    
    # Position Metrics
    report.append("\n" + "-" * 80)
    report.append("2. POSITION-BASED METRICS")
    report.append("-" * 80)
    
    position_metrics = calculate_position_metrics(queries)
    if 'top1_avg' in position_metrics:
        report.append(f"Top-1 Chunk Avg Re-Rank:  {position_metrics['top1_avg']:.2f}%")
    if 'top3_avg' in position_metrics:
        report.append(f"Top-3 Chunks Avg Re-Rank: {position_metrics['top3_avg']:.2f}%")
    if 'bottom2_avg' in position_metrics:
        report.append(f"Bottom-2 Chunks Avg Re-Rank: {position_metrics['bottom2_avg']:.2f}%")
    
    # Citation Coverage
    report.append("\n" + "-" * 80)
    report.append("3. CITATION COVERAGE")
    report.append("-" * 80)
    
    citation_stats = calculate_citation_coverage(queries)
    report.append(f"Avg Coverage Rate:        {citation_stats['avg_coverage_rate']:.2%}")
    report.append(f"Avg Top-3 Chunks Cited:   {citation_stats['avg_top3_cited']:.2f} / 3")
    report.append(f"Avg Unique Citations:     {citation_stats['avg_unique_citations']:.2f}")
    
    # Diversity Metrics
    report.append("\n" + "-" * 80)
    report.append("4. CHUNK DIVERSITY")
    report.append("-" * 80)
    
    diversity = calculate_diversity_metrics(queries)
    report.append(f"Avg Unique Sources:       {diversity['avg_unique_sources']:.2f}")
    report.append(f"Avg Unique Pages:         {diversity['avg_unique_pages']:.2f}")
    report.append(f"Avg Unique Sections:      {diversity['avg_unique_sections']:.2f}")
    
    # Per-Query Breakdown
    report.append("\n" + "-" * 80)
    report.append("5. PER-QUERY BREAKDOWN")
    report.append("-" * 80)
    
    for i, query in enumerate(queries, 1):
        report.append(f"\nQuery {i}: {query['question'][:60]}...")
        
        # Re-rank scores for this query
        rerank_scores = [c.get('rerank_score', 0) * 100 for c in query['chunks'] if c.get('rerank_score') is not None]
        if rerank_scores:
            report.append(f"  Re-Rank Scores: {', '.join(f'{s:.0f}%' for s in rerank_scores)}")
        
        # Citations
        citations = extract_citations(query['answer'])
        unique_cites = len(set(citations))
        report.append(f"  Citations: {len(citations)} total, {unique_cites} unique")
        
        # Sources
        sources = set(c['source'] for c in query['chunks'])
        report.append(f"  Sources: {', '.join(sources)}")
    
    report.append("\n" + "=" * 80)
    
    # Output
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✅ Report saved to: {output_file}")
    else:
        print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Analyze saved query outputs')
    parser.add_argument('--input', nargs='+', required=True, help='Query JSON files to analyze')
    parser.add_argument('--output', help='Output file for report (default: print to console)')
    
    args = parser.parse_args()
    
    print(f"Loading {len(args.input)} query files...")
    queries = load_query_files(args.input)
    
    print(f"Analyzing {len(queries)} queries...\n")
    generate_report(queries, args.output)


if __name__ == '__main__':
    main()


# USAGE EXAMPLES:

# Analyze all queries in a directory:
# python analyze_queries.py --input query_outputs/*.json

# Analyze and save report:
# python analyze_queries.py --input query_outputs/*.json --output analysis_report.txt

# Analyze specific queries:
# python analyze_queries.py --input query1.json query2.json query3.json