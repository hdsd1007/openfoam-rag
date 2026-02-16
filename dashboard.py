"""
Generate visual statistics dashboard from saved queries.

Usage:
    python generate_stats_dashboard.py --input query_outputs/*.json --output stats.html
"""

import argparse
import json
import statistics
from pathlib import Path
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
    citations = re.findall(r'\[(\d+)\]', answer)
    return [int(c) for c in citations]


def calculate_stats(queries):
    """Calculate all statistics."""
    
    # Collect data
    all_rerank = []
    all_sim = []
    score_gaps = []
    citation_coverage = []
    top1_scores = []
    
    for query in queries:
        rerank_scores = []
        sim_scores = []
        
        for chunk in query['chunks']:
            if chunk.get('rerank_score') is not None:
                rerank_scores.append(chunk['rerank_score'] * 100)
                all_rerank.append(chunk['rerank_score'] * 100)
            
            if chunk.get('similarity_score') is not None:
                sim_scores.append(chunk['similarity_score'] * 100)
                all_sim.append(chunk['similarity_score'] * 100)
        
        if len(rerank_scores) == len(sim_scores):
            gaps = [r - s for r, s in zip(rerank_scores, sim_scores)]
            score_gaps.extend(gaps)
        
        # Top-1 re-rank score
        if rerank_scores:
            top1_scores.append(rerank_scores[0])
        
        # Citation coverage
        citations = set(extract_citations(query['answer']))
        coverage = len(citations) / len(query['chunks']) if query['chunks'] else 0
        citation_coverage.append(coverage * 100)
    
    return {
        'rerank_scores': all_rerank,
        'sim_scores': all_sim,
        'score_gaps': score_gaps,
        'citation_coverage': citation_coverage,
        'top1_scores': top1_scores,
        'num_queries': len(queries)
    }


DASHBOARD_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Query Statistics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; padding: 40px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { font-size: 32px; color: #1e293b; margin-bottom: 8px; }
        .subtitle { color: #64748b; margin-bottom: 40px; font-size: 14px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; margin-bottom: 40px; }
        .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .card h3 { font-size: 14px; text-transform: uppercase; color: #64748b; margin-bottom: 16px; font-weight: 600; letter-spacing: 0.5px; }
        .metric { font-size: 48px; font-weight: 800; color: #1e293b; margin-bottom: 8px; }
        .metric-label { font-size: 14px; color: #64748b; }
        .chart-container { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }
        .chart-container h3 { font-size: 16px; color: #1e293b; margin-bottom: 20px; font-weight: 600; }
        canvas { max-height: 300px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Query Statistics Dashboard</h1>
        <p class="subtitle">Analysis of STATS_NUM_QUERIES queries with re-ranker enabled</p>
        
        <div class="grid">
            <div class="card">
                <h3>Avg Re-Rank Score</h3>
                <div class="metric">STATS_AVG_RERANK%</div>
                <div class="metric-label">Across all chunks</div>
            </div>
            
            <div class="card">
                <h3>Avg Score Gap</h3>
                <div class="metric">+STATS_AVG_GAP%</div>
                <div class="metric-label">Re-Rank over Similarity</div>
            </div>
            
            <div class="card">
                <h3>Citation Coverage</h3>
                <div class="metric">STATS_AVG_COVERAGE%</div>
                <div class="metric-label">Chunks cited in answers</div>
            </div>
            
            <div class="card">
                <h3>Top-1 Avg Score</h3>
                <div class="metric">STATS_TOP1_AVG%</div>
                <div class="metric-label">First chunk confidence</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Re-Rank vs Similarity Score Distribution</h3>
            <canvas id="scoreDistChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Score Gap Distribution (Re-Rank - Similarity)</h3>
            <canvas id="gapChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h3>Citation Coverage per Query</h3>
            <canvas id="citationChart"></canvas>
        </div>
    </div>

    <script>
        const stats = STATS_DATA_PLACEHOLDER;

        // Score Distribution Chart
        const ctx1 = document.getElementById('scoreDistChart');
        new Chart(ctx1, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Similarity vs Re-Rank',
                    data: stats.score_pairs,
                    backgroundColor: 'rgba(59, 130, 246, 0.6)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Similarity Score (%)' }, min: 0, max: 100 },
                    y: { title: { display: true, text: 'Re-Rank Score (%)' }, min: 0, max: 100 }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Sim: ${ctx.parsed.x.toFixed(1)}%, Rerank: ${ctx.parsed.y.toFixed(1)}%`
                        }
                    }
                }
            }
        });

        // Gap Distribution Chart
        const ctx2 = document.getElementById('gapChart');
        new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: stats.gap_bins.labels,
                datasets: [{
                    label: 'Number of Chunks',
                    data: stats.gap_bins.values,
                    backgroundColor: 'rgba(16, 185, 129, 0.6)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { title: { display: true, text: 'Count' } }
                }
            }
        });

        // Citation Coverage Chart
        const ctx3 = document.getElementById('citationChart');
        new Chart(ctx3, {
            type: 'bar',
            data: {
                labels: stats.query_labels,
                datasets: [{
                    label: 'Coverage (%)',
                    data: stats.citation_coverage,
                    backgroundColor: 'rgba(139, 92, 246, 0.6)'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { title: { display: true, text: 'Coverage (%)' }, min: 0, max: 100 }
                }
            }
        });
    </script>
</body>
</html>'''


def generate_dashboard(queries, output_file):
    """Generate visual statistics dashboard."""
    
    stats = calculate_stats(queries)
    
    # Calculate summary metrics
    avg_rerank = int(statistics.mean(stats['rerank_scores'])) if stats['rerank_scores'] else 0
    avg_sim = int(statistics.mean(stats['sim_scores'])) if stats['sim_scores'] else 0
    avg_gap = int(statistics.mean(stats['score_gaps'])) if stats['score_gaps'] else 0
    avg_coverage = int(statistics.mean(stats['citation_coverage'])) if stats['citation_coverage'] else 0
    top1_avg = int(statistics.mean(stats['top1_scores'])) if stats['top1_scores'] else 0
    
    # Prepare data for charts
    
    # Score pairs for scatter plot
    score_pairs = []
    for i in range(min(len(stats['rerank_scores']), len(stats['sim_scores']))):
        score_pairs.append({
            'x': stats['sim_scores'][i],
            'y': stats['rerank_scores'][i]
        })
    
    # Gap distribution bins
    gap_bins = {
        '<-20%': 0,
        '-20 to -10%': 0,
        '-10 to 0%': 0,
        '0 to +10%': 0,
        '+10 to +20%': 0,
        '>+20%': 0
    }
    
    for gap in stats['score_gaps']:
        if gap < -20:
            gap_bins['<-20%'] += 1
        elif gap < -10:
            gap_bins['-20 to -10%'] += 1
        elif gap < 0:
            gap_bins['-10 to 0%'] += 1
        elif gap < 10:
            gap_bins['0 to +10%'] += 1
        elif gap < 20:
            gap_bins['+10 to +20%'] += 1
        else:
            gap_bins['>+20%'] += 1
    
    # Query labels
    query_labels = [f"Q{i+1}" for i in range(len(stats['citation_coverage']))]
    
    chart_data = {
        'score_pairs': score_pairs,
        'gap_bins': {
            'labels': list(gap_bins.keys()),
            'values': list(gap_bins.values())
        },
        'citation_coverage': stats['citation_coverage'],
        'query_labels': query_labels
    }
    
    # Generate HTML
    html = DASHBOARD_TEMPLATE
    html = html.replace('STATS_NUM_QUERIES', str(stats['num_queries']))
    html = html.replace('STATS_AVG_RERANK', str(avg_rerank))
    html = html.replace('STATS_AVG_GAP', str(avg_gap))
    html = html.replace('STATS_AVG_COVERAGE', str(avg_coverage))
    html = html.replace('STATS_TOP1_AVG', str(top1_avg))
    html = html.replace('STATS_DATA_PLACEHOLDER', json.dumps(chart_data, indent=2))
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Stats dashboard generated: {output_file}")
    print(f"   Queries analyzed: {stats['num_queries']}")
    print(f"   Avg Re-Rank Score: {avg_rerank}%")
    print(f"   Avg Score Gap: +{avg_gap}%")
    print(f"   Avg Citation Coverage: {avg_coverage}%")


def main():
    parser = argparse.ArgumentParser(description='Generate visual statistics dashboard')
    parser.add_argument('--input', nargs='+', required=True, help='Query JSON files')
    parser.add_argument('--output', default='stats_dashboard.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    print(f"Loading {len(args.input)} query files...")
    queries = load_query_files(args.input)
    
    generate_dashboard(queries, args.output)


if __name__ == '__main__':
    main()


# USAGE:
# python generate_stats_dashboard.py --input query_outputs/*.json
# python generate_stats_dashboard.py --input query_outputs/*.json --output my_stats.html