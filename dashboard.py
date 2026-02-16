"""
Generate dashboard from multiple saved query JSON files.

Usage:
    python generate_multi_dashboard.py --input run4/*.json --output dashboard.html
"""

import argparse
import json
from pathlib import Path


DASHBOARD_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Query Results Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; display: flex; height: 100vh; }
        
        .sidebar { width: 320px; background: #1e293b; color: white; padding: 24px; overflow-y: auto; }
        .sidebar h1 { font-size: 20px; margin-bottom: 24px; }
        .question { padding: 12px; background: #334155; border-radius: 6px; margin-bottom: 12px; cursor: pointer; font-size: 14px; transition: background 0.2s; }
        .question:hover { background: #475569; }
        .question.active { background: #3b82f6; }
        
        .main { flex: 1; display: flex; flex-direction: column; background: #f8fafc; }
        
        .header { background: white; border-bottom: 1px solid #e2e8f0; padding: 16px 32px; }
        .header h2 { font-size: 18px; color: #1e293b; }
        
        .content { flex: 1; overflow-y: auto; padding: 32px; }
        
        .answer-box { background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .answer-box h3 { font-size: 14px; text-transform: uppercase; color: #64748b; margin-bottom: 16px; font-weight: 600; letter-spacing: 0.5px; }
        .answer-text { line-height: 1.8; color: #334155; font-size: 15px; white-space: pre-wrap; }
        
        .chunks-box { background: white; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .chunks-box h3 { font-size: 14px; text-transform: uppercase; color: #64748b; margin-bottom: 16px; font-weight: 600; letter-spacing: 0.5px; }
        
        .chunk { background: #f8fafc; border-left: 3px solid #cbd5e1; padding: 16px; margin-bottom: 12px; border-radius: 4px; }
        .chunk-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
        .chunk-left { flex: 1; }
        .chunk-rank { background: #1e293b; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; display: inline-block; margin-bottom: 8px; }
        .chunk-meta { font-size: 13px; color: #475569; line-height: 1.6; }
        .chunk-meta-label { font-weight: 600; color: #64748b; }
        .chunk-source { font-size: 12px; color: #94a3b8; margin-top: 4px; }
        
        .scores { display: flex; gap: 8px; margin-top: 12px; }
        .score-badge { padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        .score-sim { background: #dbeafe; color: #1e40af; }
        .score-rerank { background: #d1fae5; color: #065f46; }
        .score-rerank.high { background: #10b981; color: white; }
    </style>
</head>
<body>
    <div class="sidebar">
        <h1>Questions</h1>
        <div id="questions"></div>
    </div>
    
    <div class="main">
        <div class="header">
            <h2 id="current-question"></h2>
        </div>
        
        <div class="content">
            <div class="answer-box">
                <h3>Answer</h3>
                <div class="answer-text" id="answer"></div>
            </div>
            
            <div class="chunks-box">
                <h3>Retrieved Chunks</h3>
                <div id="chunks"></div>
            </div>
        </div>
    </div>

    <script>
        const data = DATA_PLACEHOLDER;
        let currentQ = 0;

        function init() {
            const qDiv = document.getElementById('questions');
            data.forEach((q, i) => {
                const div = document.createElement('div');
                div.className = `question ${i===0?'active':''}`;
                div.textContent = `Q${i+1}: ${q.question.substring(0, 50)}...`;
                div.onclick = () => loadQuestion(i);
                qDiv.appendChild(div);
            });
            loadQuestion(0);
        }

        function loadQuestion(idx) {
            currentQ = idx;
            document.querySelectorAll('.question').forEach((el, i) => {
                el.className = i === idx ? 'question active' : 'question';
            });
            
            const q = data[idx];
            document.getElementById('current-question').textContent = q.question;
            document.getElementById('answer').textContent = q.answer;
            
            const chunksDiv = document.getElementById('chunks');
            chunksDiv.innerHTML = '';
            
            q.chunks.forEach(c => {
                const div = document.createElement('div');
                div.className = 'chunk';
                
                let metaHTML = `<span class="chunk-meta-label">Section:</span> ${c.section}`;
                if (c.subsection) {
                    metaHTML += `<br><span class="chunk-meta-label">Subsection:</span> ${c.subsection}`;
                }
                metaHTML += `<br><span class="chunk-meta-label">Page:</span> ${c.page}`;
                
                let scoresHTML = '';
                if (c.similarity_score !== null && c.rerank_score !== null) {
                    const sim = (c.similarity_score * 100).toFixed(0);
                    const rerank = (c.rerank_score * 100).toFixed(0);
                    const rerankClass = rerank > 70 ? 'high' : '';
                    scoresHTML = `
                        <div class="scores">
                            <span class="score-badge score-sim">Similarity: ${sim}%</span>
                            <span class="score-badge score-rerank ${rerankClass}">Re-Rank: ${rerank}%</span>
                        </div>
                    `;
                }
                
                div.innerHTML = `
                    <div class="chunk-header">
                        <div class="chunk-left">
                            <span class="chunk-rank">Chunk #${c.rank}</span>
                            <div class="chunk-meta">${metaHTML}</div>
                            <div class="chunk-source">${c.source}</div>
                        </div>
                    </div>
                    ${scoresHTML}
                `;
                chunksDiv.appendChild(div);
            });
        }

        init();
    </script>
</body>
</html>'''


def load_json_files(filepaths):
    """Load all JSON files and extract necessary data."""
    data = []
    
    for filepath in sorted(filepaths):  # Sort to maintain order
        with open(filepath, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
            
            # Extract data
            item = {
                'question': query_data['question'],
                'answer': query_data['answer'],
                'chunks': query_data['chunks']
            }
            
            data.append(item)
    
    return data


def generate_dashboard(input_files, output_file):
    """Generate HTML dashboard from JSON files."""
    
    print(f"Loading {len(input_files)} JSON files...")
    data = load_json_files(input_files)
    
    print(f"Generating dashboard with {len(data)} questions...")
    
    # Generate HTML
    html = DASHBOARD_TEMPLATE.replace('DATA_PLACEHOLDER', json.dumps(data, indent=2))
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Dashboard generated: {output_file}")
    print(f"   Questions included: {len(data)}")


def main():
    parser = argparse.ArgumentParser(description='Generate dashboard from multiple query JSON files')
    parser.add_argument('--input', nargs='+', required=True, help='JSON files to include')
    parser.add_argument('--output', default='dashboard.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    generate_dashboard(args.input, args.output)


if __name__ == '__main__':
    main()


# USAGE:
# python generate_multi_dashboard.py --input run4/*.json --output my_dashboard.html
# python generate_multi_dashboard.py --input run4/query_*.json --output results.html