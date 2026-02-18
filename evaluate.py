"""
evaluate.py — ROUGE + RAGAS evaluation for OpenFOAM RAG pipeline outputs.

WHAT THIS NEEDS:
    - Your saved query JSON files (produced by ask.py --save)
    - A ground-truth TSV file: questions.tsv with columns: question | reference_answer
      (only needed for ROUGE and RAGAS Answer Correctness)
    - If you have NO ground truth, use --no-reference mode (RAGAS reference-free only)

INSTALL:
    pip install rouge-score ragas datasets langchain-google-genai

USAGE:
    # With ground truth reference answers (full ROUGE + RAGAS):
    python evaluate.py --input query_outputs/*.json --references questions.tsv --output eval_report.json

    # Without ground truth (RAGAS reference-free metrics only: faithfulness, answer_relevancy, context_precision):
    python evaluate.py --input query_outputs/*.json --no-reference --output eval_report.json

TSV FORMAT (questions.tsv):
    question\treference_answer
    What is runTimeModifiable?\tThe runTimeModifiable keyword, when set to yes...
    How does OpenFOAM handle missing keywords?\tOpenFOAM assigns default values...
"""

import os
import warnings
import argparse
import json
import csv
import statistics
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ── ROUGE ──────────────────────────────────────────────────────────────────────
from rouge_score import rouge_scorer


# ── RAGAS ──────────────────────────────────────────────────────────────────────
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from ragas.metrics import answer_correctness          # needs reference
from datasets import Dataset


# ==============================================================================
# STEP 1: Load query JSON files produced by ask.py
# ==============================================================================

def load_query_jsons(filepaths: list[str]) -> list[dict]:
    """Load all query output JSON files into a list of dicts."""
    queries = []
    for fp in filepaths:
        with open(fp, 'r', encoding='utf-8') as f:
            queries.append(json.load(f))
    print(f"  Loaded {len(queries)} query files.")
    return queries


# ==============================================================================
# STEP 2: Load reference answers from TSV
# ==============================================================================

def load_references(tsv_path: str) -> dict[str, str]:
    """
    Load a TSV mapping question → reference_answer.
    TSV must have a header row: question\treference_answer
    Matching is done by exact question string.
    """
    refs = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            refs[row['question'].strip()] = row['reference_answer'].strip()
    print(f"  Loaded {len(refs)} reference answers from {tsv_path}")
    return refs


# ==============================================================================
# STEP 3: ROUGE scoring
# ==============================================================================

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores for a list of predictions
    against their corresponding references.

    Returns average scores across all pairs.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1_scores, r2_scores, rl_scores = [], [], []

    for pred, ref in zip(predictions, references):
        # Skip abstained answers — they are not penalized but not scored
        if pred.strip().startswith("This information is not available"):
            continue
        scores = scorer.score(ref, pred)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)

    if not r1_scores:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "scored_count": 0}

    return {
        "rouge1":        round(statistics.mean(r1_scores), 4),
        "rouge2":        round(statistics.mean(r2_scores), 4),
        "rougeL":        round(statistics.mean(rl_scores), 4),
        "scored_count":  len(r1_scores),   # how many non-abstained answers were scored
    }


# ==============================================================================
# STEP 4: Build RAGAS Dataset from your query JSONs
# ==============================================================================

def build_ragas_dataset(queries: list[dict], references: dict | None) -> Dataset:
    """
    Convert your query JSON structure into a RAGAS-compatible HuggingFace Dataset.

    RAGAS expects these columns:
        question        : str
        answer          : str   (the generated answer)
        contexts        : list[str]  (the retrieved chunk texts — NOT the previews, the full content)
        ground_truth    : str   (optional, only needed for answer_correctness)

    NOTE: Your JSON only stores content_preview (300 chars). This is a limitation.
    For faithful RAGAS scoring, store full chunk content in your JSONs (see recommendation below).
    If previews are all you have, this will still run but faithfulness scores will be deflated.
    """
    rows = []
    for q in queries:
        question = q['question']
        answer   = q['answer']

        # Extract context strings from chunks
        # Using content_preview because that's what ask.py saves.
        # To fix: in ask.py, change content_preview to doc.page_content (full text).
        contexts = [chunk['content_preview'] for chunk in q['chunks']]

        row = {
            "question": question,
            "answer":   answer,
            "contexts": contexts,
        }

        if references:
            ref = references.get(question.strip(), "")
            row["ground_truth"] = ref

        rows.append(row)

    return Dataset.from_list(rows)


# ==============================================================================
# STEP 5: Run RAGAS evaluation
# ==============================================================================

def compute_ragas(dataset: Dataset, use_reference: bool, llm=None, embeddings=None) -> dict:
    """
    Run RAGAS evaluation.

    Metrics used:
        Always:   faithfulness, answer_relevancy, context_precision
        If refs:  answer_correctness (requires ground_truth column)

    llm and embeddings: if None, RAGAS uses its default (OpenAI).
    If you're using Gemini, pass your LangChain LLM wrapper and embeddings here.
    """
    metrics = [faithfulness, answer_relevancy, context_relevancy]
    if use_reference and 'ground_truth' in dataset.column_names:
        metrics.append(answer_correctness)

    kwargs = {"dataset": dataset, "metrics": metrics}
    if llm:
        kwargs["llm"] = llm
    if embeddings:
        kwargs["embeddings"] = embeddings

    result = ragas_evaluate(**kwargs)

    # Convert to plain dict with rounded floats
    scores = {}
    result_df = result.to_pandas()
    for metric_col in ['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness']:
        if metric_col in result_df.columns:
            col_vals = result_df[metric_col].dropna().tolist()
            if col_vals:
                scores[metric_col] = {
                    "mean":   round(statistics.mean(col_vals), 4),
                    "median": round(statistics.median(col_vals), 4),
                    "min":    round(min(col_vals), 4),
                    "max":    round(max(col_vals), 4),
                    "per_query": [round(v, 4) for v in col_vals],
                }

    return scores


# ==============================================================================
# STEP 6: Generate final report
# ==============================================================================

def generate_report(queries, rouge_results, ragas_results, output_path: str):
    """
    Merge everything into one structured JSON report and print a summary.
    """
    report = {
        "summary": {
            "total_queries": len(queries),
            "has_reranker": queries[0].get("has_reranker", False) if queries else None,
            "parser": queries[0].get("parser", "unknown") if queries else None,
        },
        "rouge": rouge_results,
        "ragas": ragas_results,
        "per_query": []
    }

    # Per-query detail
    for q in queries:
        entry = {
            "question": q['question'],
            "answer_snippet": q['answer'][:200],
            "abstained": q['answer'].strip().startswith("This information is not available"),
            "num_chunks": len(q['chunks']),
            "sources": list(set(c['source'] for c in q['chunks'])),
        }
        report["per_query"].append(entry)

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total queries: {report['summary']['total_queries']}")

    if rouge_results:
        print(f"\nROUGE (averaged over non-abstained answers):")
        print(f"  ROUGE-1 : {rouge_results.get('rouge1', 'N/A')}")
        print(f"  ROUGE-2 : {rouge_results.get('rouge2', 'N/A')}")
        print(f"  ROUGE-L : {rouge_results.get('rougeL', 'N/A')}")
        print(f"  Scored  : {rouge_results.get('scored_count', 0)} / {len(queries)} queries")

    if ragas_results:
        print(f"\nRAGAS:")
        for metric, vals in ragas_results.items():
            print(f"  {metric:<25}: mean={vals['mean']}, median={vals['median']}")

    print(f"\nFull report saved to: {output_path}")
    print("="*70)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ROUGE + RAGAS evaluation for OpenFOAM RAG pipeline outputs.'
    )
    parser.add_argument('--input',        nargs='+', required=True,
                        help='Query JSON files to evaluate (e.g. query_outputs/*.json)')
    parser.add_argument('--references',   default=None,
                        help='TSV file with question→reference_answer pairs (for ROUGE + answer_correctness)')
    parser.add_argument('--no-reference', action='store_true',
                        help='Skip ROUGE and answer_correctness; run only reference-free RAGAS metrics')
    parser.add_argument('--output',       default='eval_report.json',
                        help='Output JSON file for the evaluation report')
    parser.add_argument('--gemini-key',   default=None,
                        help='Optional: Google API key if you want RAGAS to use Gemini instead of OpenAI')
    args = parser.parse_args()

    print("\nLoading query files...")
    queries = load_query_jsons(args.input)

    # ── Optional: load references ──────────────────────────────────────────────
    references = None
    if not args.no_reference:
        if not args.references:
            print("\nWARNING: --references not provided and --no-reference not set.")
            print("         Falling back to reference-free RAGAS only (no ROUGE).")
            args.no_reference = True
        else:
            print("\nLoading reference answers...")
            references = load_references(args.references)

    # ── ROUGE ─────────────────────────────────────────────────────────────────
    rouge_results = {}
    if not args.no_reference and references:
        print("\nComputing ROUGE scores...")
        predictions, refs_list = [], []
        for q in queries:
            qtext = q['question'].strip()
            if qtext in references:
                predictions.append(q['answer'])
                refs_list.append(references[qtext])
            else:
                print(f"  WARNING: No reference found for: {qtext[:60]}...")

        if predictions:
            rouge_results = compute_rouge(predictions, refs_list)
            print(f"  ROUGE computed for {rouge_results.get('scored_count', 0)} queries.")
        else:
            print("  No matching question-reference pairs found. Check your TSV headers.")

    # ── RAGAS LLM setup (optional: use your Gemini LLM) ───────────────────────
    # If you want RAGAS to use Gemini (same LLM as your pipeline) instead of
    # defaulting to OpenAI, uncomment and configure the block below.
    # This requires: pip install langchain-google-genai
    # and your GOOGLE_API_KEY set in environment.
    #
    # ragas_llm = None
    # ragas_embeddings = None
    # if args.gemini_key:
    #     os.environ["GOOGLE_API_KEY"] = args.gemini_key
    #     from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    #     from ragas.llms import LangchainLLMWrapper
    #     from ragas.embeddings import LangchainEmbeddingsWrapper
    #     ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-1.5-flash"))
    #     ragas_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    #     print("  RAGAS will use Gemini LLM and embeddings.")

    # ── RAGAS ─────────────────────────────────────────────────────────────────
    print("\nBuilding RAGAS dataset...")
    ragas_dataset = build_ragas_dataset(queries, references if not args.no_reference else None)

    print("Running RAGAS evaluation (this makes LLM calls — may take a minute)...")
    ragas_results = compute_ragas(
        dataset=ragas_dataset,
        use_reference=(not args.no_reference),
        llm=None,           # Replace with ragas_llm if using Gemini block above
        embeddings=None     # Replace with ragas_embeddings if using Gemini block above
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    generate_report(queries, rouge_results, ragas_results, args.output)


if __name__ == '__main__':
    main()


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
#
# 1. Reference-free only (no ground truth needed):
#    clear
#
# 2. Full evaluation with ground truth references:
#    python evaluate.py --input query_outputs/*.json --references questions.tsv --output eval_report.json
#
# 3. Using Gemini as the RAGAS judge LLM:
#    python evaluate.py --input query_outputs/*.json --no-reference --gemini-key YOUR_KEY --output eval_report.json
#
# ==============================================================================
# TSV FORMAT FOR --references (questions.tsv):
# ==============================================================================
# question\treference_answer
# What is the purpose of the runTimeModifiable keyword?\tThe runTimeModifiable keyword in controlDict, when set to yes, allows dictionaries to be re-read during the simulation run if they are modified by the user.
# How does OpenFOAM handle missing keywords?\tFor optional entries omitted from a dictionary, OpenFOAM assigns default values automatically without raising an error.
#
# ==============================================================================
# IMPORTANT NOTE ON CONTENT PREVIEWS:
# ==============================================================================
# Your ask.py currently saves only content_preview (first 300 chars of each chunk).
# RAGAS Faithfulness compares claims in the answer against the full chunk text.
# With truncated previews, faithfulness scores will be underestimated.
#
# To fix this permanently, change ONE LINE in ask.py (line ~50):
#   BEFORE: "content_preview": doc.page_content[:300]
#   AFTER:  "content_preview": doc.page_content          # store full chunk
#
# Then re-run your queries with --save. The evaluate.py will automatically
# use the full content from content_preview for RAGAS context.
# ==============================================================================