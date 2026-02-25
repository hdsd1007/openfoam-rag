"""
analyze_feedback.py — CLI tool to analyze user feedback from Supabase.

Usage:
    python scripts/analyze_feedback.py                    # Full analysis
    python scripts/analyze_feedback.py --downvotes-only   # Focus on negative feedback
    python scripts/analyze_feedback.py --diagnose         # LLM diagnosis of downvotes + prompt suggestions
    python scripts/analyze_feedback.py --export report.json  # Export for records
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

# Allow importing from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

from supabase_client import fetch_feedback

# OpenFOAM topic keywords for clustering
TOPIC_KEYWORDS = {
    "mesh": ["mesh", "blockMesh", "snappyHexMesh", "grading", "cell", "face", "point", "boundary"],
    "solver": ["solver", "SIMPLE", "PISO", "PIMPLE", "fvSolution", "linear", "GAMG", "PCG", "PBiCG"],
    "scheme": ["scheme", "fvSchemes", "divSchemes", "gradSchemes", "laplacianSchemes", "interpolation", "discretization", "Gauss"],
    "boundary": ["boundary", "inlet", "outlet", "wall", "patch", "fixedValue", "zeroGradient"],
    "turbulence": ["turbulence", "k-epsilon", "k-omega", "SST", "LES", "RANS", "kEpsilon", "kOmega"],
    "runtime": ["error", "runtime", "crash", "convergence", "diverge", "fatal", "FOAM FATAL"],
    "postprocessing": ["paraview", "postProcess", "functionObject", "sample", "probe", "forces"],
    "transport": ["transport", "viscosity", "density", "nu", "rho", "thermophysical"],
}

# Current prompts — so the LLM can suggest targeted edits
CURRENT_PROMPTS = {
    "QUERY_TEMPLATE": """You are an OpenFOAM technical expert assisting users with OpenFOAM concepts, configuration, numerical schemes, and runtime errors.

Provide clear, technically precise explanations grounded strictly in the retrieved documentation.

CRITICAL CITATION RULES:
- Every paragraph containing factual or technical claims MUST include at least one inline citation in the format [n].
- Citations must correspond exactly to the numbered context chunks.
- Multiple citations may be grouped (e.g., [1][3]) when supported by multiple chunks.
- Do not invent citations or metadata.
- If a claim cannot be directly supported by the provided context, remove it.

WRITING GUIDELINES:
- Answer naturally and directly, as an expert would.
- Prefer specific configuration names, dictionary entries, file paths, equations, and exact syntax when present in the context.
- Reproduce exact technical wording or code snippets when available.
- Ignore retrieved chunks that are not directly relevant to the question.
- Do not include meta-commentary (e.g., "Based on the context...").
- Do not fabricate or infer information beyond the provided chunks.
- If mathematical equations appear, reproduce them using LaTeX.
- If the context only partially answers the question, clearly state what information is missing.
- If no relevant information exists, respond exactly:
  "This information is not available in the provided documentation."

STRUCTURE:
- Begin with a clear explanation of the concept or solution.
- Use inline citations [n] immediately after supported claims.
- End with a References section listing only cited chunks.""",

    "REPORT_TEMPLATE": """You are an OpenFOAM technical report writer producing detailed, multi-section reports for an academic workshop audience.

Write a structured technical report based ONLY on the provided context chunks. The report should be thorough and substantive — aim for 1500-2500 words across all sections.

REPORT REQUIREMENTS:
- Use ## section headings to organize the report logically.
- Every factual claim MUST have an inline citation [n].
- Do not invent information beyond what is in the context.
- Reproduce exact technical terms, dictionary entries, file paths, equations, and code snippets.

DEPTH GUIDELINES:
- Each ## section should thoroughly explain the topic using all relevant context chunks.
- Include code examples, dictionary syntax, or equations from context.
- Explain relationships between concepts.
- Compare multiple approaches or options when they exist.""",
}

DIAGNOSE_PROMPT = """You are an expert at improving RAG (Retrieval-Augmented Generation) systems. You are analyzing user feedback on an OpenFOAM documentation Q&A system.

Below are downvoted answers — each with the user's query, the system's answer, and optionally a user comment explaining what was wrong.

YOUR TASK:
1. Identify recurring PATTERNS across the downvotes (e.g., "answers are too verbose", "wrong topic retrieved", "citations missing", "hallucinated content").
2. For each pattern, diagnose the ROOT CAUSE: is it a retrieval problem (wrong chunks), a prompt problem (answer generation), or a data problem (missing docs)?
3. Suggest SPECIFIC, actionable changes to the generation prompts below. Quote the exact line to change and provide the replacement.

CURRENT PROMPTS:
{prompts}

DOWNVOTED FEEDBACK ({count} entries):
{feedback}

Respond in this exact format:

## Patterns Found
- **Pattern name**: Description (seen in N entries)

## Root Cause Analysis
- **Pattern name**: [retrieval | prompt | data] — Explanation

## Suggested Prompt Changes
For each suggestion:
- Which prompt (QUERY_TEMPLATE or REPORT_TEMPLATE)
- What to add/change (quote the specific line or section)
- The exact new text

## Summary
One paragraph: overall health of the system and priority actions.
"""


def classify_topic(query: str) -> list[str]:
    """Classify a query into OpenFOAM topic categories."""
    query_lower = query.lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw.lower() in query_lower for kw in keywords):
            topics.append(topic)
    return topics or ["other"]


def analyze(rows: list[dict], downvotes_only: bool = False) -> dict:
    """Compute feedback analytics."""
    if downvotes_only:
        rows = [r for r in rows if r["vote"] == "down"]

    total = len(rows)
    if total == 0:
        return {"total": 0, "message": "No feedback entries found."}

    upvotes = sum(1 for r in rows if r["vote"] == "up")
    downvotes = sum(1 for r in rows if r["vote"] == "down")

    # Mode breakdown
    mode_counts = Counter(r.get("mode", "quick") for r in rows)

    # Topic clustering
    topic_counts: dict[str, int] = Counter()
    for r in rows:
        for topic in classify_topic(r.get("query", "")):
            topic_counts[topic] += 1

    # Comments
    comments_up = [
        {"query": r["query"], "comment": r["comment"]}
        for r in rows if r.get("comment") and r["vote"] == "up"
    ]
    comments_down = [
        {"query": r["query"], "comment": r["comment"]}
        for r in rows if r.get("comment") and r["vote"] == "down"
    ]

    # Downvote details (full query + answer + comment)
    downvote_details = []
    for r in rows:
        if r["vote"] == "down":
            detail = {
                "query": r["query"],
                "mode": r.get("mode", "quick"),
                "created_at": r.get("created_at", ""),
            }
            if r.get("answer_text"):
                text = r["answer_text"]
                detail["answer_preview"] = text[:500] + "..." if len(text) > 500 else text
            if r.get("comment"):
                detail["comment"] = r["comment"]
            downvote_details.append(detail)

    return {
        "total": total,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "ratio": round(upvotes / downvotes, 2) if downvotes > 0 else None,
        "by_mode": dict(mode_counts),
        "by_topic": dict(topic_counts.most_common()),
        "downvote_details": downvote_details,
        "comments_on_upvotes": comments_up,
        "comments_on_downvotes": comments_down,
    }


def run_diagnosis(rows: list[dict], days: int = 7) -> str:
    """Send downvoted feedback to Gemini for pattern diagnosis and prompt suggestions."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Filter to downvotes with answer_text, within the time window
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    downvotes = []
    for r in rows:
        if r["vote"] != "down":
            continue
        created = r.get("created_at", "")
        if created:
            try:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if ts < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        entry = f"Query: {r['query']}\nMode: {r.get('mode', 'quick')}"
        if r.get("answer_text"):
            # Truncate to keep token usage reasonable
            answer = r["answer_text"][:2000]
            entry += f"\nAnswer: {answer}"
        if r.get("comment"):
            entry += f"\nUser comment: {r['comment']}"
        downvotes.append(entry)

    if not downvotes:
        return f"No downvotes found in the last {days} days. Nothing to diagnose."

    # Cap at 10 entries to control cost
    if len(downvotes) > 10:
        downvotes = downvotes[:10]
        print(f"  (Analyzing first 10 of {len(downvotes)} downvotes to control cost)")

    feedback_text = "\n\n---\n\n".join(f"[{i+1}]\n{d}" for i, d in enumerate(downvotes))
    prompts_text = "\n\n".join(f"### {name}\n```\n{text}\n```" for name, text in CURRENT_PROMPTS.items())

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_output_tokens=4096,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    prompt = ChatPromptTemplate.from_template(DIAGNOSE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "prompts": prompts_text,
        "feedback": feedback_text,
        "count": len(downvotes),
    })
    return result


def print_report(analysis: dict) -> None:
    """Print a human-readable feedback report."""
    print("=" * 60)
    print("  OPENFOAM RAG — FEEDBACK ANALYSIS")
    print("=" * 60)

    if analysis["total"] == 0:
        print("\nNo feedback entries found.")
        return

    print(f"\nTotal feedback:  {analysis['total']}")
    print(f"  Upvotes:       {analysis['upvotes']}")
    print(f"  Downvotes:     {analysis['downvotes']}")
    if analysis["ratio"] is not None:
        print(f"  Up/Down ratio: {analysis['ratio']}")

    print(f"\nBy mode:")
    for mode, count in analysis["by_mode"].items():
        print(f"  {mode}: {count}")

    print(f"\nBy topic:")
    for topic, count in analysis["by_topic"].items():
        print(f"  {topic}: {count}")

    if analysis["downvote_details"]:
        print(f"\n{'─' * 60}")
        print(f"  DOWNVOTE DETAILS ({len(analysis['downvote_details'])})")
        print(f"{'─' * 60}")
        for i, d in enumerate(analysis["downvote_details"], 1):
            print(f"\n  [{i}] Query: {d['query']}")
            print(f"      Mode: {d['mode']}  |  Date: {d.get('created_at', 'N/A')}")
            if d.get("comment"):
                print(f"      Comment: {d['comment']}")
            if d.get("answer_preview"):
                print(f"      Answer preview: {d['answer_preview'][:200]}...")

    if analysis["comments_on_downvotes"]:
        print(f"\n{'─' * 60}")
        print(f"  COMMENTS ON DOWNVOTES")
        print(f"{'─' * 60}")
        for c in analysis["comments_on_downvotes"]:
            print(f"  - [{c['query'][:50]}...] {c['comment']}")

    if analysis["comments_on_upvotes"]:
        print(f"\n{'─' * 60}")
        print(f"  COMMENTS ON UPVOTES")
        print(f"{'─' * 60}")
        for c in analysis["comments_on_upvotes"]:
            print(f"  - [{c['query'][:50]}...] {c['comment']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze OpenFOAM RAG user feedback")
    parser.add_argument("--downvotes-only", action="store_true", help="Only analyze downvoted feedback")
    parser.add_argument("--diagnose", action="store_true", help="Use LLM to diagnose downvote patterns and suggest prompt fixes")
    parser.add_argument("--days", type=int, default=7, help="For --diagnose: only analyze downvotes from last N days (default: 7)")
    parser.add_argument("--export", type=str, default=None, help="Export analysis to JSON file")
    parser.add_argument("--limit", type=int, default=500, help="Max rows to fetch (default: 500)")
    args = parser.parse_args()

    vote_filter = "down" if args.downvotes_only else None
    rows = fetch_feedback(vote_filter=vote_filter, limit=args.limit)
    analysis = analyze(rows, downvotes_only=args.downvotes_only)

    print_report(analysis)

    if args.diagnose:
        print("=" * 60)
        print(f"  LLM DIAGNOSIS (last {args.days} days)")
        print("=" * 60)
        print("\n  Analyzing downvotes with Gemini...\n")
        diagnosis = run_diagnosis(rows, days=args.days)
        print(diagnosis)
        if args.export:
            analysis["diagnosis"] = diagnosis

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
