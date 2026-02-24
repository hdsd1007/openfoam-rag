"""
reranker.py — LLM-based reranker using Gemini Flash.

Zero RAM overhead — uses Gemini API to score chunk relevance.
Free-tier compatible (no local model needed).
"""

import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


RERANK_TEMPLATE = """You are a relevance scoring system. Given a query and a list of text chunks, return the indices of the most relevant chunks in order of relevance.

Query: {query}

Chunks:
{chunks_text}

Return ONLY a JSON array of chunk indices (0-based) ordered by relevance to the query, most relevant first. Select at most {top_n} chunks.
Example: [3, 0, 7, 1, 5]"""


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 5,
    api_key: str | None = None,
) -> list[dict]:
    """
    Rerank chunks using Gemini Flash and return top_n.

    Each chunk dict must have a 'content' key.
    Returns chunks sorted by LLM-judged relevance (best first).
    """
    if not chunks:
        return []

    if len(chunks) <= top_n:
        return chunks

    # Build chunk summaries for the LLM (first 300 chars each to save tokens)
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        preview = chunk["content"][:300].replace("\n", " ")
        section = chunk.get("section", "N/A")
        chunks_text += f"[{i}] Section: {section} | {preview}\n\n"

    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)[:top_n]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_output_tokens=256,
        google_api_key=key,
    )

    prompt = ChatPromptTemplate.from_template(RERANK_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "query": query,
            "chunks_text": chunks_text,
            "top_n": top_n,
        })

        # Parse the JSON array of indices
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        indices = json.loads(cleaned)

        if isinstance(indices, list):
            # Filter valid indices and deduplicate
            seen = set()
            reranked = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                    seen.add(idx)
                    chunks[idx]["rerank_score"] = len(indices) - len(reranked)
                    reranked.append(chunks[idx])
                if len(reranked) >= top_n:
                    break
            if reranked:
                return reranked

    except Exception:
        pass

    # Fallback: return top_n by original similarity score
    return sorted(chunks, key=lambda x: x.get("similarity", 0), reverse=True)[:top_n]
