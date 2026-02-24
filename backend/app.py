"""
app.py — FastAPI backend for OpenFOAM RAG web application.

Endpoints:
    GET  /health      — Status check
    POST /api/query   — Single-question RAG
    POST /api/report  — Report generation with SSE streaming
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from embedder import get_model, embed_query
from supabase_client import search_chunks
from reranker import rerank_chunks

load_dotenv()

# ---------------------------------------------------------------------------
# Concurrency limiter — at most 2 LLM calls in parallel
# ---------------------------------------------------------------------------
_semaphore = asyncio.Semaphore(2)

# ---------------------------------------------------------------------------
# Lifespan: pre-load embedding model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model loads lazily on first request to keep startup fast
    yield


app = FastAPI(title="OpenFOAM RAG API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _get_llm(max_tokens: int = 2048, api_key: str | None = None) -> ChatGoogleGenerativeAI:
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("No Google API key provided")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=max_tokens,
        google_api_key=key,
    )


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunk dicts into numbered context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta_parts = []
        if chunk.get("source"):
            meta_parts.append(f"Source: {chunk['source']}")
        if chunk.get("section") and chunk["section"] != "N/A":
            meta_parts.append(f"Section: {chunk['section']}")
        if chunk.get("subsection") and chunk["subsection"] != "N/A":
            meta_parts.append(f"Subsection: {chunk['subsection']}")
        if chunk.get("subsubsection") and chunk["subsubsection"] != "N/A":
            meta_parts.append(f"Subsubsection: {chunk['subsubsection']}")
        if chunk.get("page"):
            meta_parts.append(f"Page: {chunk['page']}")
        if chunk.get("similarity") is not None:
            meta_parts.append(f"Score: {round(chunk['similarity'], 4)}")

        metadata_str = " | ".join(meta_parts) if meta_parts else "No metadata"
        header = f"[{i}] ({metadata_str})"
        parts.append(f"{header}\n{chunk['content']}\n")

    return "\n" + "=" * 80 + "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt templates (mirrored from src/rag/)
# ---------------------------------------------------------------------------

QUERY_TEMPLATE = """You are an OpenFOAM technical expert assisting users with OpenFOAM concepts, configuration, numerical schemes, and runtime errors.

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
- Do not include meta-commentary (e.g., "Based on the context…").
- Do not fabricate or infer information beyond the provided chunks.
- If mathematical equations appear, reproduce them using LaTeX.
- If the context only partially answers the question, clearly state what information is missing.
- If no relevant information exists, respond exactly:
  "This information is not available in the provided documentation."

STRUCTURE:
- Begin with a clear explanation of the concept or solution.
- Use inline citations [n] immediately after supported claims.
- End with a References section listing only cited chunks in the following format:

References
[n] Source | Section | Page

If any metadata field is missing, omit it rather than inventing it.

---
CONTEXT:
{context}

---
QUESTION:
{question}"""

DECOMPOSE_TEMPLATE = """You are a technical writing assistant specializing in OpenFOAM documentation.

Given a report topic, break it into 3-5 focused sub-questions that together cover the topic comprehensively.
Each sub-question should be specific enough to retrieve relevant documentation chunks.

Return ONLY a JSON array of strings. No markdown, no explanation.

Example:
["What is the SIMPLE algorithm and how does it work?", "How does PIMPLE differ from SIMPLE?", "What under-relaxation factors are used in fvSolution?"]

Report topic: {prompt}"""

REPORT_TEMPLATE = """You are an OpenFOAM technical report writer producing detailed, multi-section reports for an academic workshop audience.

Write a structured technical report based ONLY on the provided context chunks. The report should be thorough and substantive — aim for 1500-2500 words across all sections.

REPORT REQUIREMENTS:
- Use ## section headings to organize the report logically. Each section should have 2-4 paragraphs of substantive content.
- Every factual claim MUST have an inline citation [n] corresponding to the numbered context chunks.
- Multiple citations may be grouped (e.g., [1][3]) when supported by multiple chunks.
- Do not invent information beyond what is in the context.
- If a sub-topic is not covered by the context, state explicitly: "This topic is not covered in the available documentation."
- Reproduce exact technical terms, dictionary entries, file paths, equations, and code snippets when present in the context.
- If mathematical equations appear, reproduce them using LaTeX notation.
- Include specific OpenFOAM dictionary syntax, keyword names, and configuration examples when available in context.

DEPTH GUIDELINES:
- Do NOT write shallow one-sentence sections. Each ## section should thoroughly explain the topic using all relevant context chunks.
- When a context chunk contains code examples, dictionary syntax, or equations, include them in the report.
- Explain relationships between concepts (e.g., how one scheme choice affects another).
- When multiple approaches or options exist (e.g., different schemes), compare them.

STRUCTURE:
- Start with a brief introduction paragraph (2-3 sentences) framing the topic.
- Use ## headings for each major section.
- Within sections, use paragraphs and bullet lists as appropriate for clarity.
- End with a ## References section listing ONLY cited chunks in the format:
  [n] Source | Section | Page

CONTEXT:
{context}

REPORT TOPIC:
{prompt}"""


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    api_key: str | None = None
    k: int = Field(default=15, ge=1, le=50)
    top_n: int = Field(default=5, ge=1, le=20)


class ReportRequest(BaseModel):
    question: str
    api_key: str | None = None
    k_per_query: int = Field(default=10, ge=1, le=50)
    max_chunks: int = Field(default=20, ge=1, le=50)


class ChunkResponse(BaseModel):
    rank: int
    content: str
    section: str | None = None
    subsection: str | None = None
    subsubsection: str | None = None
    page: str | None = None
    source: str | None = None
    similarity: float | None = None


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkResponse]
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    model_loaded = get_model() is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Single-question RAG: embed → search → generate."""
    async with _semaphore:
        # 1. Embed the query
        query_vector = await asyncio.to_thread(embed_query, req.question)

        # 2. Search Supabase (dense retrieval)
        chunks = await asyncio.to_thread(search_chunks, query_vector, req.k)

        if not chunks:
            return QueryResponse(
                answer="No relevant documentation chunks were found for your query.",
                chunks=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # 3. Rerank with Gemini
        chunks = await asyncio.to_thread(rerank_chunks, req.question, chunks, req.top_n, req.api_key)

        # 4. Format context and generate
        context = _format_context(chunks)
        llm = _get_llm(max_tokens=2048, api_key=req.api_key)
        prompt = ChatPromptTemplate.from_template(QUERY_TEMPLATE)
        chain = prompt | llm | StrOutputParser()

        answer = await asyncio.to_thread(
            chain.invoke, {"context": context, "question": req.question}
        )

        # 5. Build response
        chunk_responses = [
            ChunkResponse(
                rank=i,
                content=c["content"],
                section=c.get("section"),
                subsection=c.get("subsection"),
                subsubsection=c.get("subsubsection"),
                page=c.get("page"),
                source=c.get("source"),
                similarity=c.get("similarity"),
            )
            for i, c in enumerate(chunks, 1)
        ]

        return QueryResponse(
            answer=answer,
            chunks=chunk_responses,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


@app.post("/api/report")
async def report_endpoint(req: ReportRequest):
    """Report generation with SSE streaming for progress updates."""

    async def event_stream():
        try:
            # --- Step 1: Decompose ---
            yield _sse_event("progress", {"step": "decomposing", "message": "Breaking down your question into sub-topics..."})

            llm = _get_llm(max_tokens=8192, api_key=req.api_key)
            decompose_prompt = ChatPromptTemplate.from_template(DECOMPOSE_TEMPLATE)
            decompose_chain = decompose_prompt | llm | StrOutputParser()

            async with _semaphore:
                raw_output = await asyncio.to_thread(
                    decompose_chain.invoke, {"prompt": req.question}
                )

            sub_questions = _parse_sub_questions(raw_output, req.question)
            yield _sse_event("progress", {
                "step": "decomposed",
                "message": f"Identified {len(sub_questions)} sub-topics",
                "sub_questions": sub_questions,
            })

            # --- Step 2: Retrieve per sub-question ---
            yield _sse_event("progress", {"step": "retrieving", "message": "Searching documentation..."})

            seen: dict[str, dict] = {}  # content_hash -> chunk dict with best score
            for i, sq in enumerate(sub_questions):
                query_vector = await asyncio.to_thread(embed_query, sq)
                chunks = await asyncio.to_thread(search_chunks, query_vector, req.k_per_query)

                # Rerank per sub-question
                chunks = await asyncio.to_thread(rerank_chunks, sq, chunks, 5, req.api_key)

                for chunk in chunks:
                    content_hash = hashlib.md5(chunk["content"].encode()).hexdigest()
                    existing = seen.get(content_hash)
                    score = chunk.get("rerank_score", chunk.get("similarity", 0))
                    existing_score = existing.get("rerank_score", existing.get("similarity", 0)) if existing else 0
                    if existing is None or score > existing_score:
                        seen[content_hash] = chunk

                yield _sse_event("progress", {
                    "step": "retrieving",
                    "message": f"Retrieved for sub-topic {i + 1}/{len(sub_questions)}",
                })

            # Sort by similarity, take top max_chunks
            merged = sorted(
                seen.values(),
                key=lambda x: x.get("rerank_score", x.get("similarity", 0)),
                reverse=True,
            )[:req.max_chunks]

            yield _sse_event("progress", {
                "step": "retrieved",
                "message": f"Found {len(merged)} unique relevant chunks",
            })

            # --- Step 3: Generate report ---
            yield _sse_event("progress", {"step": "generating", "message": "Writing report..."})

            context = _format_context(merged)
            report_prompt = ChatPromptTemplate.from_template(REPORT_TEMPLATE)
            report_chain = report_prompt | llm | StrOutputParser()

            async with _semaphore:
                report = await asyncio.to_thread(
                    report_chain.invoke, {"context": context, "prompt": req.question}
                )

            # Build chunk list for response
            chunk_list = [
                {
                    "rank": i,
                    "content": c["content"],
                    "section": c.get("section"),
                    "subsection": c.get("subsection"),
                    "subsubsection": c.get("subsubsection"),
                    "page": c.get("page"),
                    "source": c.get("source"),
                    "similarity": c.get("similarity"),
                }
                for i, c in enumerate(merged, 1)
            ]

            yield _sse_event("complete", {
                "answer": report,
                "chunks": chunk_list,
                "sub_questions": sub_questions,
                "mode": "report",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _parse_sub_questions(raw_output: str, fallback: str) -> list[str]:
    """Parse LLM output into a list of sub-questions."""
    cleaned = raw_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        sub_questions = json.loads(cleaned)
        if isinstance(sub_questions, list) and len(sub_questions) > 0:
            return sub_questions
    except json.JSONDecodeError:
        pass

    return [fallback]
