"""
report_pipeline.py — Multi-query report generation pipeline.

Decomposes a report prompt into sub-questions, retrieves chunks for each,
deduplicates and merges, then generates a structured report with citations.
"""

import json
import hashlib
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.pipeline_e2e import format_context_with_metadata


def decompose_query(prompt, llm, track_tokens=None):
    """
    Break a report-level prompt into 3-5 focused sub-questions using LLM.

    Args:
        prompt: The report topic/prompt.
        llm: LangChain-compatible LLM instance.
        track_tokens: Optional string label for token tracking.

    Returns:
        List of sub-question strings.
    """
    template = """You are a technical writing assistant specializing in OpenFOAM documentation.

Given a report topic, break it into 3-5 focused sub-questions that together cover the topic comprehensively.
Each sub-question should be specific enough to retrieve relevant documentation chunks.

Return ONLY a JSON array of strings. No markdown, no explanation.

Example:
["What is the SIMPLE algorithm and how does it work?", "How does PIMPLE differ from SIMPLE?", "What under-relaxation factors are used in fvSolution?"]

Report topic: {prompt}"""

    chat_prompt = ChatPromptTemplate.from_template(template)

    if track_tokens:
        from src.llm.token_tracker import tracker
        chain = chat_prompt | llm
        ai_message = chain.invoke({"prompt": prompt})
        raw_output = ai_message.content
        tracker.track(track_tokens, ai_message.usage_metadata)
    else:
        chain = chat_prompt | llm | StrOutputParser()
        raw_output = chain.invoke({"prompt": prompt})

    # Clean potential markdown fences
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

    # Fallback: return original prompt as single-element list
    return [prompt]


def retrieve_for_report(sub_questions, vector_db, reranker,
                        k_per_query=15, top_n_per_query=5,
                        max_unique_chunks=20):
    """
    Retrieve and rerank chunks for each sub-question, deduplicate, and merge.

    Args:
        sub_questions: List of sub-question strings.
        vector_db: ChromaDB vector store instance.
        reranker: CrossEncoder reranker model.
        k_per_query: Number of chunks to retrieve per sub-question.
        top_n_per_query: Number of chunks to keep after reranking per sub-question.
        max_unique_chunks: Maximum total unique chunks to return.

    Returns:
        (final_docs, metadata_dict)
    """
    # Track unique chunks by content hash -> (doc, best_rerank_score)
    seen = {}
    total_before_dedup = 0

    for sq in sub_questions:
        # Stage 1: Dense retrieval
        docs_with_scores = vector_db.similarity_search_with_score(sq, k=k_per_query)
        initial_docs = []
        for doc, score in docs_with_scores:
            doc.metadata['similarity_score'] = round(float(score), 4)
            initial_docs.append(doc)

        # Stage 2: Cross-encoder reranking
        pairs = [[sq, doc.page_content] for doc in initial_docs]
        rerank_scores = reranker.predict(pairs)

        for doc, score in zip(initial_docs, rerank_scores):
            doc.metadata['rerank_score'] = round(float(score), 4)

        # Sort by rerank score and take top_n
        ranked = sorted(initial_docs,
                        key=lambda x: x.metadata['rerank_score'],
                        reverse=True)[:top_n_per_query]

        total_before_dedup += len(ranked)

        # Deduplicate by content hash, keeping highest rerank score
        for doc in ranked:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            existing = seen.get(content_hash)
            if existing is None or doc.metadata['rerank_score'] > existing.metadata['rerank_score']:
                seen[content_hash] = doc

    # Sort merged set by rerank score, take top max_unique_chunks
    merged_docs = sorted(seen.values(),
                         key=lambda x: x.metadata['rerank_score'],
                         reverse=True)[:max_unique_chunks]

    metadata = {
        "sub_questions": sub_questions,
        "total_before_dedup": total_before_dedup,
        "total_after_dedup": len(seen),
        "final_chunks_used": len(merged_docs),
    }

    return merged_docs, metadata


def generate_report(prompt, merged_docs, llm, track_tokens=None):
    """
    Generate a structured report from merged retrieved chunks.

    Args:
        prompt: Original report prompt.
        merged_docs: List of LangChain Document objects (deduplicated, reranked).
        llm: LangChain-compatible LLM instance.
        track_tokens: Optional string label for token tracking.

    Returns:
        Report string with ## sections and [n] citations.
    """
    formatted_context = format_context_with_metadata(merged_docs)

    template = """You are an OpenFOAM technical report writer producing detailed, multi-section reports for an academic workshop audience.

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

    chat_prompt = ChatPromptTemplate.from_template(template)
    invoke_args = {"context": formatted_context, "prompt": prompt}

    if track_tokens:
        from src.llm.token_tracker import tracker
        chain = chat_prompt | llm
        ai_message = chain.invoke(invoke_args)
        report = ai_message.content
        tracker.track(track_tokens, ai_message.usage_metadata)
    else:
        chain = chat_prompt | llm | StrOutputParser()
        report = chain.invoke(invoke_args)

    return report


def run_report_pipeline(prompt, vector_db, llm, reranker,
                        k_per_query=15, top_n_per_query=5,
                        max_unique_chunks=20, track_tokens_prefix=None):
    """
    End-to-end report generation: decompose -> retrieve -> generate.

    Args:
        prompt: Report-level prompt string.
        vector_db: ChromaDB vector store instance.
        llm: LangChain-compatible LLM instance.
        reranker: CrossEncoder reranker model.
        k_per_query: Chunks to retrieve per sub-question (default 15).
        top_n_per_query: Chunks to keep after reranking per sub-question (default 5).
        max_unique_chunks: Max total unique chunks after dedup (default 20).
        track_tokens_prefix: Optional prefix string for token tracking labels.
                             When set, passes "{prefix}_decompose" and
                             "{prefix}_generate" to sub-functions.

    Returns:
        Dict with report, sub_questions, chunks, and retrieval_metadata.
    """
    # Step 1: Decompose prompt into sub-questions
    decompose_label = f"{track_tokens_prefix}_decompose" if track_tokens_prefix else None
    sub_questions = decompose_query(prompt, llm, track_tokens=decompose_label)

    # Step 2: Multi-retrieval with deduplication
    merged_docs, retrieval_metadata = retrieve_for_report(
        sub_questions, vector_db, reranker,
        k_per_query=k_per_query,
        top_n_per_query=top_n_per_query,
        max_unique_chunks=max_unique_chunks,
    )

    # Step 3: Generate structured report
    generate_label = f"{track_tokens_prefix}_generate" if track_tokens_prefix else None
    report = generate_report(prompt, merged_docs, llm, track_tokens=generate_label)

    # Build chunk data for output
    chunks = []
    for i, doc in enumerate(merged_docs, 1):
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

    return {
        "report": report,
        "sub_questions": sub_questions,
        "chunks": chunks,
        "retrieval_metadata": retrieval_metadata,
    }
