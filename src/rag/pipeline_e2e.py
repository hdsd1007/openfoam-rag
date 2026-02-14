from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

def format_context_with_metadata(docs):
    """
    Format retrieved documents with chunk numbers and rich metadata.
    Used by both generator and judge to ensure consistency.
    """
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        
        meta_parts = []
        if metadata.get('source'):
            meta_parts.append(f"Source: {metadata['source']}")
        if metadata.get('section') and metadata['section'] != 'N/A':
            meta_parts.append(f"Section: {metadata['section']}")
        if metadata.get('subsection') and metadata['subsection'] != 'N/A':
            meta_parts.append(f"Subsection: {metadata['subsection']}")
        if metadata.get('subsubsection') and metadata['subsubsection'] != 'N/A':
            meta_parts.append(f"Subsubsection: {metadata['subsubsection']}")
        if metadata.get('page'):
            meta_parts.append(f"Page: {metadata['page']}")
        # REMOVED: parser field for judge (not needed)
        # Add evaluation scores to metadata string
        if metadata.get('similarity_score') is not None:
            meta_parts.append(f"Sim Score: {metadata['similarity_score']}")
        if metadata.get('rerank_score') is not None:
            meta_parts.append(f"Rerank Score: {metadata['rerank_score']}")
            
        metadata_str = " | ".join(meta_parts) if meta_parts else "No metadata"
        
        chunk_header = f"[{i}] ({metadata_str})"
        chunk_text = f"{chunk_header}\n{doc.page_content}\n"
        
        context_parts.append(chunk_text)
    
    return "\n" + "="*80 + "\n".join(context_parts)


def ask_openfoam(query, vector_db, llm, return_context=False):

    # Retrieve top-k most relevant chunks
    # retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    # docs = retriever.invoke(query)
    
    # Stage 1 => Retreiving K = 15 chunks
    docs_with_scores = vector_db.similarity_search_with_score(query, k=15)
    initial_docs = []
    for doc, score in docs_with_scores:
        # Save score for evaluation; round for readability
        doc.metadata['similarity_score'] = round(float(score), 4)
        initial_docs.append(doc)
        
    # Stage 2 => Re-Ranking using Cross-Encoder K = 5
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    
    # Prepare query-chunk pairs for the re-ranker
    pairs = [[query, doc.page_content] for doc in initial_docs]
    rerank_scores = reranker.predict(pairs)
    
    # Attach rerank scores to metadata
    for doc, score in zip(initial_docs, rerank_scores):
        doc.metadata['rerank_score'] = round(float(score), 4)

    # Sort by rerank score descending (higher is more relevant) and take top 5
    final_docs = sorted(initial_docs, key=lambda x: x.metadata['rerank_score'], reverse=True)[:5]
    

    template = """
You are an OpenFOAM technical expert assisting users with OpenFOAM concepts, configuration, numerical schemes, and runtime errors.

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
- Do not include meta-commentary (e.g., “Based on the context…”).
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
{question}
"""

    # prompt = ChatPromptTemplate.from_template(template)

    # chain = (
    #     {"context": retriever | format_context_with_metadata, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # response = chain.invoke(query)
    
    # # CHANGED: Simplified response handling (Gemini returns clean strings via StrOutputParser)
    # if return_context:
    #     return response, docs

    # return response
    
    # Updated Logic for Re-Ranker System
    prompt = ChatPromptTemplate.from_template(template)

    # Format the re-ranked top 5 context for the LLM
    formatted_context = format_context_with_metadata(final_docs)

    # Execute the generation chain
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": formatted_context, 
        "question": query
    })
    
    if return_context:
        return response, final_docs

    return response
