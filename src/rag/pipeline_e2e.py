from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



def format_context_with_metadata(docs):
    """Format retrieved documents with chunk numbers and rich metadata"""
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        
        # Build metadata string from available fields
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
            
        if metadata.get('parser'):
            meta_parts.append(f"Parser: {metadata['parser']}")
        
        metadata_str = " | ".join(meta_parts) if meta_parts else "No metadata available"
        
        # Format chunk with clear numbering and metadata
        chunk_header = f"[{i}] ({metadata_str})"
        chunk_text = f"{chunk_header}\n{doc.page_content}\n"
        
        context_parts.append(chunk_text)
    
    return "\n" + "="*80 + "\n".join(context_parts)


def ask_openfoam(query, vector_db, llm, return_context=False):

    # Retrieve top-k most relevant chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    

    template = """
You are an OpenFOAM technical expert assisting users with OpenFOAM concepts, configuration, and errors.

Provide clear, technically accurate explanations grounded strictly in the retrieved documentation.

CRITICAL REQUIREMENT:
- Every factual statement MUST include at least one inline citation [n].
- If a paragraph contains 3 factual claims, it must contain 3 citations.
- Answers without inline citations are invalid.
- Do NOT place citations only in the References section.

Guidelines:
• Answer naturally and directly, as an expert would.
• Use inline citations in the format [n] for every statement derived from a retrieved chunk.
• Do not invent or assume information not present in the provided context.
• If mathematical equations appear, format them using LaTeX.
• If code snippets or configuration examples are present in the context, include them using proper code blocks.
• If the documentation does not contain enough information to fully answer the question, clearly state what is missing.
• If no relevant information exists, respond:
  "This information is not available in the provided documentation."
• Every factual or technical claim must be traceable to at least one cited chunk.


Avoid:
• Meta-commentary like “Based on the context…”
• Overly rigid formatting
• Fabricating missing details

End your response with a References section listing only the cited chunks in this format:

References
[n] Document Title | Section | Page

---
CONTEXT:
{context}

---
QUESTION:
{question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_context_with_metadata, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)
    
    # Extract text from response (handles different LLM response formats)
    if hasattr(response, 'content'):
        response_text = response.content
    elif isinstance(response, str):
        response_text = response
    else:
        response_text = str(response)
    
    if return_context:
        docs = retriever.invoke(query)
        return response_text, docs

    return response_text
