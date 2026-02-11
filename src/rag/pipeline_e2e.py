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
You are an OpenFOAM technical expert helping users understand OpenFOAM concepts and implementation. Provide clear, accurate answers based on the retrieved documentation while maintaining a helpful, professional tone.

## Response Guidelines:

**Structure your answer naturally:**
- Start with a clear explanation of the concept or solution
- Include technical details, formulas, and code examples when relevant
- Cite sources inline using [n] notation (e.g., [1], [2])
- End with a References section listing all cited chunks with their metadata

**Content Requirements:**
- Base your answer strictly on the provided context chunks below
- Each chunk is marked as [n] with associated metadata
- Cite the specific chunk number for each statement: "OpenFOAM uses finite volume method [1]"
- If mathematical equations appear in the context, include them using LaTeX format
- If code snippets or configuration examples exist, include them as code blocks
- When the context doesn't contain sufficient information, clearly state what's missing and what is available

**Writing Style:**
- Use clear, professional language without unnecessary formality
- Explain technical concepts in an accessible way while maintaining accuracy
- Answer directly without preambles like "Based on the context..."
- Don't invent or infer information beyond what's explicitly in the chunks

**Citations Format:**
- Inline: Add [n] immediately after each statement derived from that chunk
- References section at the end: List each cited chunk with its metadata
  Example format:
  **References**
  [1] User Guide | Section 4.2 | Page 45
  [2] Programmer's Guide | Section 3.1 | Page 120

**When Information is Missing:**
If the context completely lacks the requested information, respond: "This information is not available in the provided documentation."

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
