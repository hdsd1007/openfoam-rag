import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def format_context_for_judge(docs):
    """
    Format documents with chunk numbers matching what the generator sees.
    This allows the judge to verify citation correctness.
    """
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        
        # Build metadata string
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
            
        metadata_str = " | ".join(meta_parts) if meta_parts else "No metadata"
        
        # Format chunk exactly as generator sees it
        chunk_text = f"[{i}] ({metadata_str})\n{doc.page_content}\n"
        context_parts.append(chunk_text)
    
    return "\n" + "="*80 + "\n".join(context_parts)

def judge_answer(question, answer, context, llm):
    
    formatted_context = format_context_for_judge(context)

    template = """
You are a RAG system evaluator for OpenFOAM technical documentation.

Evaluate the answer using ONLY the provided context. Do not use external knowledge or assume missing information. Ignore writing style—focus solely on factual accuracy.

SCORING (1-5 scale):
1. Groundedness: Is every claim supported by the context? (1=mostly unsupported, 5=fully supported)
2. Technical Accuracy: Are facts correct per the context? (1=incorrect, 5=fully accurate)
3. Citation Correctness: Are all [n] citations valid and traceable? (1=fabricated, 5=all valid)
4. Completeness: Does it answer the question using available context? (1=doesn't answer, 5=fully answers)

Return ONLY valid JSON (no markdown, no code fences, no explanations):
{{
"groundedness": int,
"technical_accuracy": int,
"citation_correctness": int,
"completeness": int,
"overall_score": float,
"reasoning": "brief justification"
}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    raw_output = chain.invoke({
        "context": formatted_context,
        "question": question,
        "answer": answer
    })
    
        # Clean potential markdown code fences
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output[7:]  # Remove ```json
    if cleaned_output.startswith("```"):
        cleaned_output = cleaned_output[3:]  # Remove ```
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]  # Remove trailing ```
    cleaned_output = cleaned_output.strip()

    try:
        result = json.loads(cleaned_output)
        
        # Validate structure
        required_fields = ["groundedness", "technical_accuracy", "citation_correctness", 
                          "completeness", "overall_score", "reasoning"]
        
        if not all(field in result for field in required_fields):
            return {
                "error": "Missing required fields in judge output",
                "raw_output": raw_output,
                "parsed_partial": result
            }
        
        return result
        
    except json.JSONDecodeError as e:
        return {
            "error": "Invalid JSON from judge",
            "json_error": str(e),
            "raw_output": raw_output
        }

    # try:
    #     return json.loads(raw_output)
    # except:
    #     return {
    #         "error": "Invalid JSON from judge",
    #         "raw_output": raw_output
    #     }
