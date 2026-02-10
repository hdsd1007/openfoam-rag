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
You are an evaluator for a Retrieval-Augmented Generation (RAG) system using OpenFOAM technical documentation.

You must evaluate the answer strictly and exclusively based on the provided context.

Do NOT use external knowledge.
Do NOT assume missing information.
Ignore writing style and fluency.
Evaluate only factual alignment with the context.

SCORING CRITERIA (1 - 5 scale):

Groundedness

1 = Mostly unsupported or hallucinated

3 = Partially supported

5 = Fully supported by context

Technical Accuracy

1 = Incorrect relative to context

3 = Minor inaccuracies

5 = Fully consistent with context

Citation Correctness

1 = Fabricated or incorrect citations

3 = Minor formatting issues

5 = All citations valid and traceable to context

Completeness

1 = Does not answer question

3 = Partially answers

5 = Fully answers based on context

CRITICAL INSTRUCTION:

1. Return STRICT JSON only.
2. Do not include markdown.
3. Do not include code fences.
4. Do not include explanations outside JSON.

Output format:

{{
"groundedness": int,
"technical_accuracy": int,
"citation_correctness": int,
"completeness": int,
"overall_score": float,
"reasoning": "brief technical justification"
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
