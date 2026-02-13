import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.pipeline_e2e import format_context_with_metadata


def judge_answer(question, answer, context, llm):
    
    formatted_context = format_context_with_metadata(context)

    template = """
You are a strict RAG evaluator reviewing an OpenFOAM technical answer.

Evaluate the answer using ONLY the provided context. Do not use external knowledge or assume missing information.

Be skeptical and analytical, like a human reviewer grading a technical exam.

Ignore writing style. Focus only on factual grounding, correctness, and completeness.

SCORING (1-5 scale, use the full range; do not default to 5):

1. Groundedness:
   - 5 = Every factual claim is explicitly supported by the context.
   - 3 = Mostly supported, but contains minor unsupported assumptions or generalizations.
   - 1 = Contains unsupported or invented claims.

2. Technical Accuracy:
   - 5 = Fully accurate per the context.
   - 3 = Minor imprecision or slight misinterpretation.
   - 1 = Incorrect or contradicts the context.

3. Citation Correctness:
   - 5 = Every citation [n] correctly corresponds to the appropriate chunk.
   - 3 = Citations exist but are loosely matched or overused.
   - 1 = Fabricated, mismatched, or invalid citations.

4. Completeness:
   - 5 = Fully answers the question using available context.
   - 3 = Partially answers or misses important aspects present in context.
   - 1 = Fails to answer the question.

5. If the answer appears truncated (incomplete sentence, missing References section, cut-off citation like "["), deduct 2 points from Completeness.

Additional strict rules:
- If any factual sentence lacks citation support, reduce Groundedness.
- If the answer overgeneralizes beyond what is explicitly stated, reduce Groundedness.
- If it repeats one citation excessively for multiple unrelated claims, reduce Citation Correctness.
- If important information from the context is ignored, reduce Completeness.
- Do not give 5 unless the answer is rigorously supported and complete.

Return ONLY valid JSON (no markdown, no explanations):

{{
  "groundedness": int,
  "technical_accuracy": int,
  "citation_correctness": int,
  "completeness": int,
  "overall_score": float,
  "reasoning": "brief but specific justification mentioning concrete issues"
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
