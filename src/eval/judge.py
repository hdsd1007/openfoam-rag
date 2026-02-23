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


def _clean_json_output(raw_output):
    """Strip markdown fences and parse JSON from LLM output."""
    cleaned = raw_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def judge_report(question, report, context, reference_checklist, llm):
    """
    Evaluate a generated report against a reference checklist using 6 dimensions.

    Dimensions (1-5 scale each):
        - Groundedness (weight 0.25): Every claim supported by context
        - Technical Accuracy (weight 0.20): Correct per context
        - Citation Correctness (weight 0.15): [n] citations match chunks
        - Coverage (weight 0.20): Did it cover expected_sections?
        - Factual Recall (weight 0.10): Did it include must_include_facts?
        - Structure (weight 0.10): Clear sections, logical flow, no redundancy

    Args:
        question: The report prompt.
        report: The generated report text.
        context: List of LangChain Document objects used as context.
        reference_checklist: Dict with 'expected_sections' and 'must_include_facts'.
        llm: LangChain-compatible LLM for judging.

    Returns:
        Dict with per-dimension scores, overall_score (weighted average),
        sections_found/missing, facts_found/missing, and reasoning.
    """
    formatted_context = format_context_with_metadata(context)

    expected_sections = reference_checklist.get("expected_sections", [])
    must_include_facts = reference_checklist.get("must_include_facts", [])

    sections_list = "\n".join(f"  - {s}" for s in expected_sections)
    facts_list = "\n".join(f"  - {f}" for f in must_include_facts)

    template = """You are a strict evaluator reviewing a generated OpenFOAM technical report.

Evaluate the report using ONLY the provided context and reference checklist. Be skeptical and analytical.

REFERENCE CHECKLIST:
Expected Sections:
{sections_list}

Must-Include Facts:
{facts_list}

SCORING (1-5 scale, use the full range):

1. Groundedness:
   - 5 = Every factual claim is explicitly supported by the context.
   - 3 = Mostly supported, with minor unsupported assumptions.
   - 1 = Contains unsupported or invented claims.

2. Technical Accuracy:
   - 5 = Fully accurate per the context.
   - 3 = Minor imprecision or slight misinterpretation.
   - 1 = Incorrect or contradicts the context.

3. Citation Correctness:
   - 5 = Every citation [n] correctly corresponds to the appropriate chunk.
   - 3 = Citations exist but are loosely matched or overused.
   - 1 = Fabricated, mismatched, or invalid citations.

4. Coverage:
   - 5 = All expected sections are present and adequately addressed.
   - 3 = Most sections present, but some are missing or superficial.
   - 1 = Most expected sections are missing.

5. Factual Recall:
   - 5 = All must-include facts are present in the report.
   - 3 = Most facts present, a few missing.
   - 1 = Most facts are missing.

6. Structure:
   - 5 = Clear ## section headings, logical flow, no redundancy.
   - 3 = Some structure but disorganized or partially redundant.
   - 1 = No clear structure, incoherent flow.

CALIBRATION EXAMPLES:

Strong report (4-5 range): Covers 6/7 expected sections with multi-paragraph depth, includes 7/8
must-include facts, every claim has correct [n] citations, clear ## structure with logical flow.
Scores: Groundedness=5, Accuracy=4, Citations=5, Coverage=4, Factual Recall=4, Structure=5.

Average report (3 range): Covers 4/7 sections but some are one-sentence shallow, includes 4/8 facts,
citations exist but some are loosely matched, has ## headings but sections feel repetitive.
Scores: Groundedness=3, Accuracy=3, Citations=3, Coverage=3, Factual Recall=3, Structure=3.

Weak report (1-2 range): Covers 2/7 sections, misses most facts, fabricated citations or no citations,
wall of text with no headings.
Scores: Groundedness=2, Accuracy=2, Citations=1, Coverage=1, Factual Recall=1, Structure=1.

STRICT RULES:
- A section counts as "found" only if it has substantive content (2+ sentences), not just a heading.
- A fact counts as "found" only if the specific claim is present, not just a vague mention of the topic.
- If the report appears truncated (cut-off sentence, missing References), deduct 1 from Structure.
- Do not give 5 on Coverage unless ALL expected sections are present with real content.
- Do not give 5 on Factual Recall unless ALL must-include facts are present.

Return ONLY valid JSON (no markdown, no explanations):

{{
  "groundedness": int,
  "technical_accuracy": int,
  "citation_correctness": int,
  "coverage": int,
  "factual_recall": int,
  "structure": int,
  "sections_found": ["list of expected sections that ARE present"],
  "sections_missing": ["list of expected sections that are NOT present"],
  "facts_found": ["list of must-include facts that ARE present"],
  "facts_missing": ["list of must-include facts that are NOT present"],
  "reasoning": "brief but specific justification mentioning concrete issues"
}}

CONTEXT:
{context}

REPORT TOPIC:
{question}

REPORT:
{report}"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    raw_output = chain.invoke({
        "context": formatted_context,
        "question": question,
        "report": report,
        "sections_list": sections_list,
        "facts_list": facts_list,
    })

    cleaned_output = _clean_json_output(raw_output)

    try:
        result = json.loads(cleaned_output)

        required_fields = [
            "groundedness", "technical_accuracy", "citation_correctness",
            "coverage", "factual_recall", "structure", "reasoning"
        ]

        if not all(field in result for field in required_fields):
            return {
                "error": "Missing required fields in judge output",
                "raw_output": raw_output,
                "parsed_partial": result
            }

        # Compute overall_score programmatically (weighted average)
        weights = {
            "groundedness": 0.25,
            "technical_accuracy": 0.20,
            "citation_correctness": 0.15,
            "coverage": 0.20,
            "factual_recall": 0.10,
            "structure": 0.10,
        }
        result["overall_score"] = round(
            sum(result[dim] * w for dim, w in weights.items()), 2
        )

        # Ensure list fields exist even if LLM omitted them
        for field in ["sections_found", "sections_missing", "facts_found", "facts_missing"]:
            if field not in result:
                result[field] = []

        return result

    except json.JSONDecodeError as e:
        return {
            "error": "Invalid JSON from judge",
            "json_error": str(e),
            "raw_output": raw_output
        }
