import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def judge_answer(question, answer, context, llm):

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
        "context": context,
        "question": question,
        "answer": answer
    })

    try:
        return json.loads(raw_output)
    except:
        return {
            "error": "Invalid JSON from judge",
            "raw_output": raw_output
        }
