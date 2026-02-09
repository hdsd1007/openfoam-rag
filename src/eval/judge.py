import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def judge_answer(question, answer, context, llm):

    template = """
You are an expert evaluator for Retrieval-Augmented Generation systems working with technical CFD documentation.

Evaluate the answer strictly based on the provided context.

Score each criterion from 1 (poor) to 5 (excellent):

1. Groundedness - Is the answer fully supported by the context?
2. Technical Accuracy - Is it technically correct relative to the context?
3. Citation Correctness - Are citations valid and properly formatted?
4. Completeness - Does it fully answer the question?

Return STRICT JSON only:

{{
  "groundedness": int,
  "technical_accuracy": int,
  "citation_correctness": int,
  "completeness": int,
  "overall_score": float,
  "reasoning": "brief explanation"
}}

Context:
{context}

Question:
{question}

Answer:
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
