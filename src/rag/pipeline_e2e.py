from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def ask_openfoam(query, vector_db, llm, return_context=False):

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])

    template = """
You are an OpenFOAM documentation assistant.

You must answer using ONLY the provided documentation context.

STRICT RULES:

1. Use only information explicitly present in the context.

2. If the answer is not fully supported by the context, respond exactly:
"This information is not available in the provided documentation."

3. Do not infer, assume, or reconstruct missing information.

4. Reproduce equations exactly as written in the context (including LaTeX syntax).

5. Reproduce code blocks exactly as written.

6. Do not add explanations beyond what is supported by the context.

7. Do not invent section names, page numbers, or metadata.

CITATION RULES:

A. Every factual statement must include an inline citation in the format [n].

B. Each citation number must correspond to one retrieved context chunk.

C. Use the metadata exactly as provided.

D. If metadata does not contain page numbers or section names, do not fabricate them.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:

1. Provide a structured but concise answer.

2. Only include sections that are directly supported by the context.

3. Avoid repetition.

4. Avoid filler language.

5. If multiple sources support a statement, use multiple citations: [1][2].

REFERENCES:
List the references exactly as provided in the context metadata.
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)
    
    if return_context:
        return response, context_text

    return response
