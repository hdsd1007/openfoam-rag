from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def ask_openfoam(query, vector_db, llm, return_context=False):

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])

    template = """
You are an OpenFOAM documentation assistant.

Answer the question using ONLY the provided context.

STRICT RULES:

1. Use only information explicitly present in the context.

2. Do NOT use prior knowledge about OpenFOAM.

3. Do NOT invent section numbers, page numbers, or guide names.

4. Do NOT include a references section.

5. If the context does not contain enough information to answer fully, respond exactly:
"This information is not available in the provided documentation."

6. If citation markers (e.g., [1], [2]) are present in the context, you may reuse them.

7. If no citation markers are present in the context, do NOT generate any citations.

8. Do NOT state “the documentation says” — explain directly.

9. Do not repeat information.
QUESTION:
{question}

CONTEXT:
{context}

Write a direct technical explanation using only supported information.
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
