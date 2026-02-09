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

1. Do NOT mention section numbers unless explicitly shown in the context.

2. Do NOT say “the documentation describes…” — explain the concept directly.

3. Do NOT include a references section.

4. Use inline citations only in the format [n].

5. Every technical claim must be supported by context.

6. If information is missing, respond exactly:
"This information is not available in the provided documentation."

7. Do not repeat information.

CONTEXT:
{context}

QUESTION:
{question}

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
