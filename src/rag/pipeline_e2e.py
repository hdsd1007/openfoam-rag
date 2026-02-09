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

Write a direct, concise and structured technical answer using only inline citations.

Do not add introductory filler text.
Do not add a references section.
Do not repeat information.

STRICT RULES:

Use only information explicitly present in the context.

If the answer is not supported by the context, respond exactly:
"This information is not available in the provided documentation."

Do not infer or reconstruct missing information.

Reproduce equations exactly as written.

Reproduce code blocks exactly as written.

Do not fabricate section names or page numbers.

Keep the answer concise and technical.

CITATION RULES:

Every factual statement must include inline citations in the format [n].

Use only citation numbers corresponding to the provided context.

Do NOT include a separate references section.

CONTEXT:
{context}

QUESTION:
{question}

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
