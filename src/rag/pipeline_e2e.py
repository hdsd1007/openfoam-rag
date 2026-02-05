from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def ask_openfoam(query, vector_db, llm, return_context=False):

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([d.page_content for d in docs])

    template = """
You are a Senior OpenFOAM Documentation Specialist with deep expertise in:

- Computational Fluid Dynamics (CFD)
- Finite Volume Method (FVM)
- Tensor algebra
- Discretisation schemes
- OpenFOAM C++ implementation

STRICT RULES:

1. Use ONLY the provided documentation context.
2. If the answer is not explicitly present, say:

   "The requested information is not available in the provided documentation context."

3. Do NOT infer missing equations.
4. Do NOT reconstruct formulas not shown.
5. Reproduce equations exactly in LaTeX if present.
6. Preserve code blocks exactly as written.
7. Every factual claim MUST include citation in format:

   [Section X | Page Y]

If multiple references apply, cite all.

---

### Context
{context}

---

### Question
{question}

---

### Technical Answer (with mandatory citations)

Structure your answer:

1. Direct technical explanation
2. Mathematical formulation (if available)
3. Implementation detail (if present)
4. Mandatory reference table

---

### Reference Table (Mandatory)

| Section | Page |
|---------|------|
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
