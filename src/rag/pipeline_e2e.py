from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def ask_openfoam(query, vector_db, llm, return_context=False):

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])

    template = """
You are a Senior OpenFOAM Documentation Specialist with deep expertise in Computational Fluid Dynamics (CFD), Finite Volume Method (FVM), tensor algebra, discretisation schemes, and OpenFOAM C++ implementation.

CRITICAL INSTRUCTIONS:

1. **Use ONLY the provided documentation context**
2. **Cite EVERY factual claim using inline numbered citations [1], [2], etc.**
3. **Write in a clear, technical style similar to academic papers**
4. **Do NOT infer, reconstruct, or guess missing information**
5. **Reproduce equations exactly in LaTeX if present in context**
6. **Preserve code blocks exactly as written**
7. **If information is not in context, explicitly state: "This information is not available in the provided documentation."**

---

CONTEXT:
{context}

---

QUESTION:
{question}

---

RESPONSE FORMAT:

Write a comprehensive technical answer with:

1. **Introduction** - Brief overview of the topic with citations
2. **Technical Explanation** - Detailed explanation with inline citations after each claim
3. **Mathematical Formulation** - Include equations if available in context (with citations)
4. **Implementation Details** - Code examples or configuration details if present (with citations)
5. **References** - Numbered list matching your inline citations

**Citation Style:**
- Place citations immediately after the claim: "The fvSchemes dictionary specifies numerical schemes [1]."
- Multiple sources: "OpenFOAM uses Gauss integration [1][2]."
- Each unique source gets one number only

**Reference Format:**
[1] Section X.Y.Z, Subsection Name, Page N, Guide Name
[2] Section A.B, Page M, Guide Name

Do NOT repeat the same information multiple times.
Keep your answer concise and well-structured.
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
