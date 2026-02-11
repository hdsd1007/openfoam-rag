from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_vector_db(parser_name, base_dir="db"):

    persist_dir = Path(base_dir) / f"{parser_name}_db"

    if not persist_dir.exists():
        raise ValueError(f"Vector DB not found at: {persist_dir}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )

    vector_db = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vector_db
