from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_vector_db(chunks, parser_name, base_dir="db"):

    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    print(f"\n🔢 Creating embeddings for parser: {parser_name}")

    # You can upgrade later to L12 for better quality
    embedding_model = "sentence-transformers/all-MiniLM-L12-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Each parser gets its own DB folder
    persist_path = Path(base_dir) / f"{parser_name}_db"
    persist_path.mkdir(parents=True, exist_ok=True)

    vector_db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=str(persist_path)
    )

    vector_db.persist()

    print(f"✅ Vector DB stored at: {persist_path.resolve()}")

    return vector_db
