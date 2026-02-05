
import re
import tiktoken
from typing import List, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# -----------------------------
# Parser-specific header config
# -----------------------------
def _get_headers(parser_type: str):
    if parser_type == "docling":
        return [("##", "Section")]
    elif parser_type == "pymupdf":
        return [("###", "Section"), ("####", "Subsection")]
    else:  # marker
        return [
            ("#", "Section"),
            ("##", "Subsection"),
            ("###", "Subsubsection"),
            ("####", "Subsubsubsection"),
        ]


# -----------------------------
# Parser-specific page extraction
# -----------------------------
def _extract_page(content: str, parser_type: str) -> str:
    if parser_type == "marker":
        match = re.search(r'id="page-(\d+)', content)
        return match.group(1) if match else "N/A"

    if parser_type == "pymupdf":
        match = re.search(r'P-(\d+)', content)
        return match.group(1) if match else "N/A"

    return "Virtual Page"


# -----------------------------
# Main Chunking Function
# -----------------------------
def get_adaptive_chunks(
    full_md: str,
    filename: str,
    parser_type: str,
    max_tokens: int = 800,
    overlap: int = 100,
) -> List[Dict]:

    enc = tiktoken.get_encoding("cl100k_base")

    # 1️⃣ First pass — hierarchical header split
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_get_headers(parser_type)
    )

    sections = header_splitter.split_text(full_md)

    # 2️⃣ Second pass — recursive split for long sections
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,      # approx chars → token safety
        chunk_overlap=overlap * 4,
        separators=["\n\n", "\n", " ", ""],
    )

    final_chunks = []

    for sect in sections:
        content = sect.page_content.strip()
        if not content:
            continue

        token_count = len(enc.encode(content))

        # If section is small enough → keep as is
        if token_count <= max_tokens:
            split_texts = [content]
        else:
            # Recursively split oversized sections
            split_texts = recursive_splitter.split_text(content)

        for chunk_text in split_texts:
            final_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **sect.metadata,
                        "source": filename,
                        "parser": parser_type,
                        "page": _extract_page(chunk_text, parser_type),
                        "word_count": len(chunk_text.split()),
                        "token_count": len(enc.encode(chunk_text)),
                    },
                }
            )

    return final_chunks
