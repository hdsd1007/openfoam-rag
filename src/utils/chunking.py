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
        return [("###", "Section"), ("#####", "Subsection")]
    else:  # marker
        return [
            ("#", "Section"),
            ("##", "Subsection"),
            ("####", "Subsubsection"),
        ]


# -----------------------------
# Page extraction (ONLY for marker)
# -----------------------------
def _extract_page(content: str, parser_type: str) -> str:
    if parser_type == "marker":
        match = re.search(r'id="page-(\d+)', content)
        return match.group(1) if match else "N/A"

    return "Virtual Page"


# -----------------------------
# Main Chunking Function
# -----------------------------
def get_adaptive_chunks(
    full_md,
    filename: str,
    parser_type: str,
    max_tokens: int = 800,
    overlap: int = 100,
) -> List[Dict]:

    enc = tiktoken.get_encoding("cl100k_base")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_get_headers(parser_type)
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,
        chunk_overlap=overlap * 4,
        separators=["\n\n", "\n", " ", ""],
    )

    final_chunks = []

    # --------------------------------------------------
    # CASE 1: PYMUPDF (list of page dicts)
    # --------------------------------------------------
    if parser_type == "pymupdf":

        for page_number, page_dict in enumerate(full_md, start=1):

            page_text = page_dict.get("text", "").strip()
            if not page_text:
                continue

            sections = header_splitter.split_text(page_text)

            for sect in sections:
                content = sect.page_content.strip()
                if not content:
                    continue

                token_count = len(enc.encode(content))

                if token_count <= max_tokens:
                    split_texts = [content]
                else:
                    split_texts = recursive_splitter.split_text(content)

                for chunk_text in split_texts:
                    section_name = sect.metadata.get("Section", "N/A")
                    subsection_name = sect.metadata.get("Subsection", "N/A")
                    final_chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": {
                                **sect.metadata,
                                "source": filename,
                                "parser": parser_type,
                                "page": page_number,  # structural page number
                                "section": section_name,
                                "sub_section":subsection_name,
                                "word_count": len(chunk_text.split()),
                                "token_count": len(enc.encode(chunk_text)),
                            },
                        }
                    )

        return final_chunks

    # --------------------------------------------------
    # CASE 2: MARKER / DOCLING (string input)
    # --------------------------------------------------
    else:

        sections = header_splitter.split_text(full_md)

        for sect in sections:
            content = sect.page_content.strip()
            if not content:
                continue

            token_count = len(enc.encode(content))

            if token_count <= max_tokens:
                split_texts = [content]
            else:
                split_texts = recursive_splitter.split_text(content)

            for chunk_text in split_texts:
                section_name = sect.metadata.get("Section", "N/A")
                subsection_name = sect.metadata.get("Subsection", "N/A")
                final_chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **sect.metadata,
                            "source": filename,
                            "parser": parser_type,
                            "page": _extract_page(chunk_text, parser_type),  # restored
                            "section": section_name,
                            "sub_section": subsection_name,
                            "word_count": len(chunk_text.split()),
                            "token_count": len(enc.encode(chunk_text)),
                        },
                    }
                )

        return final_chunks
