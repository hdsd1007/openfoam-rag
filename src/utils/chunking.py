import re
import tiktoken
from typing import List, Dict, Union
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


# --------------------------------------------------
# Header configuration per parser
# --------------------------------------------------
def _get_headers(parser_type: str):
    if parser_type == "docling":
        return [
            ("#", "Section"),
            ("##", "Subsection"),
            ("###", "Subsubsection"),
        ]

    elif parser_type == "pymupdf":
        return [
            ("###", "Section"),
            ("#####", "Subsection"),
        ]

    else:  # marker
        return [
            ("#", "Section"),
            ("##", "Subsection"),
            ("####", "Subsubsection"),
        ]


# --------------------------------------------------
# Page extraction per parser
# --------------------------------------------------
def _extract_page(content: str, parser_type: str) -> str:
    if parser_type == "marker":
        match = re.search(r'id="page-(\d+)', content)
        return match.group(1) if match else "N/A"

    if parser_type == "docling":
        match = re.search(r'--- Page (\d+) ---', content)
        return match.group(1) if match else "N/A"

    return "N/A"


# --------------------------------------------------
# Remove page markers (Docling only)
# --------------------------------------------------
def _clean_docling_page_markers(text: str):
    return re.sub(r'--- Page \d+ ---', '', text)


# --------------------------------------------------
# Main Chunking Function
# --------------------------------------------------
def get_adaptive_chunks(
    full_md: Union[str, List[Dict]],
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

    # ==================================================
    # CASE 1 — PyMuPDF (List of page dictionaries)
    # ==================================================
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

                split_texts = (
                    [content]
                    if token_count <= max_tokens
                    else recursive_splitter.split_text(content)
                )

                for chunk_text in split_texts:
                    final_chunks.append(
                        {
                            "text": chunk_text,
                            "metadata": {
                                **sect.metadata,
                                "source": filename,
                                "parser": parser_type,
                                "page": page_number,
                                "section": sect.metadata.get("Section", "N/A"),
                                "subsection": sect.metadata.get("Subsection", "N/A"),
                                "subsubsection": sect.metadata.get("Subsubsection", "N/A"),
                                "word_count": len(chunk_text.split()),
                                "token_count": len(enc.encode(chunk_text)),
                            },
                        }
                    )

        return final_chunks


    # ==================================================
    # CASE 2 — Docling & Marker (Single Markdown String)
    # ==================================================
    else:

        if parser_type == "docling":
            # Extract page before cleaning
            page_markers = list(re.finditer(r'--- Page (\d+) ---', full_md))
            current_page = "1"
        else:
            current_page = "N/A"

        sections = header_splitter.split_text(full_md)

        for sect in sections:
            content = sect.page_content.strip()
            if not content:
                continue

            # --- Page handling ---
            if parser_type == "docling":
                page_match = re.search(r'--- Page (\d+) ---', content)
                if page_match:
                    current_page = page_match.group(1)

                content = _clean_docling_page_markers(content)

            elif parser_type == "marker":
                current_page = _extract_page(content, "marker")

            token_count = len(enc.encode(content))

            split_texts = (
                [content]
                if token_count <= max_tokens
                else recursive_splitter.split_text(content)
            )

            for chunk_text in split_texts:
                final_chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **sect.metadata,
                            "source": filename,
                            "parser": parser_type,
                            "page": current_page,
                            "section": sect.metadata.get("Section", "N/A"),
                            "subsection": sect.metadata.get("Subsection", "N/A"),
                            "subsubsection": sect.metadata.get("Subsubsection", "N/A"),
                            "word_count": len(chunk_text.split()),
                            "token_count": len(enc.encode(chunk_text)),
                        },
                    }
                )

        return final_chunks
