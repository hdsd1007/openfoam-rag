
import os
import json
import argparse
from pathlib import Path

# Parsers
from src.parsers.markerParser import extract_high_fidelity_math
from src.parsers.pymupdfParser import extract_with_layout
from src.parsers.doclingParser import extract_with_docling
from src.vectorstore.build_db import build_vector_db


# Chunker
from src.utils.chunking import get_adaptive_chunks

def run_parser(pdf_path: str, parser_type: str):

    if parser_type == "marker":
        return extract_high_fidelity_math(pdf_path)

    elif parser_type == "docling":
        return extract_with_docling(pdf_path)

    elif parser_type == "pymupdf":
        md_pages, _ = extract_with_layout(pdf_path)
        return md_pages

    else:
        raise ValueError(f"Unknown parser type: {parser_type}")


def main():

    import argparse
    import json
    from pathlib import Path
    import sys

    # ensure src imports work in Colab / terminal
    sys.path.append(".")

    from src.utils.stats import generate_chunk_stats, sample_chunks_for_audit

    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", required=True, choices=["marker", "docling", "pymupdf"])
    parser.add_argument("--pdf", default=None, help="Optional single PDF path")
    parser.add_argument("--out", default="all_chunks.json")
    parser.add_argument("--audit", action="store_true", help="Run chunk statistics + sampling")
    parser.add_argument("--embed", action="store_true", help="Build vector database")

    args = parser.parse_args()

    data_folder = Path("data")
    all_processed_chunks = []

    # Case 1️⃣: Single PDF
    if args.pdf:
        pdf_files = [Path(args.pdf)]
    else:
        # Case 2️⃣: All PDFs in data/
        pdf_files = list(data_folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDFs found.")

    for pdf_path in pdf_files:
        print(f"\n🔍 Processing: {pdf_path.name}")

        full_md = run_parser(str(pdf_path), args.parser)

        chunks = get_adaptive_chunks(
            full_md=full_md,
            filename=pdf_path.name,
            parser_type=args.parser,
        )

        print(f"   → {len(chunks)} chunks")
        all_processed_chunks.extend(chunks)

    print(f"\n✅ Total chunks across all PDFs: {len(all_processed_chunks)}")

    # 🔎 Run audit before saving (optional)
    if args.audit:
        print("\n📊 Running Chunk Audit...\n")
        generate_chunk_stats(all_processed_chunks)
        sample_chunks_for_audit(all_processed_chunks, sample_size=5)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_processed_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to {args.out}")
    
    if args.embed:
        build_vector_db(
            chunks=all_processed_chunks,
            parser_name=args.parser,
            base_dir="db"
        )



if __name__ == "__main__":
    main()
    
# IF YOU WANT TO PARSE ALL PDF IN DATA FOLDER 
# !python main.py --parser pymupdf

# IF YOU WANT TO PARSE, GET STATS and EMBED
# !python main.py --parser pymupdf --audit --embed

