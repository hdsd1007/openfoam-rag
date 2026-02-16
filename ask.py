import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import argparse
import json
from datetime import datetime
from pathlib import Path
from src.vectorstore.load_db import load_vector_db
from src.llm.load_generator_llm import load_generator_llm
from src.rag.pipeline_e2e import ask_openfoam
from src.router.query_router import QueryRouter
from src.router.vision_extractor import VisionExtractor

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="Text query or path to image containing OpenFOAM error")
    parser.add_argument("--parser", required=True, choices=["docling", "marker", "pymupdf"])
    parser.add_argument("--img", help="Optional: Image path to combine with text query")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks and metadata for debugging")
    parser.add_argument("--save", action="store_true", help="Save output to JSON file for dashboard")
    parser.add_argument("--output-dir", default="query_outputs", help="Directory to save outputs")
    args = parser.parse_args()

    vector_db = load_vector_db(args.parser)
    llm = load_generator_llm()
    
    # Initialize Multimodal Router
    vision_extractor = VisionExtractor() # You can pass Model and Device Type
    router = QueryRouter(vision_extractor)
    if args.img:
        # Multimodal: text + image
        normalized_query = router.route((args.q, args.img))
    else:
        # Original: text only or image only
        normalized_query = router.route(args.q)

    # Verbose (Showing Retrieved Chunks)
    if args.verbose:
        response, docs = ask_openfoam(normalized_query, vector_db, llm, return_context=True)
        print("RESPONSE:\n")
        print(response)
        print("\n")
        print("RETRIEVED CHUNKS (for debugging):")
        print("="*80)
        for i, doc in enumerate(docs, 1):
            print(f"\n[CHUNK {i}]")
            print(f"Metadata: {doc.metadata}")
            print(f"Content preview: {doc.page_content[:300]}...")
            print("-"*80)
    else:
        response = ask_openfoam(normalized_query, vector_db, llm)
        print(response)
        
    # Saving to JSON
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Format chunks for JSON
        chunks = []
        for i, doc in enumerate(docs, 1):
            chunk_data = {
                "rank": i,
                "section": doc.metadata.get('section', 'N/A'),
                "subsection": doc.metadata.get('subsection'),
                "page": doc.metadata.get('page', 'N/A'),
                "source": doc.metadata.get('source', 'Unknown'),
                "similarity_score": doc.metadata.get('similarity_score'),
                "rerank_score": doc.metadata.get('rerank_score'),
                "content_preview": doc.page_content[:300]
            }
            chunks.append(chunk_data)
        
        # Create output structure
        output = {
            "question": args.q,
            "normalized_query": normalized_query if args.img else args.q,
            "answer": response,
            "chunks": chunks,
            "timestamp": datetime.now().isoformat(),
            "parser": args.parser,
            "has_reranker": any(c["rerank_score"] is not None for c in chunks)
        }
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Output saved to: {filepath}")


if __name__ == "__main__":
    main()


# TO MAKE USE OF ASK.PY FIRST EMBED EVERYTHING
# THEN MAKE THE CALL SPECIFYING THE PARSER TYPE

# For TEXT only question
# !python ask.py --parser marker --q "Explain fvSchemes"
# For TEXT question with verbose output (shows retrieved chunks)
# python ask.py --parser marker --q "Explain fvSchemes" --verbose

# For Error/Query Screenshot or Image
# !python ask.py --parser marker --q error.png
# For Image with verbose output
# python ask.py --parser marker --q error.png --verbose

# For Text + Image:
# python ask.py --parser marker --q "Why am I getting this error?" --img error_screenshot.png
# python ask.py --parser marker --q "Explain this configuration" --img config_snippet.png

# Verbose mode works with all:
# python ask.py --parser marker --q "Fix this error" --img error.png --verbose

# Save with verbose output:
# python ask.py --parser marker --q "Explain fvSchemes" --save --verbose

# Multimodal with save:
# python ask.py --parser marker --q "Why this error?" --img error.png --save

# Custom output directory:
# python ask.py --parser marker --q "Explain fvSchemes" --save --output-dir my_queries
