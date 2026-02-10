import argparse
from src.vectorstore.load_db import load_vector_db
from src.llm.load_generator_llm import load_generator_llm
from src.rag.pipeline_e2e import ask_openfoam
from src.router.query_router import QueryRouter
from src.router.vision_extractor import VisionExtractor

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="Text query or path to image containing OpenFOAM error")
    parser.add_argument("--parser", required=True, choices=["docling", "marker", "pymupdf"])
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks and metadata for debugging")
    args = parser.parse_args()

    vector_db = load_vector_db(args.parser)
    llm = load_generator_llm()
    
    # Initialize Multimodal Router
    vision_extractor = VisionExtractor() # You can pass Model and Device Type
    router = QueryRouter(vision_extractor)
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
            print(f"Content preview: {doc.page_content[:200]}...")
            print("-"*80)
    else:
        response = ask_openfoam(normalized_query, vector_db, llm)
        print(response)


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
