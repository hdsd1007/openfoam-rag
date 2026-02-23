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
from src.rag.report_pipeline import run_report_pipeline
from src.router.query_router import QueryRouter
from src.router.vision_extractor import VisionExtractor
from sentence_transformers import CrossEncoder

def process_single_question(question, vector_db, llm, reranker, router, args):
    """Process a single question and return results."""
    
    # Route the query
    if args.img:
        normalized_query = router.route((question, args.img))
    else:
        normalized_query = router.route(question)
    
    # Get response and chunks
    response, docs = ask_openfoam(normalized_query, vector_db, llm, reranker, return_context=True)
    
    # Print response
    print("RESPONSE:\n")
    print(response)
    print("\n")
    
    # Verbose mode
    if args.verbose:
        print("RETRIEVED CHUNKS (for debugging):")
        print("="*80)
        for i, doc in enumerate(docs, 1):
            print(f"\n[CHUNK {i}]")
            print(f"Metadata: {doc.metadata}")
            print(f"Content preview: {doc.page_content[:300]}...")
            print("-"*80)
    
    # Prepare output data
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
            "content_preview": doc.page_content[:]
        }
        chunks.append(chunk_data)
    
    output = {
        "question": question,
        "normalized_query": normalized_query if args.img else question,
        "answer": response,
        "chunks": chunks,
        "timestamp": datetime.now().isoformat(),
        "parser": args.parser,
        "has_reranker": any(c["rerank_score"] is not None for c in chunks)
    }
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", help="Single text query")
    parser.add_argument("--questions", help="Path to text file with questions (one per line)")
    parser.add_argument("--parser", required=True, choices=["docling", "marker", "pymupdf"])
    parser.add_argument("--img", help="Optional: Image path to combine with text query")
    parser.add_argument("--report", action="store_true", help="Generate structured report via multi-query pipeline")
    parser.add_argument("--k", type=int, default=15, help="Report mode: chunks to retrieve per sub-question (default: 15)")
    parser.add_argument("--top-n", type=int, default=5, help="Report mode: chunks to keep after reranking per sub-question (default: 5)")
    parser.add_argument("--max-chunks", type=int, default=20, help="Report mode: max unique chunks after dedup (default: 20)")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks")
    parser.add_argument("--save", action="store_true", help="Save output to JSON")
    parser.add_argument("--output-dir", default="query_outputs", help="Directory to save outputs")
    args = parser.parse_args()
    
    # Validate input
    if not args.q and not args.questions:
        print("Error: Must provide either --q or --questions")
        return
    
    if args.q and args.questions:
        print("Error: Cannot use both --q and --questions together")
        return
    
    # Load resources once
    print("Loading vector database and LLM...")
    vector_db = load_vector_db(args.parser)
    max_tokens = 8192 if args.report else 2048
    llm = load_generator_llm(max_tokens=max_tokens)
    reranker = CrossEncoder('BAAI/bge-reranker-base',device='cpu')
    # With this:
    if args.img:
        vision_extractor = VisionExtractor()
        router = QueryRouter(vision_extractor)
    else:
        router = QueryRouter(None)
    
    # Get questions list
    if args.questions:
        # Read from file
        with open(args.questions, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        print(f"\n📋 Processing {len(questions)} questions from {args.questions}\n")
    else:
        # Single question
        questions = [args.q]
    
    # Process questions
    all_outputs = []

    for i, question in enumerate(questions, 1):
        if len(questions) > 1:
            print("="*80)
            print(f"QUESTION {i}/{len(questions)}: {question}")
            print("="*80)

        if args.report:
            # Report mode: multi-query decomposition + synthesis
            print(f"\nGenerating report for: {question}\n")
            result = run_report_pipeline(
                question, vector_db, llm, reranker,
                k_per_query=args.k, top_n_per_query=args.top_n,
                max_unique_chunks=args.max_chunks,
            )

            print("REPORT:\n")
            print(result["report"])
            print("\n")

            if args.verbose:
                print(f"SUB-QUESTIONS ({len(result['sub_questions'])}):")
                for j, sq in enumerate(result["sub_questions"], 1):
                    print(f"  {j}. {sq}")
                print(f"\nRETRIEVAL METADATA:")
                print(f"  Chunks before dedup: {result['retrieval_metadata']['total_before_dedup']}")
                print(f"  Chunks after dedup: {result['retrieval_metadata']['total_after_dedup']}")
                print(f"  Final chunks used: {result['retrieval_metadata']['final_chunks_used']}")
                print("="*80)
                for j, chunk in enumerate(result["chunks"], 1):
                    print(f"\n[CHUNK {j}]")
                    print(f"  Section: {chunk['section']} | Page: {chunk['page']} | Source: {chunk['source']}")
                    print(f"  Rerank: {chunk['rerank_score']} | Sim: {chunk['similarity_score']}")
                    print(f"  Preview: {chunk['content_preview'][:300]}...")
                    print("-"*80)

            output = {
                "question": question,
                "normalized_query": question,
                "answer": result["report"],
                "chunks": result["chunks"],
                "timestamp": datetime.now().isoformat(),
                "parser": args.parser,
                "has_reranker": True,
                "mode": "report",
                "sub_questions": result["sub_questions"],
                "merged_chunks_count": result["retrieval_metadata"]["final_chunks_used"],
                "retrieval_metadata": result["retrieval_metadata"],
            }
        else:
            output = process_single_question(question, vector_db, llm, reranker, router, args)

        all_outputs.append(output)

        if len(questions) > 1:
            print("\n")
    
    # Save outputs
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if len(questions) == 1:
            # Single file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"query_{timestamp}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(all_outputs[0], f, indent=2, ensure_ascii=False)
            
            print(f"✅ Output saved to: {filepath}")
        
        else:
            # Multiple files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, output in enumerate(all_outputs, 1):
                filename = f"query_{timestamp}_{i:02d}.json"
                filepath = output_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {len(all_outputs)} outputs saved to: {output_dir}")
            print(f"   Files: query_{timestamp}_01.json to query_{timestamp}_{len(all_outputs):02d}.json")


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

# Single question with save:
# python ask.py --parser marker --q "Explain fvSchemes" --save

# Multiple questions from file:
# python ask.py --parser marker --questions questions.txt --save

# Multiple questions with verbose:
# python ask.py --parser marker --questions questions.txt --save --verbose

# Report mode (multi-query decomposition + synthesis):
# python ask.py --parser marker --report --q "Write a technical overview of the discretization framework in OpenFOAM" --verbose --save