import argparse
import json
from pathlib import Path

from src.eval.eval_questions import EVAL_QUESTIONS
from src.eval.judge import judge_answer
from src.vectorstore.load_db import load_vector_db
from src.llm.load_judge_llm import load_judge_llm
from src.llm.load_generator_llm import load_generator_llm
from src.rag.pipeline_e2e import ask_openfoam, format_context_with_metadata



def evaluate_parser(parser_name):

    print(f"\n🔍 Evaluating parser: {parser_name}")

    vector_db = load_vector_db(parser_name)
    generator_llm = load_generator_llm()
    judge_llm = load_judge_llm()

    results = []

    for question in EVAL_QUESTIONS:

        print(f"\n➡ Question: {question}")

        # Get answer + context
        response, context = ask_openfoam(
            question,
            vector_db,
            generator_llm,
            return_context=True
        )

        # Judge
        score = judge_answer(
            question=question,
            answer=response,
            context=context,
            llm=judge_llm
        )
        
        # Check for judging errors
        if "error" in score:
            print(f"Judge error: {score['error']}")
        else:
            print(f"Overall score: {score.get('overall_score', 'N/A')}")

        results.append({
            "question": question,
            "answer": response,
            "context": format_context_with_metadata(context),
            "evaluation": score
        })

    return results


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parser",
        required=True,
        choices=["docling", "marker", "pymupdf"]
    )

    args = parser.parse_args()

    results = evaluate_parser(args.parser)

    output_path = Path("eval_results")
    output_path.mkdir(exist_ok=True)

    file_path = output_path / f"{args.parser}_evaluation.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Evaluation saved to {file_path}")


if __name__ == "__main__":
    main()

# TO RUN EVALUATION USE THE FOLLOWING SCRIPT
# !python -m src.eval.evaluate --parser marker
