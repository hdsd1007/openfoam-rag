import argparse
from src.vectorstore.load_db import load_vector_db
from src.llm.load_generator_llm import load_generator_llm
from src.rag.pipeline_e2e import ask_openfoam


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True)
    parser.add_argument("--parser", required=True,
                        choices=["docling", "marker", "pymupdf"])
    args = parser.parse_args()

    vector_db = load_vector_db(args.parser)
    llm = load_generator_llm()

    response = ask_openfoam(args.q, vector_db, llm)

    print(response)


if __name__ == "__main__":
    main()


# TO MAKE USE OF ASK.PY FIRST EMBED EVERYTHING
# THEN MAKE THE CALL SPECIFYING THE PARSER TYPE
# !python ask.py --parser marker --q "Explain fvSchemes"
