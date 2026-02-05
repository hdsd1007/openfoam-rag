import fitz
import pathlib
import pymupdf4llm
from tqdm import tqdm

def extract_with_layout(pdf_path: str):
    """
    Extracts markdown while respecting the visual layout of the page.
    """
    filename = pathlib.Path(pdf_path).name

    # We use the layout engine to prevent 'jumbled' text in multi-column areas
    # This is crucial for OpenFOAM tables and math descriptions
    md_text_list = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,      # Returns a list of dicts (one per page)
        write_images=False,    # We don't need the images for RAG
        force_text=True        # Ensures we get text even from complex blocks
    )

    return md_text_list, filename