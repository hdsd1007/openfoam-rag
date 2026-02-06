import re
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

def process_openfoam_guide(file_path):
    # 1. Pipeline: Enable LaTeX formula extraction
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_formula_enrichment = True
    
    # 2. Setup Converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 3. Convert and export with Page Break markers
    result = converter.convert(file_path)
    raw_md = result.document.export_to_markdown(
        page_break_placeholder="<!-- PAGE_BREAK -->"
    )

    # 4. Post-processing
    return finalize_formatting(raw_md)

def finalize_formatting(text):
    # --- Step A: Fix Headings (Chapter and Numbered Sections) ---
    def heading_logic(match):
        full_line = match.group(0)
        content = match.group(2).strip()
        
        # Rule 1: Lines starting with "Chapter" or a single digit (e.g., 3. License)
        if content.lower().startswith("chapter") or re.match(r'^\d+\s', content):
            return f"# {content}"
        
        # Rule 2: Numbered sections (e.g., 3.1.2)
        # We count the number of parts separated by dots
        numbering_match = re.match(r'^(\d+(?:\.\d+)+)', content)
        if numbering_match:
            parts = numbering_match.group(1).split('.')
            level = len(parts) # 3.1.2 (3 parts) -> level 3
            return f"{'#' * level} {content}"
            
        return full_line

    # Target lines starting with # (Docling default is usually ##)
    text = re.sub(r'^(#+)\s+(.*)', heading_logic, text, flags=re.MULTILINE)

    # --- Step B: Visible Page Numbering ---
    page_num = 1
    while "<!-- PAGE_BREAK -->" in text:
        text = text.replace("<!-- PAGE_BREAK -->", f"\n\n--- Page {page_num} ---\n\n", 1)
        page_num += 1
    
    return text
