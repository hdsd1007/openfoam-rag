import os
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import text_from_rendered

def extract_high_fidelity_math(pdf_path):
    print(f"Deep-scanning {os.path.basename(pdf_path)} ...")

    # 1) Build config
    config = {
        "output_format": "markdown",   # get markdown
        "redo_inline_math": True,      # try harder on inline math
        "force_ocr": False,            # usually best off for digital PDFs
    }

    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )

    # 2) Convert PDF
    rendered = converter(pdf_path)

    # 3) Extract markdown + images
    md_text, _, _ = text_from_rendered(rendered)
    return md_text