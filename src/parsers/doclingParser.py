from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

# 1. Advanced Pipeline Options for Math
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

# CRITICAL: Enable specialized formula detection
pipeline_options.do_formula_detection = True  # This specifically targets those tags
pipeline_options.images_scale = 3.0           # Increase resolution for better symbol recognition

# Use GPU (Required for the Formula model to run efficiently)
pipeline_options.accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA
)

# 2. Setup Converter
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)