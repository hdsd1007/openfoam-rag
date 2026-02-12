# src/router/vision_extractor.py

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


class VisionExtractor:
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-500M-Instruct", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

    def extract(self, image_input):

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Invalid image input")

        # CHANGE: Improved prompt - broader scope, handles errors + diagrams + config
        prompt = """Extract all technical content from this OpenFOAM-related image.
        Include:
        - Error messages with exact text
        - Code snippets or configuration entries
        - Diagram labels, equations, or technical annotations
        - File paths, dictionary names, or function calls
        - Any visible OpenFOAM terminology

        Omit:
        - Timestamps, terminal paths, UI decorations
        - Non-technical interface elements

        Return clean, structured text preserving technical accuracy."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )

        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=500
        )

        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        response = generated_texts[0]

        return self._post_clean(response)

    def _post_clean(self, text):
        # CHANGE: Broadened keywords to catch more content types
        lines = text.split("\n")
        important = []

        for line in lines:
            if any(k in line for k in [
                "FOAM", "ERROR", "Error", "FATAL", "Cannot", "Unknown",
                "fv", "dict", "boundary", "solver", "scheme",  # CHANGE: Added OpenFOAM keywords
                "class", "void", "const",  # CHANGE: Added code keywords
                "equation", "velocity", "pressure"  # CHANGE: Added physics keywords
            ]):
                important.append(line)

        if important:
            return "\n".join(important)

        return text.strip()
