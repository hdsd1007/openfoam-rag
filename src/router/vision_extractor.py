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
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

    def extract(self, image_input):

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Invalid image input")

        prompt = """
You are analyzing a screenshot of an error encountered while using OpenFOAM.

Extract ONLY:
- The exact OpenFOAM error message/query
- The error/query type (e.g., FOAM FATAL ERROR)
- Missing files or dictionary names if present

Remove timestamps, terminal paths, and UI clutter.
Return clean structured text.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=500
        )

        response = self.processor.batch_decode(
            output[0],
            skip_special_tokens=True
        )

        return self._post_clean(response)

    def _post_clean(self, text):
        # Optional additional filtering
        lines = text.split("\n")
        important = []

        for line in lines:
            if any(k in line for k in [
                "FOAM",
                "ERROR",
                "Error",
                "FATAL",
                "Cannot",
                "Unknown"
            ]):
                important.append(line)

        if important:
            return "\n".join(important)

        return text.strip()
