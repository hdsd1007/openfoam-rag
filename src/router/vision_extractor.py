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

        # ----------------------------
        # Build chat message
        # ----------------------------
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        # ----------------------------
        # Apply template (returns string)
        # ----------------------------
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # ----------------------------
        # Prepare multimodal tensors
        # IMPORTANT: images must be list
        # ----------------------------
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )

        inputs = inputs.to(self.device)

        # ----------------------------
        # Generate
        # ----------------------------
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=500
        )

        # ----------------------------
        # Decode (batch)
        # ----------------------------
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        response = generated_texts[0]

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
