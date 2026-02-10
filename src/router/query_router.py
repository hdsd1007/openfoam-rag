# src/router/query_router.py

from pathlib import Path
from typing import Union
from PIL import Image


class QueryRouter:
    """
    Routes user input to the correct processing pipeline.

    Always returns a normalized TEXT query
    that can be passed directly to your RAG pipeline.
    """

    SUPPORTED_IMAGE_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".bmp", ".webp"
    }

    def __init__(self, vision_extractor):
        """
        vision_extractor: instance of VisionExtractor
        """
        self.vision_extractor = vision_extractor

    def route(self, user_input: Union[str, Image.Image]) -> str:
        """
        Main routing function.

        Returns:
            normalized_query (str)
        """

        # Case 1 — PIL Image
        if isinstance(user_input, Image.Image):
            return self._handle_image(user_input)

        # Case 2 — String input
        if isinstance(user_input, str):
            user_input = user_input.strip()

            # If it's a valid image path
            if self._is_image_path(user_input):
                return self._handle_image(user_input)

            # Otherwise treat as text query
            return self._handle_text(user_input)

        raise ValueError(
            "Unsupported input type. Provide text, image path, or PIL.Image."
        )

    # -----------------------------------------------------
    # Internal Handlers
    # -----------------------------------------------------

    def _handle_text(self, text: str) -> str:
        """
        Process plain text query.
        """
        return text.strip()

    def _handle_image(self, image_input: Union[str, Image.Image]) -> str:
        """
        Process image input using VisionExtractor.
        """
        extracted_text = self.vision_extractor.extract(image_input)

        if not extracted_text or not extracted_text.strip():
            raise ValueError("No readable text extracted from image.")

        return extracted_text.strip()

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------

    def _is_image_path(self, path_str: str) -> bool:
        """
        Check if string is a valid image file path.
        """
        path = Path(path_str)

        return (
            path.exists()
            and path.is_file()
            and path.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
        )
