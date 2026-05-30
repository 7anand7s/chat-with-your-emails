"""VLM-based image attachment processing using Ollama."""

import base64
import ollama
from config.settings import config

VISION_PROMPT = """Describe this image in detail. If it's a screenshot, describe what application or website it shows and what information is visible. If it's a document or receipt, extract the key information. If it's a photo, describe the scene and any text visible.

Be thorough and specific - this description will be used for search."""


class VLMProcessor:
    def __init__(self):
        self.model = config.models.vision_llm
        self.client = ollama.Client(host=config.ollama.base_url)

    def describe_image(self, image_data: bytes, mime_type: str, filename: str) -> str:
        """Use VLM to describe an image attachment."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [{"data": image_data, "mime_type": mime_type}],
                }],
                options={"temperature": 0.1},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error describing image {filename}: {e}"

    def process_attachments(self, attachments: list) -> list[str]:
        """Process all image attachments and return descriptions."""
        descriptions = []
        image_types = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}

        for att in attachments:
            if att.mime_type in image_types:
                desc = self.describe_image(att.data, att.mime_type, att.filename)
                descriptions.append(f"[{att.filename}]: {desc}")
            else:
                descriptions.append(f"[{att.filename}]: Non-image attachment ({att.mime_type})")

        return descriptions
