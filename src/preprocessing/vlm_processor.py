"""VLM-based image attachment processing using Ollama.

Handles classification (skip logos/signatures) and detailed description
of meaningful images.
"""

import ollama
from config.settings import config

# First pass: classify the image to decide if it's worth processing
CLASSIFY_PROMPT = """Classify this image into exactly ONE category. Reply with ONLY the category name, nothing else.

Categories:
- LOGO: company/brand logo, app icon
- SIGNATURE: email signature graphic, handwritten signature
- TRACKING_PIXEL: tiny invisible tracking image, web beacon
- AVATAR: profile picture, user avatar
- DECORATIVE: divider, banner, decorative element, background pattern
- SCREENSHOT: application screenshot, website screenshot, error message
- DOCUMENT: scanned document, receipt, invoice, form
- PHOTO: real-world photograph
- CHART: graph, chart, diagram, data visualization
- DIAGRAM: flowchart, architecture diagram, wireframe
- CODE: screenshot of code, terminal output
- OTHER: anything that doesn't fit above"""

# Second pass: detailed description for meaningful images
DESCRIBE_PROMPT = """Describe this image in thorough detail. This description will be used for semantic search, so be comprehensive.

Include:
- What type of content it is (screenshot, photo, document, etc.)
- All visible text (transcribe it exactly)
- Key information, data, or facts shown
- Context clues (what app/website/document it relates to)
- Any numbers, dates, amounts, or identifiers visible
- Colors, layout, and visual structure if relevant

Be specific and factual. Do not describe what you can't see."""


class VLMProcessor:
    def __init__(self):
        self.model = config.models.vision_llm
        self.client = ollama.Client(host=config.ollama.base_url)
        self.skip_categories = {"LOGO", "SIGNATURE", "TRACKING_PIXEL", "AVATAR", "DECORATIVE"}

    def classify_image(self, image_data: bytes, mime_type: str) -> str:
        """Classify an image to decide if it's worth processing."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": CLASSIFY_PROMPT,
                    "images": [{"data": image_data, "mime_type": mime_type}],
                }],
                options={"temperature": 0.0},
            )
            return response["message"]["content"].strip().upper().split("\n")[0].strip()
        except Exception:
            return "OTHER"

    def describe_image(self, image_data: bytes, mime_type: str, filename: str) -> str:
        """Generate detailed description of a meaningful image."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": DESCRIBE_PROMPT,
                    "images": [{"data": image_data, "mime_type": mime_type}],
                }],
                options={"temperature": 0.1},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error describing image {filename}: {e}"

    def process_attachments(self, attachments: list) -> tuple[list[str], list[str]]:
        """Process all image attachments.

        Returns:
            (descriptions, skipped_filenames): list of descriptions and list of skipped files
        """
        image_types = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp", "image/svg+xml"}
        descriptions = []
        skipped = []
        images_processed = 0

        for att in attachments:
            if att.mime_type not in image_types:
                continue

            if att.size < 500:
                # Likely a tracking pixel — skip without VLM call
                skipped.append(att.filename)
                continue

            if images_processed >= config.max_images_per_email:
                skipped.append(att.filename)
                continue

            # Classify first
            category = self.classify_image(att.data, att.mime_type)

            if category in self.skip_categories:
                skipped.append(att.filename)
                continue

            # Meaningful image — describe it
            desc = self.describe_image(att.data, att.mime_type, att.filename)
            descriptions.append(f"[{att.filename} ({category})]: {desc}")
            images_processed += 1

        return descriptions, skipped
