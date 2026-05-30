"""VLM-based processing using Ollama.

Handles:
1. Image attachments: classify (skip noise) → describe meaningful ones
2. Document page images: OCR and describe content (scanned docs, complex layouts)
"""

import ollama
from config.settings import config

# Classify image attachments
CLASSIFY_PROMPT = """Classify this image into exactly ONE category. Reply with ONLY the category name, nothing else.

Categories:
- LOGO: company/brand logo, app icon
- SIGNATURE: email signature graphic, handwritten signature
- TRACKING_PIXEL: tiny invisible tracking image, web beacon
- AVATAR: profile picture, user avatar
- DECORATIVE: divider, banner, decorative element, background pattern
- SCREENSHOT: application screenshot, website screenshot, error message
- DOCUMENT: scanned document, receipt, invoice, form, letter
- PHOTO: real-world photograph
- CHART: graph, chart, data visualization
- DIAGRAM: flowchart, architecture diagram, wireframe, mind map
- CODE: screenshot of code, terminal output, IDE
- OTHER: anything that doesn't fit above"""

# Describe meaningful image attachments
DESCRIBE_IMAGE_PROMPT = """Describe this image in thorough detail. This description will be used for semantic search, so be comprehensive.

Include:
- What type of content it is (screenshot, photo, document, chart, etc.)
- ALL visible text — transcribe it exactly as written
- Key information, data, facts, numbers, dates, amounts shown
- Context clues (what app/website/document/email it relates to)
- Visual structure, layout, colors if relevant to understanding the content
- Any actions, notifications, or UI elements visible

Be specific and factual. Do not describe what you can't see."""

# Describe document pages (converted from PDF/DOCX to images)
DESCRIBE_DOCUMENT_PAGE_PROMPT = """This is page {page_num} of a document converted to an image. Extract ALL text and information from this page.

Include:
- All text content, transcribed as accurately as possible
- Tables: preserve the structure with rows and columns
- Headers, footers, page numbers
- Any charts, diagrams, or images — describe them
- Form fields and their values
- Signatures, stamps, seals
- Any numbers, dates, amounts, reference numbers

Be thorough — this is the primary way we extract document content. Do not summarize, transcribe."""

# Describe scanned documents
DESCRIBE_SCANNED_PROMPT = """This appears to be a scanned document page. Extract ALL visible text and information.

Focus on:
- Transcribing every word exactly as written
- Preserving the document structure (paragraphs, lists, sections)
- Noting any handwritten text vs printed text
- Tables and their data
- Any stamps, seals, signatures, or handwritten annotations
- Numbers, dates, amounts, account numbers, reference codes
- Sender/recipient information if visible

Be extremely thorough. This scanned document may contain important financial, legal, or personal information."""


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
        """Generate detailed description of a meaningful image attachment."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": DESCRIBE_IMAGE_PROMPT,
                    "images": [{"data": image_data, "mime_type": mime_type}],
                }],
                options={"temperature": 0.1},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error describing image {filename}: {e}"

    def describe_document_page(self, image_data: bytes, page_num: int, filename: str, is_scanned: bool = False) -> str:
        """Describe a document page converted to image."""
        prompt = DESCRIBE_SCANNED_PROMPT if is_scanned else DESCRIBE_DOCUMENT_PAGE_PROMPT.format(page_num=page_num)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [{"data": image_data, "mime_type": "image/png"}],
                }],
                options={"temperature": 0.1},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"Error processing page {page_num} of {filename}: {e}"

    def process_attachments(self, attachments: list) -> tuple[list[str], list[str]]:
        """Process image attachments: classify then describe meaningful ones.

        Returns:
            (descriptions, skipped_filenames)
        """
        image_types = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}
        descriptions = []
        skipped = []
        images_processed = 0

        for att in attachments:
            if att.mime_type not in image_types:
                continue

            # Skip tiny images (likely tracking pixels) without VLM call
            if att.size < 500:
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

    def process_document_images(self, page_images: list[bytes], filename: str, is_scanned: bool = False) -> list[str]:
        """Process document pages converted to images.

        Returns list of page descriptions.
        """
        descriptions = []
        for i, image_data in enumerate(page_images[:10]):  # Cap at 10 pages
            desc = self.describe_document_page(image_data, i + 1, filename, is_scanned)
            descriptions.append(f"[{filename} - Page {i+1}]: {desc}")
        return descriptions
