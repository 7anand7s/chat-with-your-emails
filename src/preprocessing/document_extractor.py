"""Extract text content from various document attachment types."""

import csv
import io
import json
import os
import zipfile

from config.settings import config


class DocumentExtractor:
    """Extract text from document attachments. Returns (text, metadata) or None if unsupported."""

    # Text-based file types we can read directly
    TEXT_TYPES = {
        "text/plain", "text/csv", "text/html", "text/xml", "text/markdown",
        "text/x-python", "text/x-java", "text/x-c", "text/x-script",
        "application/json", "application/xml", "application/javascript",
        "application/x-yaml", "application/yaml", "text/yaml",
    }

    TEXT_EXTENSIONS = {
        ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json",
        ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php", ".sh", ".bash", ".zsh",
        ".html", ".htm", ".css", ".scss", ".less",
        ".sql", ".r", ".scala", ".kt", ".swift",
        ".log", ".out", ".err", ".env", ".gitignore",
        ".dockerfile", ".makefile", ".cmake",
    }

    def extract(self, filename: str, mime_type: str, data: bytes) -> str | None:
        """Extract text from an attachment. Returns text or None if unsupported."""
        ext = os.path.splitext(filename.lower())[1]

        # Size check
        if len(data) > config.max_attachment_size:
            return f"[File too large to process: {len(data)} bytes]"

        # Direct text files
        if mime_type in self.TEXT_TYPES or ext in self.TEXT_EXTENSIONS:
            return self._extract_text(data)

        # PDF
        if mime_type == "application/pdf" or ext == ".pdf":
            return self._extract_pdf(data)

        # Office documents
        if ext in (".docx", ".doc"):
            return self._extract_docx(data)
        if ext in (".xlsx", ".xls"):
            return self._extract_xlsx(data)
        if ext in (".pptx", ".ppt"):
            return self._extract_pptx(data)

        # Archives
        if ext in (".zip", ".jar", ".war"):
            return self._extract_zip(data)

        # Email files
        if ext in (".eml", ".msg"):
            return self._extract_email_file(data)

        # vCard
        if ext == ".vcf":
            return self._extract_text(data)

        # Calendar
        if ext == ".ics":
            return self._extract_text(data)

        return None

    def _extract_text(self, data: bytes) -> str:
        """Extract text from a text-based file."""
        try:
            text = data.decode("utf-8", errors="replace")
            # Truncate very long files
            if len(text) > 50000:
                text = text[:50000] + f"\n\n[... truncated, {len(text)} total characters]"
            return text
        except Exception:
            return "[Unable to decode file]"

    def _extract_pdf(self, data: bytes) -> str:
        """Extract text from a PDF file."""
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages.append(f"--- Page {i+1} ---\n{text}")
                    # Also extract tables
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        table_text = "\n".join(
                            " | ".join(str(cell) if cell else "" for cell in row)
                            for row in table
                        )
                        pages.append(f"--- Table {j+1} on Page {i+1} ---\n{table_text}")
                if pages:
                    return "\n\n".join(pages)
                return "[PDF with no extractable text — likely scanned/image-based]"
        except ImportError:
            return "[PDF file — install pdfplumber for text extraction]"
        except Exception as e:
            return f"[PDF extraction error: {e}]"

    def _extract_docx(self, data: bytes) -> str:
        """Extract text from a DOCX file."""
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    parts.append(para.text)
            # Also extract tables
            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    parts.append(" | ".join(cells))
            return "\n".join(parts) if parts else "[Empty document]"
        except ImportError:
            return "[DOCX file — install python-docx for text extraction]"
        except Exception as e:
            return f"[DOCX extraction error: {e}]"

    def _extract_xlsx(self, data: bytes) -> str:
        """Extract text from an XLSX file."""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            parts = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                parts.append(f"=== Sheet: {sheet_name} ===")
                for row in ws.iter_rows(max_row=200, values_only=True):
                    cells = [str(c) if c is not None else "" for c in row]
                    if any(cells):
                        parts.append(" | ".join(cells))
            wb.close()
            return "\n".join(parts) if parts else "[Empty spreadsheet]"
        except ImportError:
            return "[XLSX file — install openpyxl for text extraction]"
        except Exception as e:
            return f"[XLSX extraction error: {e}]"

    def _extract_pptx(self, data: bytes) -> str:
        """Extract text from a PPTX file."""
        try:
            from pptx import Presentation
            prs = Presentation(io.BytesIO(data))
            parts = []
            for i, slide in enumerate(prs.slides, 1):
                parts.append(f"--- Slide {i} ---")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        parts.append(shape.text)
            return "\n".join(parts) if parts else "[Empty presentation]"
        except ImportError:
            return "[PPTX file — install python-pptx for text extraction]"
        except Exception as e:
            return f"[PPTX extraction error: {e}]"

    def _extract_zip(self, data: bytes) -> str:
        """List contents of a ZIP archive."""
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                entries = []
                for info in zf.infolist():
                    entries.append(f"{info.filename} ({info.file_size} bytes)")
                return "ZIP archive contents:\n" + "\n".join(entries)
        except Exception as e:
            return f"[ZIP extraction error: {e}]"

    def _extract_email_file(self, data: bytes) -> str:
        """Extract basic info from .eml files."""
        try:
            from email import message_from_bytes
            msg = message_from_bytes(data)
            parts = [
                f"Subject: {msg.get('subject', '')}",
                f"From: {msg.get('from', '')}",
                f"To: {msg.get('to', '')}",
                f"Date: {msg.get('date', '')}",
            ]
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        parts.append(part.get_payload(decode=True).decode("utf-8", errors="replace"))
                        break
            else:
                parts.append(msg.get_payload(decode=True).decode("utf-8", errors="replace"))
            return "\n".join(parts)
        except Exception as e:
            return f"[EML extraction error: {e}]"
