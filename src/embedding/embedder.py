"""Embedding generation using Ollama and chunking logic."""

import uuid
import ollama
from config.settings import config


class EmailChunker:
    """Splits processed emails into chunks for embedding."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_email(self, email: dict) -> list[dict]:
        """Create chunks from a processed email with metadata."""
        chunks = []

        # Build the full searchable text from structured data
        parts = []
        parts.append(f"Subject: {email.get('subject', '')}")
        parts.append(f"From: {email.get('sender', '')}")
        parts.append(f"To: {', '.join(email.get('to', []))}")
        parts.append(f"Date: {email.get('date', '')}")
        parts.append(f"Category: {email.get('category', '')}")
        parts.append(f"Summary: {email.get('summary', '')}")
        if email.get("topics"):
            parts.append(f"Topics: {', '.join(email['topics'])}")
        if email.get("entities"):
            parts.append(f"Entities: {', '.join(email['entities'])}")
        if email.get("action_items"):
            parts.append(f"Action items: {'. '.join(email['action_items'])}")
        if email.get("attachment_descriptions"):
            parts.append(f"Attachments: {'; '.join(email['attachment_descriptions'])}")
        parts.append(f"Body: {email.get('raw_body', '')}")

        full_text = "\n".join(parts)

        # Split into overlapping chunks
        words = full_text.split()
        if len(words) <= self.chunk_size:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": full_text,
                "message_id": email.get("message_id", ""),
                "subject": email.get("subject", ""),
                "sender": email.get("sender", ""),
                "date": str(email.get("date", "")),
                "category": email.get("category", ""),
                "chunk_index": 0,
            })
        else:
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i : i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "message_id": email.get("message_id", ""),
                    "subject": email.get("subject", ""),
                    "sender": email.get("sender", ""),
                    "date": str(email.get("date", "")),
                    "category": email.get("category", ""),
                    "chunk_index": len(chunks),
                })

        return chunks


class EmailEmbedder:
    """Generate embeddings using Ollama."""

    def __init__(self):
        self.model = config.models.embedding
        self.client = ollama.Client(host=config.ollama.base_url)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self.client.embed(model=self.model, input=text)
        return response["embeddings"][0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        response = self.client.embed(model=self.model, input=texts)
        return response["embeddings"]
