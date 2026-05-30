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

        # Build the full searchable text from ALL structured data
        parts = []
        parts.append(f"Subject: {email.get('subject', '')}")
        parts.append(f"From: {email.get('sender', '')}")
        parts.append(f"To: {', '.join(email.get('to', []))}")
        if email.get('cc'):
            parts.append(f"CC: {', '.join(email['cc'])}")
        parts.append(f"Date: {email.get('date', '')}")
        parts.append(f"Category: {email.get('category', '')}")
        if email.get('subcategory'):
            parts.append(f"Subcategory: {email['subcategory']}")
        if email.get('email_type'):
            parts.append(f"Type: {email['email_type']}")
        if email.get('relationship'):
            parts.append(f"Relationship: {email['relationship']}")
        parts.append(f"Summary: {email.get('summary', '')}")

        # Entities
        entities = email.get('entities', {})
        if isinstance(entities, dict):
            for key, values in entities.items():
                if values:
                    parts.append(f"{key.replace('_', ' ').title()}: {', '.join(str(v) for v in values)}")
        elif isinstance(entities, list):
            parts.append(f"Entities: {', '.join(str(e) for e in entities)}")

        if email.get("topics"):
            parts.append(f"Topics: {', '.join(email['topics'])}")

        # Action items
        action_items = email.get("action_items", [])
        if action_items:
            if isinstance(action_items[0], dict):
                parts.append(f"Action items: {'. '.join(a.get('task', str(a)) for a in action_items)}")
            else:
                parts.append(f"Action items: {'. '.join(str(a) for a in action_items)}")

        if email.get("key_information"):
            parts.append(f"Key information: {'. '.join(email['key_information'])}")
        if email.get("questions_asked"):
            parts.append(f"Questions: {'. '.join(email['questions_asked'])}")
        if email.get("decisions_made"):
            parts.append(f"Decisions: {'. '.join(email['decisions_made'])}")
        if email.get("deadlines_mentioned"):
            parts.append(f"Deadlines: {'. '.join(email['deadlines_mentioned'])}")

        # Financial info
        fin = email.get("financial_info", {})
        if fin.get("is_financial"):
            parts.append(f"Financial: {fin.get('transaction_type', '')} - {', '.join(fin.get('amounts', []))}")

        # Attachment descriptions (images + doc pages + text)
        if email.get("attachment_descriptions"):
            parts.append(f"Image attachments: {'; '.join(email['attachment_descriptions'])}")
        if email.get("attachment_page_descriptions"):
            parts.append(f"Document pages: {'; '.join(email['attachment_page_descriptions'])}")
        if email.get("attachment_contents"):
            parts.append(f"Document contents: {'; '.join(email['attachment_contents'][:3])}")

        if email.get("links"):
            parts.append(f"Links: {', '.join(email['links'][:10])}")

        # Future query context (synonyms, related terms)
        if email.get("context_for_future_queries"):
            parts.append(f"Search context: {email['context_for_future_queries']}")

        # Body
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
