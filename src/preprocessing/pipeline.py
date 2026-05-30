"""Full preprocessing pipeline: ingest → LLM extract → VLM describe → embed → store."""

import json
import os
from rich.console import Console
from rich.progress import Progress

from config.settings import config
from src.embedding.embedder import EmailChunker, EmailEmbedder
from src.models import ProcessedEmail
from src.preprocessing.llm_processor import LLMProcessor
from src.preprocessing.vlm_processor import VLMProcessor
from src.storage.vector_store import EmailVectorStore


console = Console()


class PreprocessingPipeline:
    def __init__(self):
        self.llm = LLMProcessor()
        self.vlm = VLMProcessor()
        self.chunker = EmailChunker(config.chunk_size, config.chunk_overlap)
        self.embedder = EmailEmbedder()
        self.store = EmailVectorStore()

    def process_emails(self, emails: list[dict]) -> list[ProcessedEmail]:
        """Run the full preprocessing pipeline on a list of emails."""
        processed = []

        with Progress() as progress:
            task = progress.add_task("Processing emails...", total=len(emails))

            for email in emails:
                # Step 1: LLM extraction
                progress.update(task, description=f"LLM: {email['subject'][:40]}...")
                llm_result = self.llm.process_email(email)

                # Step 2: VLM for image attachments
                attachment_descs = []
                if email.get("attachments"):
                    progress.update(task, description=f"VLM: {email['subject'][:40]}...")
                    attachment_descs = self.vlm.process_attachments(email["attachments"])

                # Step 3: Merge results
                processed_email = ProcessedEmail(
                    message_id=email["message_id"],
                    thread_id=email["thread_id"],
                    subject=email["subject"],
                    sender=email["sender"],
                    to=email["to"],
                    cc=email.get("cc", []),
                    date=email["date"],
                    labels=email.get("labels", []),
                    raw_body=email["raw_body"],
                    summary=llm_result.get("summary", ""),
                    category=llm_result.get("category", ""),
                    entities=llm_result.get("entities", []),
                    action_items=llm_result.get("action_items", []),
                    sentiment=llm_result.get("sentiment", ""),
                    topics=llm_result.get("topics", []),
                    is_important=llm_result.get("is_important", False),
                    attachment_descriptions=attachment_descs,
                )
                processed.append(processed_email)
                progress.advance(task)

        return processed

    def embed_and_store(self, processed_emails: list[ProcessedEmail]):
        """Chunk, embed, and store processed emails in Qdrant."""
        with Progress() as progress:
            task = progress.add_task("Embedding & storing...", total=len(processed_emails))

            for email in processed_emails:
                email_dict = {
                    "message_id": email.message_id,
                    "subject": email.subject,
                    "sender": email.sender,
                    "to": email.to,
                    "date": email.date.isoformat(),
                    "category": email.category,
                    "summary": email.summary,
                    "topics": email.topics,
                    "entities": email.entities,
                    "action_items": email.action_items,
                    "attachment_descriptions": email.attachment_descriptions,
                    "raw_body": email.raw_body,
                }

                # Chunk
                chunks = self.chunker.chunk_email(email_dict)

                if chunks:
                    # Embed
                    texts = [c["text"] for c in chunks]
                    embeddings = self.embedder.embed_batch(texts)

                    # Store
                    self.store.upsert_chunks(chunks, embeddings)

                    email.chunks = [c["text"] for c in chunks]
                    email.embedding_ids = [c["id"] for c in chunks]

                progress.advance(task)

    def run(self, emails: list[dict]):
        """Run the complete pipeline."""
        console.print(f"[bold green]Starting preprocessing pipeline for {len(emails)} emails[/bold green]")

        console.print("[bold]Phase 1: LLM & VLM processing...[/bold]")
        processed = self.process_emails(emails)

        console.print("[bold]Phase 2: Embedding & storing...[/bold]")
        self.embed_and_store(processed)

        # Save processed emails
        os.makedirs("data/processed", exist_ok=True)
        for email in processed:
            filepath = f"data/processed/{email.message_id}.json"
            with open(filepath, "w") as f:
                f.write(email.model_dump_json(indent=2))

        console.print(f"[bold green]Done! Processed {len(processed)} emails[/bold green]")
        info = self.store.get_collection_info()
        console.print(f"Vector store: {info['points_count']} chunks in '{config.qdrant.collection}' collection")

        return processed


def main():
    """CLI entry point for preprocessing."""
    # Load raw emails from disk
    raw_dir = "data/raw_emails"
    if not os.path.exists(raw_dir):
        console.print("[red]No raw emails found. Run email-ingest first.[/red]")
        return

    emails = []
    for filename in sorted(os.listdir(raw_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(raw_dir, filename)) as f:
                email = json.load(f)
                # Convert date string back to datetime
                from datetime import datetime
                email["date"] = datetime.fromisoformat(email["date"])
                # Reload attachments if saved separately
                emails.append(email)

    console.print(f"Loaded {len(emails)} emails from {raw_dir}")

    pipeline = PreprocessingPipeline()
    pipeline.run(emails)


if __name__ == "__main__":
    main()
