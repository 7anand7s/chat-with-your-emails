"""Full preprocessing pipeline:
1. Clean email body (strip signatures, quotes, disclaimers, tracking)
2. Extract text from document attachments (PDF, DOCX, XLSX, etc.)
3. Classify & describe image attachments via VLM (skip logos/signatures)
4. LLM extracts detailed structured data
5. Chunk, embed, and store in Qdrant
"""

import json
import os
from datetime import datetime
from rich.console import Console
from rich.progress import Progress

from config.settings import config
from src.embedding.embedder import EmailChunker, EmailEmbedder
from src.models import ProcessedEmail
from src.preprocessing.body_cleaner import EmailBodyCleaner
from src.preprocessing.document_extractor import DocumentExtractor
from src.preprocessing.llm_processor import LLMProcessor
from src.preprocessing.vlm_processor import VLMProcessor
from src.storage.vector_store import EmailVectorStore


console = Console()


class PreprocessingPipeline:
    def __init__(self):
        self.body_cleaner = EmailBodyCleaner()
        self.doc_extractor = DocumentExtractor()
        self.llm = LLMProcessor()
        self.vlm = VLMProcessor()
        self.chunker = EmailChunker(config.chunk_size, config.chunk_overlap)
        self.embedder = EmailEmbedder()
        self.store = EmailVectorStore()

    def process_single_email(self, email: dict) -> ProcessedEmail:
        """Run the full preprocessing pipeline on a single email."""
        attachments = email.get("attachments", [])

        # Step 1: Clean the body
        cleaned_body = self.body_cleaner.clean(email.get("raw_body", ""))
        links = self.body_cleaner.extract_links(email.get("raw_body", ""))

        # Step 2: Process attachments
        attachment_info = []        # Combined list for LLM context
        attachment_descriptions = []  # VLM image descriptions
        attachment_contents = []     # Document text extracts
        attachment_skipped = []      # Skipped files

        for att in attachments:
            # Try document extraction first
            doc_text = self.doc_extractor.extract(att.filename, att.mime_type, att.data)
            if doc_text:
                attachment_contents.append(f"[{att.filename}]: {doc_text[:3000]}")
                attachment_info.append(f"Document {att.filename}: {doc_text[:500]}")
                continue

            # Image — let VLM classify and describe
            if att.mime_type.startswith("image/"):
                if att.size < 500:
                    attachment_skipped.append(att.filename)
                    continue

                category = self.vlm.classify_image(att.data, att.mime_type)
                if category in self.vlm.skip_categories:
                    attachment_skipped.append(att.filename)
                    continue

                desc = self.vlm.describe_image(att.data, att.mime_type, att.filename)
                attachment_descriptions.append(f"[{att.filename} ({category})]: {desc}")
                attachment_info.append(f"Image {att.filename}: {desc}")
                continue

            # Unsupported type
            attachment_skipped.append(att.filename)
            attachment_info.append(f"{att.filename}: unsupported file type ({att.mime_type})")

        # Step 3: LLM extraction
        llm_result = self.llm.process_email(
            email=email,
            cleaned_body=cleaned_body,
            attachment_info=attachment_info,
            links=links,
        )

        # Step 4: Build processed email
        return ProcessedEmail(
            message_id=email["message_id"],
            thread_id=email["thread_id"],
            subject=email["subject"],
            sender=email["sender"],
            to=email["to"],
            cc=email.get("cc", []),
            date=email["date"],
            labels=email.get("labels", []),
            raw_body=email["raw_body"],
            cleaned_body=cleaned_body,
            summary=llm_result.get("summary", ""),
            category=llm_result.get("category", ""),
            subcategory=llm_result.get("subcategory", ""),
            entities=llm_result.get("entities", {}),
            action_items=llm_result.get("action_items", []),
            key_information=llm_result.get("key_information", []),
            questions_asked=llm_result.get("questions_asked", []),
            decisions_made=llm_result.get("decisions_made", []),
            deadlines_mentioned=llm_result.get("deadlines_mentioned", []),
            sentiment=llm_result.get("sentiment", ""),
            tone=llm_result.get("tone", ""),
            topics=llm_result.get("topics", []),
            requires_response=llm_result.get("requires_response", False),
            is_important=llm_result.get("is_important", False),
            is_thread_starter=llm_result.get("is_thread_starter", False),
            relationship=llm_result.get("relationship", ""),
            attachment_descriptions=attachment_descriptions,
            attachment_contents=attachment_contents,
            attachment_skipped=attachment_skipped,
            links=links,
        )

    def process_emails(self, emails: list[dict]) -> list[ProcessedEmail]:
        """Process all emails with progress tracking."""
        processed = []

        with Progress() as progress:
            task = progress.add_task("Processing emails...", total=len(emails))

            for email in emails:
                subject = email.get("subject", "")[:40]
                try:
                    result = self.process_single_email(email)
                    processed.append(result)
                    progress.update(task, description=f"Done: {subject}...")
                except Exception as e:
                    console.print(f"[red]Error processing '{subject}': {e}[/red]")
                progress.advance(task)

        return processed

    def embed_and_store(self, processed_emails: list[ProcessedEmail]):
        """Chunk, embed, and store processed emails in Qdrant."""
        with Progress() as progress:
            task = progress.add_task("Embedding & storing...", total=len(processed_emails))

            for email in processed_emails:
                # Build searchable text from all extracted data
                email_dict = {
                    "message_id": email.message_id,
                    "subject": email.subject,
                    "sender": email.sender,
                    "to": email.to,
                    "cc": email.cc,
                    "date": email.date.isoformat(),
                    "category": email.category,
                    "subcategory": email.subcategory,
                    "summary": email.summary,
                    "topics": email.topics,
                    "entities": email.entities,
                    "action_items": [a.get("task", "") for a in email.action_items],
                    "key_information": email.key_information,
                    "questions_asked": email.questions_asked,
                    "decisions_made": email.decisions_made,
                    "deadlines_mentioned": email.deadlines_mentioned,
                    "relationship": email.relationship,
                    "sentiment": email.sentiment,
                    "tone": email.tone,
                    "attachment_descriptions": email.attachment_descriptions,
                    "attachment_contents": email.attachment_contents,
                    "links": email.links,
                    "raw_body": email.cleaned_body,
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

    def run(self, emails: list[dict]) -> list[ProcessedEmail]:
        """Run the complete pipeline."""
        console.print(f"[bold green]Starting preprocessing pipeline for {len(emails)} emails[/bold green]")

        console.print("[bold]Phase 1: Clean → Extract → Analyze...[/bold]")
        processed = self.process_emails(emails)

        console.print("[bold]Phase 2: Chunk → Embed → Store...[/bold]")
        self.embed_and_store(processed)

        # Save processed emails
        os.makedirs("data/processed", exist_ok=True)
        for email in processed:
            filepath = f"data/processed/{email.message_id}.json"
            with open(filepath, "w") as f:
                f.write(email.model_dump_json(indent=2))

        # Stats
        console.print(f"\n[bold green]Pipeline complete![/bold green]")
        console.print(f"  Emails processed: {len(processed)}")
        total_chunks = sum(len(e.chunks) for e in processed)
        console.print(f"  Chunks created: {total_chunks}")
        total_att_desc = sum(len(e.attachment_descriptions) for e in processed)
        total_att_skip = sum(len(e.attachment_skipped) for e in processed)
        console.print(f"  Images described: {total_att_desc}")
        console.print(f"  Images skipped (logos/sig/etc): {total_att_skip}")
        info = self.store.get_collection_info()
        console.print(f"  Vector store: {info['points_count']} chunks in '{config.qdrant.collection}'")

        return processed


def main():
    """CLI entry point for preprocessing."""
    raw_dir = "data/raw_emails"
    if not os.path.exists(raw_dir):
        console.print("[red]No raw emails found. Run email-ingest first.[/red]")
        return

    emails = []
    for filename in sorted(os.listdir(raw_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(raw_dir, filename)) as f:
                email = json.load(f)
                email["date"] = datetime.fromisoformat(email["date"])
                emails.append(email)

    console.print(f"Loaded {len(emails)} emails from {raw_dir}")

    pipeline = PreprocessingPipeline()
    pipeline.run(emails)


if __name__ == "__main__":
    main()
