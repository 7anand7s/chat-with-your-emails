"""Full preprocessing pipeline — nothing is lost, everything is analyzed.

Pipeline stages:
1. Body cleaning: flag sections (never delete), extract links
2. Attachment processing:
   a. Documents → text extraction + page images → VLM for page descriptions
   b. Images → classify (skip noise) → describe meaningful ones
   c. Encrypted PDFs → decrypt with stored passwords
3. LLM extraction: maximum detail structured data
4. Chunk → embed (bge-m3) → store (Qdrant)
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

        # ═══════════════════════════════════════════════
        # STAGE 1: Body cleaning (flag, never delete)
        # ═══════════════════════════════════════════════
        cleaned = self.body_cleaner.clean(email.get("raw_body", ""))

        # ═══════════════════════════════════════════════
        # STAGE 2: Attachment processing
        # ═══════════════════════════════════════════════
        attachment_info = []          # Combined list for LLM context
        attachment_descriptions = []  # VLM image descriptions
        attachment_page_descriptions = []  # VLM document page descriptions
        attachment_contents = []      # Document text extracts
        attachment_skipped = []       # Skipped files

        for att in attachments:
            # Try document extraction
            doc = self.doc_extractor.extract(att.filename, att.mime_type, att.data)

            if doc is not None:
                # Document with text content
                attachment_contents.append(f"[{att.filename}]: {doc.text[:5000]}")
                attachment_info.append(f"Document {att.filename} ({doc.metadata.get('type', 'unknown')}): {doc.text[:1000]}")

                # If document has page images, send to VLM
                if doc.images:
                    page_descs = self.vlm.process_document_images(
                        doc.images, att.filename, is_scanned=doc.is_scanned
                    )
                    attachment_page_descriptions.extend(page_descs)
                    attachment_info.extend(page_descs)

                # If scanned with no images (pdf2image not installed), note it
                if doc.is_scanned and not doc.images:
                    attachment_info.append(f"[{att.filename}]: Scanned document — install pdf2image for VLM page analysis")

                continue

            # Image — classify and describe
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
                attachment_info.append(f"Image {att.filename} ({category}): {desc}")
                continue

            # Unsupported type
            attachment_skipped.append(att.filename)
            attachment_info.append(f"{att.filename}: unsupported type ({att.mime_type})")

        # ═══════════════════════════════════════════════
        # STAGE 3: LLM extraction (maximum detail)
        # ═══════════════════════════════════════════════
        llm_result = self.llm.process_email(
            email=email,
            cleaned_body=cleaned.primary_text,
            full_body=cleaned.full_text,
            noise_ratio=cleaned.noise_ratio,
            section_flags=[s.to_dict() for s in cleaned.sections],
            attachment_info=attachment_info,
            links=cleaned.links,
        )

        # ═══════════════════════════════════════════════
        # STAGE 4: Build processed email
        # ═══════════════════════════════════════════════
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
            cleaned_body=cleaned.primary_text,
            full_body=cleaned.full_text,
            body_sections=[s.to_dict() for s in cleaned.sections],
            noise_ratio=cleaned.noise_ratio,
            summary=llm_result.get("summary", ""),
            category=llm_result.get("category", ""),
            subcategory=llm_result.get("subcategory", ""),
            entities=llm_result.get("entities", {}),
            action_items=llm_result.get("action_items", []),
            key_information=llm_result.get("key_information", []),
            questions_asked=llm_result.get("questions_asked", []),
            decisions_made=llm_result.get("decisions_made", []),
            deadlines_mentioned=llm_result.get("deadlines_mentioned", []),
            financial_info=llm_result.get("financial_info", {}),
            dates_and_times=llm_result.get("dates_and_times", {}),
            sentiment=llm_result.get("sentiment", ""),
            tone=llm_result.get("tone", ""),
            topics=llm_result.get("topics", []),
            requires_response=llm_result.get("requires_response", False),
            is_important=llm_result.get("is_important", False),
            is_thread_starter=llm_result.get("is_thread_starter", False),
            is_automated=llm_result.get("is_automated", False),
            is_promotional=llm_result.get("is_promotional", False),
            is_financial=llm_result.get("is_financial", False),
            is_legal=llm_result.get("is_legal", False),
            is_transactional=llm_result.get("is_transactional", False),
            relationship=llm_result.get("relationship", ""),
            email_type=llm_result.get("email_type", ""),
            context_for_future_queries=llm_result.get("context_for_future_queries", ""),
            attachment_descriptions=attachment_descriptions,
            attachment_page_descriptions=attachment_page_descriptions,
            attachment_contents=attachment_contents,
            attachment_skipped=attachment_skipped,
            links=cleaned.links,
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
                # Build searchable text from ALL extracted data
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
                    "financial_info": email.financial_info,
                    "relationship": email.relationship,
                    "sentiment": email.sentiment,
                    "tone": email.tone,
                    "email_type": email.email_type,
                    "context_for_future_queries": email.context_for_future_queries,
                    "attachment_descriptions": email.attachment_descriptions,
                    "attachment_page_descriptions": email.attachment_page_descriptions,
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
        total_page_desc = sum(len(e.attachment_page_descriptions) for e in processed)
        total_att_skip = sum(len(e.attachment_skipped) for e in processed)
        total_doc_contents = sum(len(e.attachment_contents) for e in processed)
        console.print(f"  Images described (VLM): {total_att_desc}")
        console.print(f"  Document pages described (VLM): {total_page_desc}")
        console.print(f"  Documents text-extracted: {total_doc_contents}")
        console.print(f"  Attachments skipped (noise): {total_att_skip}")
        avg_noise = sum(e.noise_ratio for e in processed) / max(len(processed), 1)
        console.print(f"  Avg noise ratio: {avg_noise:.0%}")
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
