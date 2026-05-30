"""Full preprocessing pipeline with tracking and resumability.

Pipeline stages:
fetched → cleaned → llm_extracted → vlm_processed → embedded → stored

Features:
- Persists state after every email — survives crashes/kills
- Resume from where it left off
- Deduplicates Qdrant chunks on re-run
- Real-time progress display
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
from src.tracking.display import ProgressDisplay
from src.tracking.state import PipelineStage, PipelineStateManager


console = Console()
display = ProgressDisplay(console)


class PreprocessingPipeline:
    def __init__(self, state_manager: PipelineStateManager = None):
        self.body_cleaner = EmailBodyCleaner()
        self.doc_extractor = DocumentExtractor()
        self.llm = LLMProcessor()
        self.vlm = VLMProcessor()
        self.chunker = EmailChunker(config.chunk_size, config.chunk_overlap)
        self.embedder = EmailEmbedder()
        self.store = EmailVectorStore()
        self.state = state_manager or PipelineStateManager()

    def process_single_email(self, email: dict, index: int = 0, total: int = 0) -> ProcessedEmail:
        """Run the full preprocessing pipeline on a single email.

        Tracks progress through each stage. Saves immediately on completion.
        """
        mid = email["message_id"]
        subject = email.get("subject", "")[:50]
        sender = email.get("sender", "")

        # Update display
        if total > 0:
            display.show_current(subject, sender, "cleaned", index, total)

        # ═══════════════════════════════════════════════
        # STAGE 1: Body cleaning (flag, never delete)
        # ═══════════════════════════════════════════════
        cleaned = self.body_cleaner.clean(email.get("raw_body", ""))
        self.state.set_stage(mid, PipelineStage.CLEANED)

        # ═══════════════════════════════════════════════
        # STAGE 2: Attachment processing
        # ═══════════════════════════════════════════════
        if total > 0:
            display.show_current(subject, sender, "vlm_processed", index, total)

        attachment_info = []
        attachment_descriptions = []
        attachment_page_descriptions = []
        attachment_contents = []
        attachment_skipped = []

        for att in attachments:
            doc = self.doc_extractor.extract(att.filename, att.mime_type, att.data)

            if doc is not None:
                attachment_contents.append(f"[{att.filename}]: {doc.text[:5000]}")
                attachment_info.append(f"Document {att.filename} ({doc.metadata.get('type', 'unknown')}): {doc.text[:1000]}")

                if doc.images:
                    page_descs = self.vlm.process_document_images(
                        doc.images, att.filename, is_scanned=doc.is_scanned
                    )
                    attachment_page_descriptions.extend(page_descs)
                    attachment_info.extend(page_descs)

                if doc.is_scanned and not doc.images:
                    attachment_info.append(f"[{att.filename}]: Scanned document — install pdf2image for VLM page analysis")

                continue

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

            attachment_skipped.append(att.filename)
            attachment_info.append(f"{att.filename}: unsupported type ({att.mime_type})")

        self.state.set_stage(mid, PipelineStage.VLM_PROCESSED)

        # ═══════════════════════════════════════════════
        # STAGE 3: LLM extraction (maximum detail)
        # ═══════════════════════════════════════════════
        if total > 0:
            display.show_current(subject, sender, "llm_extracted", index, total)

        llm_result = self.llm.process_email(
            email=email,
            cleaned_body=cleaned.primary_text,
            full_body=cleaned.full_text,
            noise_ratio=cleaned.noise_ratio,
            section_flags=[s.to_dict() for s in cleaned.sections],
            attachment_info=attachment_info,
            links=cleaned.links,
        )
        self.state.set_stage(mid, PipelineStage.LLM_EXTRACTED)

        # ═══════════════════════════════════════════════
        # STAGE 4: Build processed email
        # ═══════════════════════════════════════════════
        result = ProcessedEmail(
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

        # Save processed email IMMEDIATELY — don't wait for batch
        self._save_processed_email(result)

        return result

    def _save_processed_email(self, email: ProcessedEmail):
        """Save a single processed email to disk immediately."""
        os.makedirs("data/processed", exist_ok=True)
        filepath = f"data/processed/{email.message_id}.json"
        with open(filepath, "w") as f:
            f.write(email.model_dump_json(indent=2))

    def embed_and_store_single(self, email: ProcessedEmail, index: int = 0, total: int = 0):
        """Chunk, embed, and store a single processed email.

        Deduplicates: deletes existing chunks for this message_id before upserting.
        """
        mid = email.message_id
        subject = email.subject[:50]
        sender = email.sender

        if total > 0:
            display.show_current(subject, sender, "embedded", index, total)

        # Deduplicate: remove old chunks for this email from Qdrant
        self.store.delete_by_message_id(mid)

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

        self.state.set_stage(mid, PipelineStage.EMBEDDED)
        self.state.set_chunks_created(mid, len(chunks))
        self.state.set_stage(mid, PipelineStage.STORED)

    def run(self, emails: list[dict]) -> list[ProcessedEmail]:
        """Run the complete pipeline with tracking and resumability.

        - Skips emails already at 'stored' stage
        - Saves state after every email
        - Handles KeyboardInterrupt gracefully (saves state, marks paused)
        - Deduplicates Qdrant on re-run
        """
        total = len(emails)

        # Register all emails in state
        subjects = {e["message_id"]: e.get("subject", "") for e in emails}
        senders = {e["message_id"]: e.get("sender", "") for e in emails}
        self.state.register_emails(
            [e["message_id"] for e in emails],
            subjects=subjects,
            senders=senders,
        )

        # Phase 1: Process emails (skip already extracted)
        needs_processing = self.state.get_emails_needing_processing(PipelineStage.LLM_EXTRACTED)
        emails_to_process = [e for e in emails if e["message_id"] in needs_processing]

        if not emails_to_process:
            console.print("[green]All emails already processed — skipping Phase 1[/green]")
        else:
            console.print(f"\n[bold]Phase 1: Processing {len(emails_to_process)} emails...[/bold]")
            processed = []

            for i, email in enumerate(emails_to_process, 1):
                mid = email["message_id"]
                try:
                    result = self.process_single_email(email, index=i, total=len(emails_to_process))
                    processed.append(result)
                except KeyboardInterrupt:
                    self.state.set_error(mid, "Interrupted by user", PipelineStage.LLM_EXTRACTED)
                    console.print(f"\n[yellow]Paused at email {i}/{len(emails_to_process)}. State saved. Run again to resume.[/yellow]")
                    self.state.set_status("paused")
                    return processed
                except Exception as e:
                    self.state.set_error(mid, str(e), PipelineStage.LLM_EXTRACTED)
                    console.print(f"[red]Error processing '{mid}': {e}[/red]")

        # Phase 2: Embed and store (skip already stored)
        needs_embedding = self.state.get_emails_needing_processing(PipelineStage.STORED)

        # Load processed emails from disk for embedding
        processed_for_embedding = []
        for mid in needs_embedding:
            filepath = f"data/processed/{mid}.json"
            if os.path.exists(filepath):
                with open(filepath) as f:
                    data = json.load(f)
                    data["date"] = datetime.fromisoformat(data["date"])
                    processed_for_embedding.append(ProcessedEmail(**data))

        if not processed_for_embedding:
            console.print("[green]All emails already embedded — skipping Phase 2[/green]")
        else:
            console.print(f"\n[bold]Phase 2: Embedding {len(processed_for_embedding)} emails...[/bold]")

            for i, email in enumerate(processed_for_embedding, 1):
                try:
                    self.embed_and_store_single(email, index=i, total=len(processed_for_embedding))
                except KeyboardInterrupt:
                    self.state.set_error(email.message_id, "Interrupted by user", PipelineStage.EMBEDDED)
                    console.print(f"\n[yellow]Paused at email {i}/{len(processed_for_embedding)}. State saved. Run again to resume.[/yellow]")
                    self.state.set_status("paused")
                    break
                except Exception as e:
                    self.state.set_error(email.message_id, str(e), PipelineStage.EMBEDDED)
                    console.print(f"[red]Error embedding '{email.message_id}': {e}[/red]")

        # Final stats
        if self.state.is_complete():
            self.state.mark_complete()

        display.show_overview(self.state)

        # Return all processed emails
        all_processed = []
        for mid in self.state.get_emails_at_stage(PipelineStage.STORED):
            filepath = f"data/processed/{mid}.json"
            if os.path.exists(filepath):
                with open(filepath) as f:
                    data = json.load(f)
                    data["date"] = datetime.fromisoformat(data["date"])
                    all_processed.append(ProcessedEmail(**data))

        return all_processed


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

    state = PipelineStateManager()

    # Show resume info if this is a resumed run
    if state.status == "paused" or state.total_emails > 0:
        display.show_resume_info(state)

    pipeline = PreprocessingPipeline(state_manager=state)
    pipeline.run(emails)


if __name__ == "__main__":
    main()
