"""Full preprocessing pipeline with tracking, resumability, and concurrency.

Pipeline stages:
fetched → cleaned → llm_extracted → vlm_processed → embedded → stored

Features:
- ThreadPoolExecutor for concurrent LLM/VLM/embedding calls
- tqdm progress bars
- Crash-safe: saves each email immediately
- Resume from where it left off
- Deduplicates Qdrant on re-run
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

from config.settings import config
from src.embedding.embedder import EmailChunker, EmailEmbedder
from src.models import ProcessedEmail
from src.preprocessing.body_cleaner import EmailBodyCleaner
from src.preprocessing.document_extractor import DocumentExtractor
from src.preprocessing.llm_processor import LLMProcessor
from src.preprocessing.vlm_processor import VLMProcessor
from src.storage.vector_store import EmailVectorStore
from src.tracking.state import PipelineStage, PipelineStateManager

# Number of concurrent workers (matches OLLAMA_NUM_PARALLEL)
WORKERS = 4


def _load_raw_emails(limit: int = 0) -> list[dict]:
    """Load raw emails from data/raw_emails/."""
    raw_dir = "data/raw_emails"
    if not os.path.exists(raw_dir):
        return []

    emails = []
    for filename in sorted(os.listdir(raw_dir)):
        if filename.endswith(".json"):
            with open(os.path.join(raw_dir, filename)) as f:
                email = json.load(f)
                email["date"] = datetime.fromisoformat(email["date"])
                emails.append(email)
            if limit > 0 and len(emails) >= limit:
                break

    return emails


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

    def _normalize_attachments(self, attachments) -> list:
        """Convert attachment dicts to EmailAttachment objects."""
        from src.models import EmailAttachment
        result = []
        for att in attachments:
            if isinstance(att, dict):
                # Load from saved path if available, otherwise skip binary
                saved_path = att.get("saved_path", "")
                data = b""
                if saved_path and os.path.exists(saved_path):
                    with open(saved_path, "rb") as f:
                        data = f.read()
                result.append(EmailAttachment(
                    filename=att.get("filename", "unknown"),
                    mime_type=att.get("mime_type", "application/octet-stream"),
                    data=data,
                    size=att.get("size", 0),
                ))
            else:
                result.append(att)
        return result

    def process_single_email(self, email: dict) -> ProcessedEmail:
        """Run the full preprocessing pipeline on a single email."""
        mid = email["message_id"]
        attachments = self._normalize_attachments(email.get("attachments", []))

        # ── Stage 1: Body cleaning ──
        cleaned = self.body_cleaner.clean(email.get("raw_body", ""))
        self.state.set_stage(mid, PipelineStage.CLEANED)

        # ── Stage 2: Attachment processing ──
        attachment_info = []
        attachment_descriptions = []
        attachment_page_descriptions = []
        attachment_contents = []
        attachment_skipped = []

        for att in attachments:
            # Handle both dict (from JSON) and EmailAttachment objects
            if isinstance(att, dict):
                att_filename = att.get("filename", "")
                att_mime = att.get("mime_type", "")
                att_size = att.get("size", 0)
                # Load binary from disk if saved
                att_data = att.get("data", b"")
                if not att_data:
                    att_path = f"data/attachments/{mid}/{att_filename}"
                    if os.path.exists(att_path):
                        with open(att_path, "rb") as f:
                            att_data = f.read()
            else:
                att_filename = att.filename
                att_mime = att.mime_type
                att_size = att.size
                att_data = att.data

            doc = self.doc_extractor.extract(att_filename, att_mime, att_data)

            if doc is not None:
                attachment_contents.append(f"[{att_filename}]: {doc.text[:5000]}")
                attachment_info.append(f"Document {att_filename} ({doc.metadata.get('type', 'unknown')}): {doc.text[:1000]}")

                if doc.images:
                    page_descs = self.vlm.process_document_images(
                        doc.images, att_filename, is_scanned=doc.is_scanned
                    )
                    attachment_page_descriptions.extend(page_descs)
                    attachment_info.extend(page_descs)

                continue

            if att_mime.startswith("image/"):
                if att_size < 500:
                    attachment_skipped.append(att_filename)
                    continue

                category = self.vlm.classify_image(att_data, att_mime)

                if category in self.vlm.skip_categories:
                    attachment_skipped.append(att_filename)
                    continue

                desc = self.vlm.describe_image(att_data, att_mime, att_filename)
                attachment_descriptions.append(f"[{att_filename} ({category})]: {desc}")
                attachment_info.append(f"Image {att_filename} ({category}): {desc}")
                continue

            attachment_skipped.append(att_filename)
            attachment_info.append(f"{att_filename}: unsupported type ({att_mime})")

        self.state.set_stage(mid, PipelineStage.VLM_PROCESSED)

        # ── Stage 3: LLM extraction ──
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

        # ── Build ProcessedEmail ──
        result = ProcessedEmail(
            message_id=mid,
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

        # Save IMMEDIATELY
        self._save_processed_email(result)
        return result

    def _save_processed_email(self, email: ProcessedEmail):
        """Save a single processed email to disk."""
        os.makedirs("data/processed", exist_ok=True)
        filepath = f"data/processed/{email.message_id}.json"
        with open(filepath, "w") as f:
            f.write(email.model_dump_json(indent=2))

    def embed_and_store_single(self, email: ProcessedEmail):
        """Chunk, embed, and store a single processed email."""
        mid = email.message_id

        # Deduplicate
        self.store.delete_by_message_id(mid)

        # Build searchable text
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

        chunks = self.chunker.chunk_email(email_dict)

        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_batch(texts)
            self.store.upsert_chunks(chunks, embeddings)
            email.chunks = [c["text"] for c in chunks]
            email.embedding_ids = [c["id"] for c in chunks]

        self.state.set_stage(mid, PipelineStage.EMBEDDED)
        self.state.set_chunks_created(mid, len(chunks))
        self.state.set_stage(mid, PipelineStage.STORED)

    def _preprocess_one(self, email: dict) -> tuple[str, str | None]:
        """Preprocess a single email. Returns (message_id, error_or_None)."""
        mid = email["message_id"]
        try:
            self.process_single_email(email)
            return (mid, None)
        except Exception as e:
            self.state.set_error(mid, str(e), PipelineStage.LLM_EXTRACTED)
            return (mid, str(e))

    def _embed_one(self, email: ProcessedEmail) -> tuple[str, str | None]:
        """Embed a single email. Returns (message_id, error_or_None)."""
        try:
            self.embed_and_store_single(email)
            return (email.message_id, None)
        except Exception as e:
            self.state.set_error(email.message_id, str(e), PipelineStage.EMBEDDED)
            return (email.message_id, str(e))

    # ── Independent entry points ──

    def run_preprocess(self, emails: list[dict], limit: int = 0) -> list[ProcessedEmail]:
        """Preprocess emails concurrently using ThreadPoolExecutor."""
        if limit > 0:
            emails = emails[:limit]

        # Register
        subjects = {e["message_id"]: e.get("subject", "") for e in emails}
        senders = {e["message_id"]: e.get("sender", "") for e in emails}
        self.state.register_emails([e["message_id"] for e in emails], subjects=subjects, senders=senders)

        # Filter to those needing preprocessing
        needs_processing = self.state.get_emails_needing_processing(PipelineStage.LLM_EXTRACTED)
        to_process = [e for e in emails if e["message_id"] in needs_processing]

        # Also detect stale "unknown" emails from failed LLM calls — retry them
        stale = []
        proc_dir = "data/processed"
        if os.path.exists(proc_dir):
            for fname in os.listdir(proc_dir):
                if not fname.endswith(".json"):
                    continue
                filepath = os.path.join(proc_dir, fname)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    if data.get("category") == "unknown" and "Error" in data.get("summary", ""):
                        mid = data.get("message_id", fname.replace(".json", ""))
                        stale.append(mid)
                        os.remove(filepath)
                        # Reset stage so it gets picked up for reprocessing
                        if mid in self.state._state.get("emails", {}):
                            self.state._state["emails"][mid]["stage"] = None
                            self.state._state["emails"][mid]["error"] = None
                except (json.JSONDecodeError, KeyError):
                    pass

        if stale:
            self.state.save()
            print(f"Found {len(stale)} stale unknown emails — will retry")
            stale_emails = [e for e in emails if e["message_id"] in stale]
            to_process.extend(stale_emails)

        if not to_process:
            print("All emails already preprocessed!")
            return []

        print(f"Preprocessing {len(to_process)} emails with {WORKERS} workers...")
        errors = 0

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(self._preprocess_one, email): email for email in to_process}

            with tqdm(total=len(to_process), desc="Preprocessing", unit="email") as pbar:
                for future in as_completed(futures):
                    mid, error = future.result()
                    email = futures[future]
                    subject = email.get("subject", "")[:50]

                    if error:
                        tqdm.write(f"  ✗ {subject}: {error}")
                        errors += 1
                    else:
                        # Load the processed email to show stats
                        filepath = f"data/processed/{mid}.json"
                        if os.path.exists(filepath):
                            with open(filepath) as f:
                                data = json.load(f)
                            tqdm.write(f"  ✓ {subject} | {data.get('category', '')} | {len(data.get('key_information', []))} facts")
                        else:
                            tqdm.write(f"  ✓ {subject}")

                    pbar.update(1)

        if self.state.is_complete():
            self.state.mark_complete()

        print(f"\nDone: {len(to_process) - errors} preprocessed, {errors} errors")
        return []

    def run_embed(self, limit: int = 0) -> int:
        """Embed preprocessed emails concurrently using ThreadPoolExecutor."""
        needs_embedding = self.state.get_emails_needing_processing(PipelineStage.STORED)

        to_embed = []
        for mid in needs_embedding:
            filepath = f"data/processed/{mid}.json"
            if os.path.exists(filepath):
                with open(filepath) as f:
                    data = json.load(f)
                    data["date"] = datetime.fromisoformat(data["date"])
                    to_embed.append(ProcessedEmail(**data))

        if limit > 0:
            to_embed = to_embed[:limit]

        if not to_embed:
            print("All preprocessed emails already embedded!")
            return 0

        print(f"Embedding {len(to_embed)} emails with {WORKERS} workers...")
        errors = 0

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(self._embed_one, email): email for email in to_embed}

            with tqdm(total=len(to_embed), desc="Embedding", unit="email") as pbar:
                for future in as_completed(futures):
                    mid, error = future.result()
                    email = futures[future]

                    if error:
                        tqdm.write(f"  ✗ {email.subject[:50]}: {error}")
                        errors += 1
                    else:
                        tqdm.write(f"  ✓ {email.subject[:50]} | {len(email.chunks)} chunks")

                    pbar.update(1)

        if self.state.is_complete():
            self.state.mark_complete()

        print(f"\nDone: {len(to_embed) - errors} embedded, {errors} errors")
        return len(to_embed) - errors

    def run(self, emails: list[dict]) -> list[ProcessedEmail]:
        """Run both phases: preprocess + embed."""
        self.run_preprocess(emails)
        self.run_embed()

        all_processed = []
        for mid in self.state.get_emails_at_stage(PipelineStage.STORED):
            filepath = f"data/processed/{mid}.json"
            if os.path.exists(filepath):
                with open(filepath) as f:
                    data = json.load(f)
                    data["date"] = datetime.fromisoformat(data["date"])
                    all_processed.append(ProcessedEmail(**data))
        return all_processed


# ── CLI entry points ──

def main():
    """CLI: email-preprocess [--limit N]"""
    limit = 0
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    emails = _load_raw_emails()
    if not emails:
        print("No raw emails found. Run email-ingest first.")
        return

    if limit > 0:
        print(f"Limiting to {limit} emails")

    state = PipelineStateManager()
    if state.status == "paused" or state.total_emails > 0:
        stored = state.stages.get("stored", {}).get("completed", 0)
        print(f"Resuming: {stored}/{state.total_emails} already stored")

    pipeline = PreprocessingPipeline(state_manager=state)
    pipeline.run_preprocess(emails, limit=limit)


def main_embed():
    """CLI: email-embed [--limit N]"""
    limit = 0
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    state = PipelineStateManager()
    if state.total_emails == 0:
        print("No pipeline state. Run email-ingest + email-preprocess first.")
        return

    stored = state.stages.get("stored", {}).get("completed", 0)
    print(f"State: {stored}/{state.total_emails} stored")

    pipeline = PreprocessingPipeline(state_manager=state)
    pipeline.run_embed(limit=limit)


if __name__ == "__main__":
    main()
