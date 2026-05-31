"""Gmail API client for fetching emails.

Features:
- Saves each email immediately as it's fetched (crash-safe)
- Skips already-fetched message_ids
- tqdm progress bar during fetch
- Saves attachments to data/attachments/{message_id}/
"""

import base64
import json
import os
import sys
from datetime import datetime
from email.utils import parsedate_to_datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from tqdm import tqdm

from config.settings import config
from src.models import EmailAttachment
from src.tracking.state import PipelineStage, PipelineStateManager


class GmailClient:
    def __init__(self):
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        token_path = config.gmail.token_file
        creds_path = config.gmail.credentials_file

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, config.gmail.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(creds_path):
                    raise FileNotFoundError(
                        f"Gmail credentials file not found: {creds_path}\n"
                        "Download from Google Cloud Console → APIs & Services → Credentials → OAuth 2.0"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, config.gmail.scopes)
                creds = flow.run_local_server(port=0)

            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, "w") as f:
                f.write(creds.to_json())

        return build("gmail", "v1", credentials=creds)

    def _get_message_ids(self, max_results: int = 100, query: str = "") -> list[str]:
        """Get message IDs only (fast, no content)."""
        ids = []
        page_token = None

        while len(ids) < max_results:
            batch_size = min(100, max_results - len(ids))
            result = self.service.users().messages().list(
                userId="me",
                maxResults=batch_size,
                pageToken=page_token,
                q=query,
            ).execute()

            msg_list = result.get("messages", [])
            if not msg_list:
                break

            for m in msg_list:
                ids.append(m["id"])
                if len(ids) >= max_results:
                    break

            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return ids

    def get_message(self, message_id: str) -> dict:
        """Fetch a single message by ID."""
        msg = self.service.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()
        return self._parse_message(msg)

    def _parse_message(self, msg: dict) -> dict:
        """Parse a Gmail message into our format."""
        headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}

        body = self._extract_body(msg["payload"])
        attachments = self._extract_attachments(msg["id"], msg["payload"])

        return {
            "message_id": msg["id"],
            "thread_id": msg["threadId"],
            "subject": headers.get("subject", ""),
            "sender": headers.get("from", ""),
            "to": [addr.strip() for addr in headers.get("to", "").split(",") if addr.strip()],
            "cc": [addr.strip() for addr in headers.get("cc", "").split(",") if addr.strip()],
            "date": parsedate_to_datetime(headers["date"]) if headers.get("date") else datetime.now(),
            "labels": msg.get("labelIds", []),
            "raw_body": body,
            "attachments": attachments,
        }

    def _extract_body(self, payload: dict) -> str:
        """Recursively extract text body from email payload."""
        if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

        if payload.get("mimeType") == "text/html" and payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

        for part in payload.get("parts", []):
            body = self._extract_body(part)
            if body:
                return body

        return ""

    def _extract_attachments(self, message_id: str, payload: dict) -> list[EmailAttachment]:
        """Extract attachments from email payload."""
        attachments = []

        for part in payload.get("parts", []):
            filename = part.get("filename")
            if filename and part.get("body", {}).get("attachmentId"):
                att = self.service.users().messages().attachments().get(
                    userId="me",
                    messageId=message_id,
                    id=part["body"]["attachmentId"],
                ).execute()
                data = base64.urlsafe_b64decode(att["data"])
                attachments.append(EmailAttachment(
                    filename=filename,
                    mime_type=part.get("mimeType", "application/octet-stream"),
                    data=data,
                    size=len(data),
                ))

            if part.get("parts"):
                attachments.extend(self._extract_attachments(message_id, part))

        return attachments


def save_email(email: dict):
    """Save a fetched email to disk."""
    mid = email["message_id"]

    # Save JSON metadata
    os.makedirs("data/raw_emails", exist_ok=True)
    filepath = f"data/raw_emails/{mid}.json"
    email_copy = {**email, "date": email["date"].isoformat()}
    email_copy["attachments"] = [
        {"filename": a.filename, "mime_type": a.mime_type, "size": a.size}
        for a in email["attachments"]
    ]
    with open(filepath, "w") as f:
        json.dump(email_copy, f, indent=2)

    # Save attachment binaries
    if email["attachments"]:
        att_dir = f"data/attachments/{mid}"
        os.makedirs(att_dir, exist_ok=True)
        for att in email["attachments"]:
            att_path = os.path.join(att_dir, att.filename)
            with open(att_path, "wb") as f:
                f.write(att.data)


def main():
    """CLI: email-ingest [--limit N] [--query 'from:bank']

    Fetches emails from Gmail, saves each one immediately.
    Skips already-fetched emails. Can stop/resume anytime.
    """
    # Parse args
    limit = 100
    query = ""
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == "--query" and i + 1 < len(args):
            query = args[i + 1]
            i += 2
        else:
            i += 1

    state = PipelineStateManager()

    # Already-fetched IDs (skip these)
    raw_dir = "data/raw_emails"
    existing = set()
    if os.path.exists(raw_dir):
        existing = {f.replace(".json", "") for f in os.listdir(raw_dir) if f.endswith(".json")}

    print(f"Fetching up to {limit} emails from Gmail...")
    if existing:
        print(f"Already on disk: {len(existing)} (will skip)")
    if query:
        print(f"Query: {query}")

    client = GmailClient()

    # Step 1: Get message IDs (fast)
    print("Getting message list...")
    message_ids = client._get_message_ids(max_results=limit, query=query)
    print(f"Found {len(message_ids)} messages")

    # Filter out already-fetched
    new_ids = [mid for mid in message_ids if mid not in existing]
    skipped = len(message_ids) - len(new_ids)
    if skipped > 0:
        print(f"Skipping {skipped} already on disk")
    if not new_ids:
        print("No new emails to fetch")
        return

    # Step 2: Fetch + save each email immediately (with progress bar)
    fetched = 0
    errors = 0

    for mid in tqdm(new_ids, desc="Fetching", unit="email"):
        try:
            email = client.get_message(mid)
            save_email(email)
            state.register_emails(
                [mid],
                subjects={mid: email.get("subject", "")},
                senders={mid: email.get("sender", "")},
            )
            state.set_stage(mid, PipelineStage.FETCHED)
            fetched += 1
        except KeyboardInterrupt:
            print(f"\nStopped. Fetched {fetched}/{len(new_ids)}. Re-run to resume.")
            state.set_status("paused")
            return
        except Exception as e:
            tqdm.write(f"  Error {mid}: {e}")
            errors += 1

    print(f"\nDone: {fetched} fetched, {errors} errors, {skipped} skipped")


if __name__ == "__main__":
    main()
