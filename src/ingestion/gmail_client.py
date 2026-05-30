"""Gmail API client for fetching emails."""

import base64
import json
import os
from datetime import datetime
from email.utils import parsedate_to_datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config.settings import config
from src.models import EmailAttachment


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
                        "Download it from Google Cloud Console → APIs → Credentials → OAuth 2.0"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, config.gmail.scopes)
                creds = flow.run_local_server(port=0)

            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, "w") as f:
                f.write(creds.to_json())

        return build("gmail", "v1", credentials=creds)

    def fetch_emails(self, max_results: int = 100, query: str = "") -> list[dict]:
        """Fetch email messages from Gmail."""
        messages = []
        page_token = None

        while len(messages) < max_results:
            batch_size = min(100, max_results - len(messages))
            result = self.service.users().messages().list(
                userId="me",
                maxResults=batch_size,
                pageToken=page_token,
                q=query,
            ).execute()

            msg_list = result.get("messages", [])
            if not msg_list:
                break

            for msg_meta in msg_list:
                msg = self.service.users().messages().get(
                    userId="me",
                    id=msg_meta["id"],
                    format="full",
                ).execute()
                messages.append(self._parse_message(msg))
                if len(messages) >= max_results:
                    break

            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return messages

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
            # We'll use the LLM to extract text from HTML later if needed
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

            # Recurse into nested parts
            if part.get("parts"):
                attachments.extend(self._extract_attachments(message_id, part))

        return attachments


def main():
    """CLI entry point for fetching emails."""
    from rich.console import Console
    console = Console()

    console.print("[bold]Fetching emails from Gmail...[/bold]")
    client = GmailClient()
    emails = client.fetch_emails(max_results=10)
    console.print(f"Fetched {len(emails)} emails")

    # Save to disk for preprocessing
    os.makedirs("data/raw_emails", exist_ok=True)
    for email in emails:
        filepath = f"data/raw_emails/{email['message_id']}.json"
        # Convert datetime for JSON serialization
        email_copy = {**email, "date": email["date"].isoformat()}
        # Don't save raw attachment bytes to JSON
        email_copy["attachments"] = [
            {"filename": a.filename, "mime_type": a.mime_type, "size": a.size}
            for a in email["attachments"]
        ]
        with open(filepath, "w") as f:
            json.dump(email_copy, f, indent=2)

    console.print(f"Saved {len(emails)} emails to data/raw_emails/")


if __name__ == "__main__":
    main()
