from datetime import datetime
from pydantic import BaseModel


class EmailAttachment(BaseModel):
    filename: str
    mime_type: str
    data: bytes
    size: int


class ProcessedEmail(BaseModel):
    # Original metadata
    message_id: str
    thread_id: str
    subject: str
    sender: str
    to: list[str]
    cc: list[str] = []
    date: datetime
    labels: list[str] = []
    raw_body: str
    attachments: list[EmailAttachment] = []

    # LLM-extracted structured data
    summary: str = ""
    category: str = ""  # personal, work, newsletter, notification, etc.
    entities: list[str] = []  # people, companies, products mentioned
    action_items: list[str] = []
    sentiment: str = ""  # positive, neutral, negative
    topics: list[str] = []
    is_important: bool = False
    attachment_descriptions: list[str] = []  # VLM descriptions of images

    # Embedding data
    chunks: list[str] = []
    embedding_ids: list[str] = []
