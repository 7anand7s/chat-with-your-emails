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
    cleaned_body: str = ""
    attachments: list[EmailAttachment] = []

    # LLM-extracted structured data
    summary: str = ""
    category: str = ""
    subcategory: str = ""
    entities: dict = {}  # people, companies, products, locations, dates, monetary
    action_items: list[dict] = []  # [{task, assignee, deadline, priority}]
    key_information: list[str] = []
    questions_asked: list[str] = []
    decisions_made: list[str] = []
    deadlines_mentioned: list[str] = []
    sentiment: str = ""
    tone: str = ""
    topics: list[str] = []
    requires_response: bool = False
    is_important: bool = False
    is_thread_starter: bool = False
    relationship: str = ""

    # Attachment processing results
    attachment_descriptions: list[str] = []  # VLM descriptions of images
    attachment_contents: list[str] = []  # Extracted text from documents
    attachment_skipped: list[str] = []  # Files that were skipped (logos, etc.)

    # Links
    links: list[str] = []

    # Embedding data
    chunks: list[str] = []
    embedding_ids: list[str] = []
