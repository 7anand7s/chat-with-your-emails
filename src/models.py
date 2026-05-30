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

    # Cleaned body (flagged, not stripped)
    cleaned_body: str = ""  # Primary text only (high importance)
    full_body: str = ""  # Complete text, nothing removed
    body_sections: list[dict] = []  # [{text, role, importance, reason}]
    noise_ratio: float = 0.0

    attachments: list[EmailAttachment] = []

    # LLM-extracted structured data
    summary: str = ""
    category: str = ""
    subcategory: str = ""
    entities: dict = {}  # people, companies, products, locations, dates, monetary, account_numbers, etc.
    action_items: list[dict] = []  # [{task, assignee, deadline, priority, status}]
    key_information: list[str] = []
    questions_asked: list[str] = []
    decisions_made: list[str] = []
    deadlines_mentioned: list[str] = []
    financial_info: dict = {}  # {is_financial, amounts, currency, transaction_type, account_info}
    dates_and_times: dict = {}  # {mentioned_dates, mentioned_times, is_deadline, is_appointment, is_recurring}
    sentiment: str = ""
    tone: str = ""
    topics: list[str] = []
    requires_response: bool = False
    is_important: bool = False
    is_thread_starter: bool = False
    is_automated: bool = False
    is_promotional: bool = False
    is_financial: bool = False
    is_legal: bool = False
    is_transactional: bool = False
    relationship: str = ""
    email_type: str = ""
    context_for_future_queries: str = ""

    # Attachment processing results
    attachment_descriptions: list[str] = []  # VLM descriptions of images
    attachment_page_descriptions: list[str] = []  # VLM descriptions of doc pages
    attachment_contents: list[str] = []  # Extracted text from documents
    attachment_skipped: list[str] = []  # Files that were skipped (logos, etc.)

    # Links
    links: list[str] = []

    # Embedding data
    chunks: list[str] = []
    embedding_ids: list[str] = []
