"""LLM-based email preprocessing using Ollama.

Extracts detailed structured data from email content.
"""

import json
import ollama
from config.settings import config

EXTRACTION_PROMPT = """You are an expert email analysis assistant. Analyze the following email and extract ALL structured information with maximum detail.

Email:
From: {sender}
To: {to}
CC: {cc}
Date: {date}
Subject: {subject}

Cleaned Body:
{body}

Attachment Info:
{attachment_info}

Links found:
{links}

Respond with ONLY valid JSON (no markdown, no code blocks) in this exact format:
{{
  "summary": "Detailed 2-4 sentence summary capturing the full context and purpose of this email",
  "category": "one of: personal, work, newsletter, notification, promotion, transactional, social, support, legal, finance, marketing, system_alert, meeting, project_update",
  "subcategory": "more specific classification within the category (e.g. 'shipping_update' for transactional, 'weekly_digest' for newsletter)",
  "entities": {{
    "people": ["full names of people mentioned"],
    "companies": ["company/organization names"],
    "products": ["product or service names"],
    "locations": ["places, cities, addresses mentioned"],
    "dates": ["specific dates or time references mentioned"],
    "monetary": ["any amounts, prices, costs mentioned"]
  }},
  "action_items": [
    {{
      "task": "description of what needs to be done",
      "assignee": "who should do it (if mentioned)",
      "deadline": "when it's due (if mentioned)",
      "priority": "high/medium/low"
    }}
  ],
  "key_information": ["list of important facts, numbers, decisions, or data points in this email"],
  "questions_asked": ["any questions the sender is asking"],
  "decisions_made": ["any decisions or conclusions reached"],
  "deadlines_mentioned": ["any deadlines or time-sensitive items"],
  "sentiment": "positive/neutral/negative/mixed/urgent",
  "tone": "formal/casual/friendly/urgent/apologetic/congratulatory/concerned",
  "topics": ["detailed list of topics discussed"],
  "requires_response": true_or_false,
  "is_important": true_or_false,
  "is_thread_starter": true_or_false,
  "relationship": "the relationship context (e.g. 'boss to subordinate', 'vendor to customer', 'friend to friend', 'automated notification')"
}}"""


class LLMProcessor:
    def __init__(self):
        self.model = config.models.text_llm
        self.client = ollama.Client(host=config.ollama.base_url)

    def process_email(
        self,
        email: dict,
        cleaned_body: str,
        attachment_info: list[str],
        links: list[str],
    ) -> dict:
        """Use LLM to extract detailed structured data from an email."""
        # Build attachment description
        if attachment_info:
            att_text = "\n".join(f"- {a}" for a in attachment_info)
        else:
            att_text = "No attachments"

        # Build links description
        links_text = "\n".join(f"- {l}" for l in links[:10]) if links else "No links"

        prompt = EXTRACTION_PROMPT.format(
            sender=email.get("sender", ""),
            to=", ".join(email.get("to", [])),
            cc=", ".join(email.get("cc", [])),
            date=email.get("date", ""),
            subject=email.get("subject", ""),
            body=cleaned_body[:4000],
            attachment_info=att_text,
            links=links_text,
        )

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            content = response["message"]["content"].strip()

            # Try to extract JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            result = json.loads(content)

            # Normalize defaults
            result.setdefault("summary", "")
            result.setdefault("category", "unknown")
            result.setdefault("subcategory", "")
            result.setdefault("entities", {})
            result.setdefault("action_items", [])
            result.setdefault("key_information", [])
            result.setdefault("questions_asked", [])
            result.setdefault("decisions_made", [])
            result.setdefault("deadlines_mentioned", [])
            result.setdefault("sentiment", "neutral")
            result.setdefault("tone", "")
            result.setdefault("topics", [])
            result.setdefault("requires_response", False)
            result.setdefault("is_important", False)
            result.setdefault("is_thread_starter", False)
            result.setdefault("relationship", "")

            return result

        except (json.JSONDecodeError, KeyError, Exception) as e:
            return {
                "summary": f"Error processing email: {e}",
                "category": "unknown",
                "subcategory": "",
                "entities": {},
                "action_items": [],
                "key_information": [],
                "questions_asked": [],
                "decisions_made": [],
                "deadlines_mentioned": [],
                "sentiment": "neutral",
                "tone": "",
                "topics": [],
                "requires_response": False,
                "is_important": False,
                "is_thread_starter": False,
                "relationship": "",
            }
