"""LLM-based email preprocessing using Ollama."""

import json
import ollama
from config.settings import config

EXTRACTION_PROMPT = """You are an email analysis assistant. Analyze the following email and extract structured information.

Email:
From: {sender}
To: {to}
Subject: {subject}
Date: {date}
Body:
{body}

Respond with ONLY valid JSON (no markdown, no code blocks) in this exact format:
{{
  "summary": "A concise 1-2 sentence summary of the email",
  "category": "one of: personal, work, newsletter, notification, promotion, transactional, social, spam",
  "entities": ["list of people, companies, products, or organizations mentioned"],
  "action_items": ["list of action items, tasks, or things that need a response"],
  "sentiment": "positive, neutral, or negative",
  "topics": ["list of key topics or themes"],
  "is_important": true or false
}}"""


class LLMProcessor:
    def __init__(self):
        self.model = config.models.text_llm
        self.client = ollama.Client(host=config.ollama.base_url)

    def process_email(self, email: dict) -> dict:
        """Use LLM to extract structured data from an email."""
        prompt = EXTRACTION_PROMPT.format(
            sender=email.get("sender", ""),
            to=", ".join(email.get("to", [])),
            subject=email.get("subject", ""),
            date=email.get("date", ""),
            body=email.get("raw_body", "")[:3000],  # Truncate very long emails
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
            return json.loads(content)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            return {
                "summary": f"Error processing email: {e}",
                "category": "unknown",
                "entities": [],
                "action_items": [],
                "sentiment": "neutral",
                "topics": [],
                "is_important": False,
            }
