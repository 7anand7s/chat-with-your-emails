"""LLM-based email preprocessing using Ollama.

Extracts maximum detail from email content — never loses context.
"""

import json
import ollama
from config.settings import config

EXTRACTION_PROMPT = """You are an expert email analyst. Your job is to extract EVERY piece of meaningful information from this email. Do NOT summarize away details — capture everything.

=== EMAIL METADATA ===
From: {sender}
To: {to}
CC: {cc}
Date: {date}
Subject: {subject}

=== EMAIL BODY (cleaned, primary content) ===
{body}

=== FULL BODY (including signatures, quotes, disclaimers) ===
{full_body}

=== EMAIL NOISE RATIO ===
{noise_ratio:.0%} of this email was flagged as noise (signatures, quotes, disclaimers, tracking)

=== ATTACHMENT DESCRIPTIONS ===
{attachment_info}

=== LINKS FOUND ===
{links}

=== BODY SECTION FLAGS ===
{section_flags}

Respond with ONLY valid JSON (no markdown, no code blocks). Be extremely detailed:

{{
  "summary": "A comprehensive 3-6 sentence summary that captures the FULL context, purpose, and key details of this email. Include who is writing, why, what action is needed, and any critical details. Do not lose important information.",

  "category": "EXACTLY one of: personal, work, banking_finance, legal, medical, travel, shopping, receipts, job_search, real_estate, insurance, tax, investment, newsletter, digest, promotion, advertisement, social_media, notification, system_alert, support, project_update, meeting, announcement, government, education, charity, subscription, security_alert, compliance, vendor, customer, hr, marketing, sales",

  "subcategory": "A specific subcategory. Examples: 'shipping_update', 'weekly_digest', 'salary_slip', 'credit_card_statement', 'flight_confirmation', 'password_reset', 'invoice', 'contract', 'prescription', 'appointment_reminder', 'job_application', 'interview_invite', 'offer_letter', 'rent_receipt', 'tax_document', 'insurance_claim', 'stock_trade', 'dividend_notice'. Be specific.",

  "entities": {{
    "people": ["full names and roles if mentioned, e.g. 'John Smith (Project Manager)'"],
    "companies": ["all company/organization names"],
    "products": ["product names, service names, model numbers"],
    "locations": ["addresses, cities, countries, venues"],
    "dates": ["all dates mentioned with context, e.g. 'March 15, 2024 (deadline)'"],
    "monetary": ["all amounts with currency and context, e.g. '$1,234.56 (invoice total)'"],
    "account_numbers": ["any account numbers, reference numbers, transaction IDs, confirmation codes"],
    "phone_numbers": ["any phone numbers mentioned"],
    "email_addresses": ["any email addresses mentioned beyond sender/recipient"],
    "urls": ["any important URLs mentioned in the body"]
  }},

  "action_items": [
    {{
      "task": "detailed description of what needs to be done",
      "assignee": "who should do it (if mentioned)",
      "deadline": "when it's due (if mentioned)",
      "priority": "urgent/high/medium/low",
      "status": "pending/in_progress/completed/not_started"
    }}
  ],

  "key_information": ["EVERY important fact, number, decision, data point, or piece of information in this email. Be exhaustive."],

  "questions_asked": ["all questions the sender asks, verbatim if possible"],

  "decisions_made": ["all decisions, conclusions, or agreements reached"],

  "deadlines_mentioned": ["all deadlines with context"],

  "financial_info": {{
    "is_financial": true_or_false,
    "amounts": ["all monetary amounts"],
    "currency": "primary currency",
    "transaction_type": "payment/invoice/refund/subscription/transfer/etc if applicable",
    "account_info": "any account or reference numbers"
  }},

  "dates_and_times": {{
    "mentioned_dates": ["all dates with context"],
    "mentioned_times": ["all times with context"],
    "is_deadline": true_or_false,
    "is_appointment": true_or_false,
    "is_recurring": true_or_false
  }},

  "sentiment": "positive/neutral/negative/mixed/urgent/frustrated/happy/grateful/apologetic",
  "tone": "formal/casual/friendly/urgent/apologetic/congratulatory/concerned/angry/neutral/promotional/automated",
  "topics": ["detailed list of ALL topics discussed"],

  "requires_response": true_or_false,
  "is_important": true_or_false,
  "is_thread_starter": true_or_false,
  "is_automated": true_or_false,
  "is_promotional": true_or_false,
  "is_financial": true_or_false,
  "is_legal": true_or_false,
  "is_transactional": true_or_false,

  "relationship": "describe the relationship context, e.g. 'bank to customer', 'employer to employee', 'vendor to buyer', 'newsletter to subscriber', 'automated system notification', 'friend to friend'",

  "email_type": "direct/conversation/forward/reply/newsletter/notification/alert/digest/marketing/transactional/legal/financial",

  "context_for_future_queries": "Write 2-3 sentences of additional context that would help someone searching for this email in the future. Include synonyms, related terms, and contextual information that isn't explicitly in the email but is implied."
}}"""


class LLMProcessor:
    def __init__(self):
        self.model = config.models.text_llm
        self.client = ollama.Client(host=config.ollama.base_url)

    def process_email(
        self,
        email: dict,
        cleaned_body: str,
        full_body: str,
        noise_ratio: float,
        section_flags: list[dict],
        attachment_info: list[str],
        links: list[str],
    ) -> dict:
        """Use LLM to extract maximum detail from an email."""
        # Build attachment description
        att_text = "\n".join(f"- {a}" for a in attachment_info) if attachment_info else "No attachments"

        # Build links description
        links_text = "\n".join(f"- {l}" for l in links[:15]) if links else "No links"

        # Build section flags summary
        flags_text = "\n".join(
            f"- [{s['role']}] (importance: {s['importance']:.0%}) {s['reason']}"
            for s in section_flags[:20]
        ) if section_flags else "No section analysis available"

        prompt = EXTRACTION_PROMPT.format(
            sender=email.get("sender", ""),
            to=", ".join(email.get("to", [])),
            cc=", ".join(email.get("cc", [])),
            date=email.get("date", ""),
            subject=email.get("subject", ""),
            body=cleaned_body[:5000],
            full_body=full_body[:3000],
            noise_ratio=noise_ratio,
            attachment_info=att_text,
            links=links_text,
            section_flags=flags_text,
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

            # Ensure all fields exist with defaults
            defaults = {
                "summary": "",
                "category": "unknown",
                "subcategory": "",
                "entities": {},
                "action_items": [],
                "key_information": [],
                "questions_asked": [],
                "decisions_made": [],
                "deadlines_mentioned": [],
                "financial_info": {},
                "dates_and_times": {},
                "sentiment": "neutral",
                "tone": "",
                "topics": [],
                "requires_response": False,
                "is_important": False,
                "is_thread_starter": False,
                "is_automated": False,
                "is_promotional": False,
                "is_financial": False,
                "is_legal": False,
                "is_transactional": False,
                "relationship": "",
                "email_type": "",
                "context_for_future_queries": "",
            }
            for key, default in defaults.items():
                result.setdefault(key, default)

            return result

        except json.JSONDecodeError as e:
            # Bad JSON from LLM — retry later
            raise RuntimeError(f"LLM returned invalid JSON: {e}") from e
        except Exception as e:
            # Ollama unreachable, timeout, etc. — retry later
            raise RuntimeError(f"LLM call failed: {e}") from e
