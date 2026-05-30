"""Email body cleaner — NEVER deletes data, only flags sections.

Rule #1: Never throw away any data.
Every section gets tagged with a role and importance score.
The LLM decides what's useful downstream.
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class SectionRole(str, Enum):
    BODY = "body"
    SIGNATURE = "signature"
    QUOTED_REPLY = "quoted_reply"
    DISCLAIMER = "disclaimer"
    TRACKING = "tracking"
    UNSUBSCRIBE = "unsubscribe"
    HEADER = "header"  # Forwarded email headers
    FOOTER = "footer"
    ADVERTISEMENT = "advertisement"
    SIGNATURE_IMAGE = "signature_image"  # <img> in signature area


@dataclass
class FlaggedSection:
    """A section of the email body with flags."""
    text: str
    role: SectionRole
    importance: float  # 0.0 = noise, 1.0 = critical
    reason: str  # Why this flag was assigned

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "role": self.role.value,
            "importance": self.importance,
            "reason": self.reason,
        }


@dataclass
class CleanedBody:
    """Result of body cleaning — nothing is removed, everything is flagged."""
    sections: list[FlaggedSection] = field(default_factory=list)
    primary_text: str = ""  # The main body text (high importance)
    full_text: str = ""  # Complete text with all sections
    links: list[str] = field(default_factory=list)
    noise_ratio: float = 0.0  # How much of the email is noise (0-1)

    def to_dict(self) -> dict:
        return {
            "primary_text": self.primary_text,
            "full_text": self.full_text,
            "sections": [s.to_dict() for s in self.sections],
            "links": self.links,
            "noise_ratio": round(self.noise_ratio, 2),
        }


class EmailBodyCleaner:
    """Flag email body sections by role and importance. Never delete data."""

    # Signature patterns — mark as SIGNATURE with low importance
    SIGNATURE_MARKERS = [
        (r'\n--\s*\n', "Standard signature delimiter"),
        (r'\n_{3,}\s*\n', "Underscore separator"),
        (r'\n-{3,}\s*\n', "Dash separator"),
        (r'\n={3,}\s*\n', "Equal separator"),
    ]

    SIGNATURE_PHRASES = [
        "best regards", "kind regards", "warm regards", "regards",
        "cheers", "thanks,", "thank you,", "thanks!", "thank you!",
        "sent from my", "get outlook for", "sent from iphone",
        "sent from my iphone", "sent from my android",
        "_sent from", "get outlook", "download outlook",
        "yours sincerely", "yours truly", "sincerely",
        "with best wishes", "all the best", "best wishes",
        "confidentiality notice", "this email is intended",
    ]

    # Quoted reply markers
    QUOTE_MARKERS = [
        (r'\nOn .+ wrote:\n', "Gmail-style reply header"),
        (r'\n-{3,} Original Message -{3,}', "Outlook-style original message"),
        (r'\nFrom: .+\nSent: .+\nTo: .+\nSubject:.*\n', "Outlook forwarded header"),
        (r'\n_{3,}\nFrom:', "Underscore-separated forward"),
    ]

    # Disclaimer patterns
    DISCLAIMER_PATTERNS = [
        (r'This email and any files?.*?(?:confidential|privileged|disclaimer)', "Confidentiality disclaimer"),
        (r'(?:CONFIDENTIAL|DISCLAIMER|LEGAL NOTICE):', "Legal notice header"),
        (r'This message contains information.*?confidential', "Confidentiality notice"),
        (r'The information contained in this.*?(?:email|message|communication)', "Information disclaimer"),
        (r'This email is intended only for the', "Intended recipient notice"),
        (r'If you have received this email in error', "Error disclaimer"),
        (r'This communication.*?may contain.*?privileged', "Privileged communication notice"),
    ]

    # Unsubscribe / marketing footer
    UNSUBSCRIBE_PATTERNS = [
        (r'(?:To unsubscribe|Unsubscribe|Manage your preferences|Opt out)', "Unsubscribe link"),
        (r'(?:You are receiving this|This email was sent to)', "Mailing list notice"),
        (r'(?:Update your email preferences|Email preferences)', "Email preferences"),
        (r'(?:View in browser|View this email in your browser)', "Web view link"),
        (r'(?:No longer wish to receive|Stop receiving)', "Opt-out notice"),
        (r'(?:Copyright|©)\s*\d{4}.*?(?:rights reserved|all rights)', "Copyright footer"),
    ]

    # Tracking patterns
    TRACKING_PATTERNS = [
        (r'<img[^>]*width=["\']?1["\']?[^>]*height=["\']?1["\']?[^>]*>', "1x1 tracking pixel"),
        (r'<img[^>]*height=["\']?1["\']?[^>]*width=["\']?1["\']?[^>]*>', "1x1 tracking pixel"),
    ]

    TRACKING_URL_KEYWORDS = {'track', 'pixel', 'beacon', 'open', 'analytics', 'click'}

    # Advertisement patterns
    AD_PATTERNS = [
        (r'(?:Special offer|Limited time|Act now|Don\'t miss|Exclusive deal)', "Promotional language"),
        (r'(?:% off|discount|save \$|free trial|buy now|shop now)', "Sales language"),
        (r'(?:Click here to|Learn more|Get started|Sign up now)', "CTA language"),
    ]

    def clean(self, body: str) -> CleanedBody:
        """Clean email body by flagging sections. Never deletes data."""
        result = CleanedBody()

        if not body:
            result.primary_text = ""
            result.full_text = ""
            return result

        # Convert HTML if needed
        if self._is_html(body):
            body = self._html_to_text(body)

        # Extract all links first
        result.links = self._extract_links(body)

        # Split into lines for analysis
        lines = body.split('\n')
        total_chars = max(len(body), 1)

        # Track which lines are flagged
        line_flags: list[tuple[str, SectionRole, float, str]] = []
        # Each entry: (line_text, role, importance, reason)

        i = 0
        while i < len(lines):
            line = lines[i]
            line_lower = line.strip().lower()

            # Check for tracking pixels in HTML
            tracking_match = self._match_any(line, self.TRACKING_PATTERNS)
            if tracking_match:
                line_flags.append((line, SectionRole.TRACKING, 0.0, tracking_match))
                i += 1
                continue

            # Check for signature delimiters
            sig_match = self._match_any(line, self.SIGNATURE_MARKERS)
            if sig_match:
                # Everything from here to end is likely signature
                remaining = '\n'.join(lines[i:])
                line_flags.append((remaining, SectionRole.SIGNATURE, 0.1, sig_match))
                break

            # Check for signature phrases
            if any(phrase in line_lower for phrase in self.SIGNATURE_PHRASES):
                remaining = '\n'.join(lines[i:])
                line_flags.append((remaining, SectionRole.SIGNATURE, 0.1, f"Signature phrase: {line.strip()[:50]}"))
                break

            # Check for quoted reply headers
            quote_match = self._match_any(line, self.QUOTE_MARKERS)
            if quote_match:
                remaining = '\n'.join(lines[i:])
                line_flags.append((remaining, SectionRole.QUOTED_REPLY, 0.2, quote_match))
                break

            # Check for inline quotes (> lines)
            if line.strip().startswith('>'):
                quoted_block = []
                while i < len(lines) and lines[i].strip().startswith('>'):
                    quoted_block.append(lines[i])
                    i += 1
                line_flags.append(('\n'.join(quoted_block), SectionRole.QUOTED_REPLY, 0.2, "Inline quote (>)"))
                continue

            # Check for disclaimers
            disclaimer_match = self._match_any(line, self.DISCLAIMER_PATTERNS)
            if disclaimer_match:
                remaining = '\n'.join(lines[i:])
                line_flags.append((remaining, SectionRole.DISCLAIMER, 0.05, disclaimer_match))
                break

            # Check for unsubscribe/marketing footer
            unsub_match = self._match_any(line, self.UNSUBSCRIBE_PATTERNS)
            if unsub_match:
                remaining = '\n'.join(lines[i:])
                line_flags.append((remaining, SectionRole.UNSUBSCRIBE, 0.05, unsub_match))
                break

            # Check for advertisements
            ad_match = self._match_any(line, self.AD_PATTERNS)
            if ad_match:
                line_flags.append((line, SectionRole.ADVERTISEMENT, 0.1, ad_match))
                i += 1
                continue

            # Default: body text
            line_flags.append((line, SectionRole.BODY, 1.0, "Main content"))
            i += 1

        # Build sections (group consecutive same-role lines)
        if line_flags:
            current_role = line_flags[0][1]
            current_importance = line_flags[0][2]
            current_reason = line_flags[0][3]
            current_lines = [line_flags[0][0]]

            for text, role, importance, reason in line_flags[1:]:
                if role == current_role and role == SectionRole.BODY:
                    # Merge consecutive body lines
                    current_lines.append(text)
                else:
                    # Flush previous section
                    result.sections.append(FlaggedSection(
                        text='\n'.join(current_lines),
                        role=current_role,
                        importance=current_importance,
                        reason=current_reason,
                    ))
                    current_role = role
                    current_importance = importance
                    current_reason = reason
                    current_lines = [text]

            # Flush last section
            result.sections.append(FlaggedSection(
                text='\n'.join(current_lines),
                role=current_role,
                importance=current_importance,
                reason=current_reason,
            ))

        # Build primary text (only high-importance sections)
        primary_parts = [s.text for s in result.sections if s.importance >= 0.5]
        result.primary_text = '\n'.join(primary_parts).strip()

        # Full text preserves everything
        result.full_text = body

        # Calculate noise ratio
        noise_chars = sum(len(s.text) for s in result.sections if s.importance < 0.3)
        result.noise_ratio = noise_chars / total_chars if total_chars > 0 else 0

        return result

    def _match_any(self, text: str, patterns: list[tuple[str, str]]) -> str | None:
        """Check if text matches any pattern. Returns reason or None."""
        for pattern, reason in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return reason
        return None

    def _is_html(self, text: str) -> bool:
        """Check if text looks like HTML."""
        lower = text.lower()
        return "<html" in lower or "<body" in lower or "<div" in lower or "<table" in lower

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text, preserving structure."""
        text = html
        replacements = [
            (r'<br\s*/?>', '\n'),
            (r'<p[^>]*>', '\n'),
            (r'</p>', '\n'),
            (r'<li[^>]*>', '\n- '),
            (r'<h[1-6][^>]*>', '\n## '),
            (r'</h[1-6]>', '\n'),
            (r'<tr[^>]*>', '\n'),
            (r'<td[^>]*>', ' | '),
            (r'<div[^>]*>', '\n'),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Preserve image alt text
        text = re.sub(
            r'<img[^>]*alt=["\']([^"\']*)["\'][^>]*>',
            r'[Image: \1]',
            text,
            flags=re.IGNORECASE,
        )

        # Strip remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Decode HTML entities
        entities = [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"')]
        for entity, char in entities:
            text = text.replace(entity, char)
        text = re.sub(r'&#\d+;', '', text)

        return text

    def _extract_links(self, body: str) -> list[str]:
        """Extract links, categorizing tracking vs useful."""
        url_pattern = r'https?://[^\s<>"\')\]]+'
        urls = re.findall(url_pattern, body)
        useful = []
        for url in urls:
            url_lower = url.lower()
            is_tracking = any(kw in url_lower for kw in self.TRACKING_URL_KEYWORDS)
            if not is_tracking:
                useful.append(url)
        return list(set(useful))[:30]
