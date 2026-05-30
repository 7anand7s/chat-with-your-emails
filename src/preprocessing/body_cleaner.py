"""Clean email body content by removing noise: signatures, quotes, disclaimers, tracking."""

import re


class EmailBodyCleaner:
    """Strip noise from email bodies to get clean, useful content."""

    # Common signature delimiters
    SIGNATURE_PATTERNS = [
        r'\n--\s*\n',  # Standard signature delimiter
        r'\n_{3,}\s*\n',  # Underscore separator
        r'\n-{3,}\s*\n',  # Dash separator
        r'\n={3,}\s*\n',  # Equal separator
        r'\nBest regards.*$',
        r'\nCheers.*$',
        r'\nThanks.*$',
        r'\nThank you.*$',
        r'\nSent from my .*$',
        r'\nGet Outlook for .*$',
        r'\n_Sent from_.*$',
    ]

    # Quoted reply headers
    QUOTE_PATTERNS = [
        r'\nOn .+ wrote:\n',  # Gmail-style
        r'\n-{3,} Original Message -{3,}',  # Outlook-style
        r'\nFrom: .+\nSent: .+\nTo: .+\nSubject: .*\n',  # Outlook forward
        r'\n_{3,}\nFrom:',  # Underscore separator
        r'\n>.*$',  # Inline quotes (lines starting with >)
    ]

    # Disclaimers and legal footers
    DISCLAIMER_PATTERNS = [
        r'This email and any files?.*?(?:confidential|privileged|disclaimer).*$',
        r'(?:CONFIDENTIAL|DISCLAIMER|LEGAL NOTICE):.*$',
        r'This message contains information.*?confidential.*$',
        r'(?:Please consider the environment|Think before you print).*$',
    ]

    # Tracking and pixels
    TRACKING_PATTERNS = [
        r'<img[^>]*width=["\']?1["\']?[^>]*height=["\']?1["\']?[^>]*>',  # 1x1 tracking pixels
        r'<img[^>]*height=["\']?1["\']?[^>]*width=["\']?1["\']?[^>]*>',
        r'https?://[^\s]*?(?:track|pixel|beacon|open|analytics)[^\s]*',
        r'https?://[^\s]*?\.gif\?[^\s]*',  # Tracking GIFs
    ]

    # Unsubscribe blocks
    UNSUBSCRIBE_PATTERNS = [
        r'(?:To unsubscribe|Unsubscribe|Manage your preferences|Opt out|Email preferences).*?$',
        r'(?:You are receiving this|This email was sent to|Update your email preferences).*$',
        r'(?:View in browser|View this email in your browser).*$',
    ]

    # HTML cleanup
    HTML_PATTERNS = [
        (r'<br\s*/?>', '\n'),
        (r'<p[^>]*>', '\n'),
        (r'</p>', '\n'),
        (r'<li[^>]*>', '\n- '),
        (r'<h[1-6][^>]*>', '\n## '),
        (r'</h[1-6]>', '\n'),
        (r'<tr[^>]*>', '\n'),
        (r'<td[^>]*>', ' | '),
        (r'<[^>]+>', ''),  # Strip remaining HTML tags
        (r'&nbsp;', ' '),
        (r'&amp;', '&'),
        (r'&lt;', '<'),
        (r'&gt;', '>'),
        (r'&quot;', '"'),
        (r'&#\d+;', ''),
    ]

    def clean(self, body: str) -> str:
        """Clean email body, returning useful content only."""
        if not body:
            return ""

        # Detect and convert HTML to text if needed
        if "<html" in body.lower() or "<body" in body.lower() or "<div" in body.lower():
            body = self._html_to_text(body)

        # Remove tracking pixels and URLs
        for pattern in self.TRACKING_PATTERNS:
            body = re.sub(pattern, '', body, flags=re.IGNORECASE | re.MULTILINE)

        # Find signature boundary and cut
        body = self._strip_signature(body)

        # Strip quoted replies
        body = self._strip_quotes(body)

        # Strip disclaimers
        for pattern in self.DISCLAIMER_PATTERNS:
            body = re.sub(pattern, '', body, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

        # Strip unsubscribe blocks
        for pattern in self.UNSUBSCRIBE_PATTERNS:
            body = re.sub(pattern, '', body, flags=re.IGNORECASE | re.MULTILINE)

        # Clean up whitespace
        body = re.sub(r'\n{3,}', '\n\n', body)
        body = re.sub(r'[ \t]+', ' ', body)
        body = body.strip()

        return body

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        text = html
        for pattern, replacement in self.HTML_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _strip_signature(self, body: str) -> str:
        """Remove email signature."""
        for pattern in self.SIGNATURE_PATTERNS:
            match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
            if match:
                # Keep content before the signature
                return body[:match.start()].rstrip()
        return body

    def _strip_quotes(self, body: str) -> str:
        """Remove quoted reply text."""
        for pattern in self.QUOTE_PATTERNS:
            match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
            if match:
                return body[:match.start()].rstrip()

        # Also strip lines starting with >
        lines = body.split('\n')
        clean_lines = []
        in_quote = False
        for line in lines:
            if line.strip().startswith('>'):
                in_quote = True
                continue
            if in_quote and line.strip() == '':
                continue
            in_quote = False
            clean_lines.append(line)

        return '\n'.join(clean_lines)

    def extract_links(self, body: str) -> list[str]:
        """Extract useful links from email body (not tracking links)."""
        url_pattern = r'https?://[^\s<>"\')\]]+'
        urls = re.findall(url_pattern, body)

        # Filter out tracking/pixel URLs
        tracking_keywords = {'track', 'pixel', 'beacon', 'open', 'analytics', 'unsubscribe', 'click'}
        useful_urls = []
        for url in urls:
            url_lower = url.lower()
            if not any(kw in url_lower for kw in tracking_keywords):
                useful_urls.append(url)

        return list(set(useful_urls))[:20]  # Cap at 20 links
