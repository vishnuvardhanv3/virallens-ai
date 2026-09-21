"""Helper functions for URL normalization, shortcode extraction, and cleaning."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

REEL_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/(?:reel|reels|p)/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def extract_shortcode(url_or_id: Any) -> str:
    """Extract standard Instagram shortcode from URL or string identifier."""
    if not url_or_id:
        return ""
    text = str(url_or_id).strip()
    match = REEL_URL_PATTERN.search(text)
    if match:
        return match.group(1)
    # If already a bare shortcode like 'C2_vRNYhykr'
    clean = re.sub(r"[^A-Za-z0-9_-]", "", text)
    return clean if 5 <= len(clean) <= 35 else ""


def normalize_reel_url(url_or_id: Any) -> str:
    """Normalize any Instagram post/reel URL or shortcode to standard /reel/ URL."""
    shortcode = extract_shortcode(url_or_id)
    if shortcode:
        return f"https://www.instagram.com/reel/{shortcode}/"
    return ""


STOP_WORDS = {
    "and", "the", "a", "of", "to", "is", "in", "for", "on", "with", "as", "at", "by",
    "an", "be", "this", "that", "from", "it", "are", "was", "or", "your", "you", "we",
    "i", "my", "me", "he", "she", "they", "them", "his", "her", "their", "its", "but",
    "not", "about", "would", "there", "what", "so", "up", "out", "if", "who", "get",
    "go", "can", "has", "just", "did", "why", "reels", "reel", "video", "post",
    "uncategorized", "undefined", "null", "none", "other", "general"
}

PROFANITY_AND_GARBAGE = {
    "fuck", "fucking", "fucked", "shit", "bitch", "ass", "crap", "damn",
    "uncategorized", "undefined", "null", "none", "pregnant", "sex"
}


def sanitize_term(term: Any) -> str:
    """Clean a string term removing punctuation, profanity, and short garbage."""
    if not term:
        return ""
    words = re.findall(r"[a-z0-9]+", str(term).lower())
    clean = [w for w in words if w not in PROFANITY_AND_GARBAGE and len(w) > 1]
    return " ".join(clean).strip()
