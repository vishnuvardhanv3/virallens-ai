"""Transcript Quality Agent - validates speech confidence and quality."""
from __future__ import annotations

from typing import Any


def assess_quality(speech_info: dict[str, Any]) -> dict[str, Any]:
    """Assess transcript quality."""
    text = speech_info.get("speech_text", "")
    return {
        "has_speech": bool(text),
        "word_count": len(text.split()),
        "confidence": speech_info.get("speech_confidence", 0.0),
    }
