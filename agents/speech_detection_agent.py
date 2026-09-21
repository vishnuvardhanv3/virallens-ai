"""Speech Detection Agent - checks whether speech is present."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from services.whisper_service import transcribe_video


def detect_speech(video_path: str | Path) -> dict[str, Any]:
    """Determine whether video has detectable speech."""
    res = transcribe_video(video_path)
    return {
        "has_speech": res.get("has_speech", False),
        "confidence": res.get("speech_confidence", 0.0),
        "transcript_snippet": (res.get("speech_text") or "")[:120],
    }
