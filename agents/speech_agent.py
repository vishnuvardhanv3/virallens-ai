"""Speech Agent - speech transcription using Whisper."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from services.whisper_service import transcribe_video


def transcribe(video_path: str | Path, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Transcribe speech in video file."""
    return transcribe_video(video_path)
