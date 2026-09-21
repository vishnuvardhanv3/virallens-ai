"""Audio Agent - wraps audio processing and acoustics extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from services.audio_processor import extract_technical_audio_signals


def analyze_audio(video_path: str | Path, cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Analyze audio track from video file."""
    return extract_technical_audio_signals(video_path, cache_dir)
