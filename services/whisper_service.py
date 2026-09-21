"""Local Whisper speech transcription service."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from config.settings import WHISPER_MODEL

_WHISPER_MODEL_INSTANCE = None


def transcribe_video(video_path: str | Path) -> dict[str, Any]:
    """Transcribe video audio locally using Whisper."""
    global _WHISPER_MODEL_INSTANCE
    path = Path(video_path)
    if not path.exists():
        return {
            "has_speech": False,
            "speech_text": "",
            "speech_language": "",
            "speech_confidence": 0.0,
            "error": f"Video not found: {path}",
        }

    try:
        import whisper
        if _WHISPER_MODEL_INSTANCE is None:
            _WHISPER_MODEL_INSTANCE = whisper.load_model(WHISPER_MODEL)

        result = _WHISPER_MODEL_INSTANCE.transcribe(str(path), fp16=False)
        text = result.get("text", "").strip()
        lang = result.get("language", "en")

        return {
            "has_speech": bool(text),
            "speech_text": text,
            "speech_language": lang,
            "speech_confidence": 0.9 if text else 0.0,
            "error": "",
        }
    except Exception as exc:
        return {
            "has_speech": False,
            "speech_text": "",
            "speech_language": "",
            "speech_confidence": 0.0,
            "error": f"Whisper transcription unavailable: {exc}",
        }
