"""Audio Processor - extracts technical acoustic signals using librosa.

Guarantees automatic cleanup of temporary audio files.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from config.settings import FFMPEG_BIN


def extract_technical_audio_signals(
    video_path: str | Path, cache_dir: str | Path | None = None
) -> dict[str, Any]:
    """Extract acoustic signals (RMS energy, tempo, duration) from video using librosa."""
    path = Path(video_path)
    if not path.exists():
        return {
            "has_audio": False,
            "tempo": None,
            "rms_mean": None,
            "error": f"File not found: {path}",
        }

    # Extract temporary WAV file via FFmpeg
    temp_wav = Path(tempfile.gettempdir()) / f"temp_vl_{os.getpid()}_{path.stem}.wav"
    try:
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-v", "error",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", "22050",
            str(temp_wav),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0 or not temp_wav.exists() or temp_wav.stat().st_size == 0:
            return {
                "has_audio": False,
                "tempo": None,
                "rms_mean": None,
                "error": "No decodable audio stream found.",
            }

        # Analyze with librosa
        import librosa
        import numpy as np

        y, sr = librosa.load(str(temp_wav), sr=22050, duration=60.0)
        if len(y) == 0:
            return {
                "has_audio": False,
                "tempo": None,
                "rms_mean": None,
                "error": "Empty audio stream.",
            }

        rms = float(np.mean(librosa.feature.rms(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)

        return {
            "has_audio": True,
            "tempo": round(tempo_val, 1),
            "rms_mean": round(rms, 4),
            "error": "",
        }
    except Exception as exc:
        return {
            "has_audio": False,
            "tempo": None,
            "rms_mean": None,
            "error": f"Audio processing error: {exc}",
        }
    finally:
        # Guaranteed cleanup of temporary WAV file
        if temp_wav.exists():
            temp_wav.unlink(missing_ok=True)
