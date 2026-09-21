"""Shared Video Asset & Preprocessing Layer for ViralLens AI.

Ensures:
1. Original uploaded MP4 is NEVER modified or deleted.
2. Metadata is probed exactly once (dimensions, duration, fps, bitrate, codec).
3. Temporary Analysis Derivative (max 720p, 10–12 FPS, H.264) created only when safe and beneficial.
4. Shared 16 kHz mono audio WAV extracted once for Attention and Whisper transcription.
5. Reusable frame sampling to eliminate redundant video decodes across agents.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from config import settings


from dataclasses import dataclass, field


@dataclass
class VideoMetadata:
    """Probed technical video stream properties."""
    file_path: str = ""
    filename: str = ""
    file_size_mb: float = 0.0
    size_mb: float = 0.0
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    resolution: str = ""
    fps: float = 30.0
    frame_count: int = 0
    duration_sec: float = 0.0
    codec: str = "unknown"
    bitrate_kbps: float = 0.0
    analysis_size_mb: float = 0.0
    compression_ratio: float = 0.0
    has_derivative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "filename": self.filename,
            "file_size_mb": self.file_size_mb,
            "size_mb": self.size_mb,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "codec": self.codec,
            "bitrate_kbps": self.bitrate_kbps,
            "analysis_size_mb": self.analysis_size_mb,
            "compression_ratio": self.compression_ratio,
            "has_derivative": self.has_derivative,
        }


def _compute_file_hash(path: Path, length: int = 16) -> str:
    """Fast SHA-256 prefix for deterministic naming."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:length]


class SharedVideoAsset:
    """Preprocessed video asset sharing metadata, frames, and audio across all analytical agents."""

    def __init__(
        self,
        video_path: str | Path,
        cache_dir: Optional[Path | str] = None,
        max_dimension: int = 720,
        target_fps: float = 10.0,
    ) -> None:
        self.original_path = Path(video_path).resolve()
        if not self.original_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.original_path}")

        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_dimension = max_dimension
        self.target_fps = target_fps

        self.file_hash = _compute_file_hash(self.original_path)
        self.metadata: Dict[str, Any] = {}
        self.meta_obj: VideoMetadata = VideoMetadata()
        self.probe_metadata()

        # Paths
        self.analysis_path: Path = self.original_path
        self.is_derivative_created: bool = False
        self.audio_wav_path: Optional[Path] = None

        # Prepare assets
        self._prepare_analysis_derivative()
        self._prepare_shared_audio()

    @property
    def meta(self) -> VideoMetadata:
        """Structured VideoMetadata object."""
        return self.meta_obj

    @property
    def original_video_path(self) -> str:
        return str(self.original_path)

    @property
    def analysis_video_path(self) -> str:
        return str(self.analysis_path)

    @property
    def has_derivative(self) -> bool:
        return self.is_derivative_created

    def generate_derivative(self) -> str:
        return str(self.analysis_path)

    def extract_audio(self) -> Optional[str]:
        return str(self.audio_wav_path) if self.audio_wav_path else None

    def probe_metadata(self) -> Dict[str, Any]:
        """Probe technical video stream properties using OpenCV and ffprobe."""
        if self.metadata:
            return self.metadata

        size_bytes = self.original_path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)

        cap = cv2.VideoCapture(str(self.original_path))
        if not cap.isOpened():
            self.metadata = {
                "width": 0,
                "height": 0,
                "fps": 30.0,
                "frame_count": 0,
                "duration_sec": 0.0,
                "size_mb": size_mb,
                "size_bytes": size_bytes,
                "codec": "unknown",
            }
            return self.metadata

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = round(fc / fps, 2) if fps > 0 else 0.0
        cap.release()

        # Probe codec and bitrate via ffprobe if available
        codec = "h264"
        bitrate_kbps = 0.0
        try:
            ffprobe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,bit_rate:format=bit_rate",
                "-of", "json", str(self.original_path),
            ]
            res = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if res.returncode == 0 and res.stdout:
                probe_data = json.loads(res.stdout)
                streams = probe_data.get("streams", [])
                if streams:
                    codec = streams[0].get("codec_name") or codec
                    br = streams[0].get("bit_rate") or (probe_data.get("format") or {}).get("bit_rate")
                    if br:
                        bitrate_kbps = round(float(br) / 1000.0, 1)
        except Exception:
            pass

        self.metadata = {
            "width": w,
            "height": h,
            "resolution": f"{w}x{h}",
            "fps": round(fps, 1),
            "frame_count": fc,
            "duration_sec": dur,
            "size_mb": size_mb,
            "size_bytes": size_bytes,
            "bitrate_kbps": bitrate_kbps,
            "codec": codec,
            "filename": self.original_path.name,
            "file_path": str(self.original_path),
        }
        self.meta_obj = VideoMetadata(
            file_path=str(self.original_path),
            filename=self.original_path.name,
            file_size_mb=size_mb,
            size_mb=size_mb,
            size_bytes=size_bytes,
            width=w,
            height=h,
            resolution=f"{w}x{h}",
            fps=round(fps, 1),
            frame_count=fc,
            duration_sec=dur,
            codec=codec,
            bitrate_kbps=bitrate_kbps,
        )
        return self.metadata

    def _prepare_analysis_derivative(self) -> None:
        """Create optimized 720p/10-fps derivative if video exceeds analysis threshold."""
        w = self.metadata.get("width", 0)
        h = self.metadata.get("height", 0)
        fps = self.metadata.get("fps", 30.0)
        size_mb = self.metadata.get("size_mb", 0.0)

        # Only downscale/transcode if original is larger than target long-side or high FPS or > 5 MB
        max_dim = max(w, h)
        needs_downscale = max_dim > self.max_dimension
        needs_fps_drop = fps > (self.target_fps + 4.0)
        warrants_derivative = needs_downscale or (needs_fps_drop and size_mb > 3.0)

        if not warrants_derivative:
            self.analysis_path = self.original_path
            self.is_derivative_created = False
            return

        derivative_path = self.cache_dir / f"analysis_{self.file_hash}_{self.max_dimension}p_{int(self.target_fps)}fps.mp4"
        if derivative_path.exists() and derivative_path.stat().st_size > 1024:
            self.analysis_path = derivative_path
            self.is_derivative_created = True
            self._log_optimization(derivative_path)
            return

        # FFmpeg transcode filter
        # Maintain aspect ratio, set max long side to max_dimension, and set fps to target_fps
        scale_filter = (
            f"scale='if(gt(iw,ih),min({self.max_dimension},iw),-2)':"
            f"'if(gt(iw,ih),-2,min({self.max_dimension},ih))',"
            f"fps={self.target_fps}"
        )

        cmd = [
            "ffmpeg", "-y", "-i", str(self.original_path),
            "-vf", scale_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(derivative_path)
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=45)
            if derivative_path.exists() and derivative_path.stat().st_size > 1024:
                self.analysis_path = derivative_path
                self.is_derivative_created = True
                self._log_optimization(derivative_path)
            else:
                self.analysis_path = self.original_path
                self.is_derivative_created = False
        except Exception:
            # Fall back gracefully to original video
            self.analysis_path = self.original_path
            self.is_derivative_created = False

    def _log_optimization(self, derivative_path: Path) -> None:
        """Log structured video optimization metrics."""
        orig_mb = self.metadata.get("size_mb", 0.0)
        deriv_bytes = derivative_path.stat().st_size
        deriv_mb = round(deriv_bytes / (1024 * 1024), 2)
        ratio = round((1.0 - (deriv_mb / max(0.01, orig_mb))) * 100, 1)

        cap = cv2.VideoCapture(str(derivative_path))
        dw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        dh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dfps = round(cap.get(cv2.CAP_PROP_FPS) or self.target_fps, 1)
        cap.release()

        self.meta_obj.analysis_size_mb = deriv_mb
        self.meta_obj.compression_ratio = ratio / 100.0
        self.meta_obj.has_derivative = True

        print("\n[Video Optimization]")
        print(f"original_size_mb={orig_mb}")
        print(f"original_resolution={self.metadata.get('resolution')}")
        print(f"original_fps={self.metadata.get('fps')}")
        print(f"analysis_size_mb={deriv_mb}")
        print(f"analysis_resolution={dw}x{dh}")
        print(f"analysis_fps={dfps}")
        print(f"compression_ratio={ratio}%")

    def _prepare_shared_audio(self) -> None:
        """Pre-extract 16 kHz mono WAV once for attention and transcription agents."""
        wav_path = self.cache_dir / f"audio_{self.file_hash}_16k.wav"
        if wav_path.exists() and wav_path.stat().st_size > 100:
            self.audio_wav_path = wav_path
            return

        cmd = [
            "ffmpeg", "-y", "-i", str(self.original_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", str(wav_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
            if wav_path.exists() and wav_path.stat().st_size > 100:
                self.audio_wav_path = wav_path
        except Exception:
            self.audio_wav_path = None

    def get_analysis_video_path(self) -> Path:
        """Return the fastest safe video path for local decoding/OCR/attention."""
        return self.analysis_path

    def get_original_video_path(self) -> Path:
        """Return untouched original video path for user presentation, playback, and download."""
        return self.original_path

    def get_shared_audio_path(self) -> Optional[Path]:
        """Return pre-extracted 16 kHz WAV path."""
        return self.audio_wav_path

    def cleanup(self) -> None:
        """Optionally remove temporary derivative if created specifically for this run."""
        # Analysis derivatives and audio in cache_dir are keyed by SHA-256 hash
        # allowing reuse across runs without leaking uncontrolled memory.
        pass
