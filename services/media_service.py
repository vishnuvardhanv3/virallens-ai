"""Media Service - downloads, validates, and manages competitor Reel video files.

Guarantees:
- Real HTTP download of candidate MP4s.
- Video validation via OpenCV and file inspection.
- On failure: explicit 'DOWNLOAD_FAILED' status.
- NEVER creates fake placeholder video files.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import requests

from config.settings import FFMPEG_BIN, MEDIA_DIR

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


class MediaFetcher:
    """Downloads accessible media files (MP4) from URLs or local paths."""

    def fetch_from_url(
        self, media_url: str, destination: Path | str, headers: dict | None = None
    ) -> tuple[bool, str]:
        """Fetch video file from media_url and save to destination.

        Returns (ok: bool, error: str).
        """
        if not media_url:
            return False, "No media URL provided for candidate."

        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)

        clean_url = str(media_url).strip()
        # Local file paths or file:// URLs
        if clean_url.startswith("file://"):
            local_path = clean_url.replace("file://", "", 1)
            if os.path.exists(local_path):
                try:
                    shutil.copy(local_path, dest)
                    return True, ""
                except Exception as exc:
                    return False, f"Failed to copy local file: {exc}"
        elif os.path.exists(clean_url):
            try:
                shutil.copy(clean_url, dest)
                return True, ""
            except Exception as exc:
                return False, f"Failed to copy local file: {exc}"

        # Remote HTTP/HTTPS download
        try:
            req_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                "Referer": "https://www.instagram.com/",
                "Accept": "*/*",
            }
            if headers:
                req_headers.update(headers)
            response = requests.get(
                clean_url, headers=req_headers, stream=True, timeout=(10, 30)
            )
            response.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            if not dest.exists() or dest.stat().st_size == 0:
                dest.unlink(missing_ok=True)
                return False, "Downloaded media file is empty (0 bytes)."

            return True, ""
        except Exception as exc:
            dest.unlink(missing_ok=True)
            return False, f"HTTP download failed: {exc}"


def validate_video_file(path: str | Path) -> dict[str, Any]:
    """Verify that a video file exists, is non-empty, and has valid decodable video frames."""
    file_path = Path(path)
    if not file_path.exists():
        return {"valid": False, "error": "Video file does not exist."}
    size_bytes = file_path.stat().st_size
    if size_bytes <= 0:
        return {"valid": False, "error": "Video file is zero bytes.", "size_bytes": 0}
    if file_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        return {
            "valid": False,
            "error": f"Unsupported video format: {file_path.suffix or 'none'}.",
            "size_bytes": size_bytes,
        }

    # Verify video readability with OpenCV
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return {
                "valid": False,
                "error": "OpenCV could not open video stream.",
                "size_bytes": size_bytes,
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = round(frame_count / fps, 2) if fps > 0 and frame_count > 0 else 0.0

        # Attempt to read at least the first frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {
                "valid": False,
                "error": "Video frames could not be decoded.",
                "size_bytes": size_bytes,
            }

        return {
            "valid": True,
            "error": "",
            "size_bytes": size_bytes,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        }
    except Exception as exc:
        return {
            "valid": False,
            "error": f"Video validation exception: {exc}",
            "size_bytes": size_bytes,
        }


def acquire(candidate: Any, target_dir: Path | str) -> dict[str, Any]:
    """Acquire real video media for candidate.

    Returns dict describing download status.
    Failure sets status='media_download_failed' and video_download_status='DOWNLOAD_FAILED'.
    """
    dest_dir = Path(target_dir) / "competitors"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Resolve candidate properties
    cand_id = getattr(candidate, "id", None) or (candidate.get("id") if isinstance(candidate, dict) else "unknown")
    shortcode = getattr(candidate, "shortcode", None) or (candidate.get("shortcode") if isinstance(candidate, dict) else cand_id)
    media_url = getattr(candidate, "media_url", None) or (candidate.get("media_url") if isinstance(candidate, dict) else "")

    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", str(shortcode or cand_id))
    local_path = dest_dir / f"apify_{safe_name}.mp4"

    # Check if already present and valid
    if local_path.exists():
        val = validate_video_file(local_path)
        if val.get("valid"):
            return {
                "media_status": "available",
                "video_download_status": "DOWNLOADED",
                "local_path": str(local_path),
                "cached": True,
                "download_attempted": False,
                "size_bytes": val.get("size_bytes", 0),
                "duration_sec": val.get("duration_sec", 0.0),
                "error": "",
            }
        else:
            local_path.unlink(missing_ok=True)

    if not media_url:
        return {
            "media_status": "media_unavailable",
            "video_download_status": "DOWNLOAD_FAILED",
            "local_path": "",
            "cached": False,
            "download_attempted": False,
            "size_bytes": 0,
            "duration_sec": 0.0,
            "error": "No media URL available for video download.",
        }

    fetcher = MediaFetcher()
    ok, err = fetcher.fetch_from_url(media_url, local_path)
    if not ok:
        return {
            "media_status": "media_download_failed",
            "video_download_status": "DOWNLOAD_FAILED",
            "local_path": "",
            "cached": False,
            "download_attempted": True,
            "size_bytes": 0,
            "duration_sec": 0.0,
            "error": err or "Media download failed.",
        }

    val = validate_video_file(local_path)
    if not val.get("valid"):
        local_path.unlink(missing_ok=True)
        return {
            "media_status": "media_download_failed",
            "video_download_status": "DOWNLOAD_FAILED",
            "local_path": "",
            "cached": False,
            "download_attempted": True,
            "size_bytes": val.get("size_bytes", 0),
            "duration_sec": 0.0,
            "error": val.get("error", "Downloaded video is corrupted or undecodable."),
        }

    return {
        "media_status": "available",
        "video_download_status": "DOWNLOADED",
        "local_path": str(local_path),
        "cached": False,
        "download_attempted": True,
        "size_bytes": val.get("size_bytes", 0),
        "duration_sec": val.get("duration_sec", 0.0),
        "error": "",
    }
