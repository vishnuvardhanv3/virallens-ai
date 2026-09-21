"""Video Agent - extracts objective technical video metrics using OpenCV."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import cv2
import numpy as np


def analyze_video(video_path: str | Path, shared_asset: Any = None) -> dict[str, Any]:
    """Extract duration, FPS, scene changes, motion, and contrast using OpenCV."""
    target_path = Path(shared_asset.analysis_video_path if shared_asset else video_path)
    path = Path(video_path)
    if not path.exists() and not target_path.exists():
        return {
            "duration_sec": None,
            "fps": None,
            "frame_count": 0,
            "scene_change_count": 0,
            "scene_change_rate": 0.0,
            "motion_mean": 0.0,
            "contrast_mean": 0.0,
            "error": f"Video not found: {path}",
        }

    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {
                "duration_sec": None,
                "fps": None,
                "frame_count": 0,
                "scene_change_count": 0,
                "scene_change_rate": 0.0,
                "motion_mean": 0.0,
                "contrast_mean": 0.0,
                "error": "Failed to open video file",
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(frame_count / fps, 2) if fps > 0 and frame_count > 0 else 0.0

        # Sample frames for scene changes and motion
        sample_step = max(1, int(fps / 2))  # ~2 frames per second
        prev_gray = None
        motion_diffs = []
        contrast_vals = []
        scene_changes = 0

        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if current_frame % sample_step == 0:
                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                contrast_vals.append(float(np.std(gray)))

                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = float(np.mean(diff))
                    motion_diffs.append(mean_diff)
                    if mean_diff > 35.0:  # Threshold for scene change
                        scene_changes += 1

                prev_gray = gray

            current_frame += 1

        cap.release()

        motion_mean = float(np.mean(motion_diffs)) if motion_diffs else 0.0
        contrast_mean = float(np.mean(contrast_vals)) if contrast_vals else 0.0
        scene_change_rate = round(scene_changes / duration_sec, 2) if duration_sec > 0 else 0.0

        return {
            "duration_sec": duration_sec,
            "fps": round(fps, 1),
            "frame_count": frame_count,
            "scene_change_count": scene_changes,
            "scene_change_rate": scene_change_rate,
            "motion_mean": round(motion_mean, 2),
            "contrast_mean": round(contrast_mean, 2),
            "error": "",
        }
    except Exception as exc:
        return {
            "duration_sec": None,
            "fps": None,
            "frame_count": 0,
            "scene_change_count": 0,
            "scene_change_rate": 0.0,
            "motion_mean": 0.0,
            "contrast_mean": 0.0,
            "error": f"OpenCV processing error: {exc}",
        }
