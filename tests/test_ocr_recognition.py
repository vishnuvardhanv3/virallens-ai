"""Unit & integration tests for specialist Video OCR layer."""
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

from agents.ocr_agent import (
    extract_video_ocr,
    _clean_ocr_token,
    _is_substantive_token,
    _classify_position,
)


def create_synthetic_video(file_path: str, text: str = "", duration_sec: float = 2.0, fps: int = 10):
    """Generate a temporary synthetic MP4 video for OCR testing."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    for i in range(total_frames):
        # Dark gray background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        if text:
            # White high-contrast text centered
            cv2.putText(
                frame,
                text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        out.write(frame)
    out.release()


def test_token_filtering_and_cleaning():
    """Verify noise, garbage tokens, and punctuation-only words are cleanly rejected."""
    # Garbage tokens
    garbage = ["#", "///", "...", "--", "x", "a", "12", "\ufffd\ufffd", "|", "@!"]
    for g in garbage:
        assert _is_substantive_token(g) is False, f"Expected '{g}' to be rejected as non-substantive"

    # Substantive words
    substantive = ["Hello", "World", "ViralLens", "Attention", "Growth", "Camera"]
    for s in substantive:
        cleaned = _clean_ocr_token(s)
        assert _is_substantive_token(cleaned) is True, f"Expected '{cleaned}' to be accepted"


def test_position_classification():
    """Verify bounding box vertical coordinates map to screen thirds accurately."""
    frame_h = 1000
    # Top region (y < 333)
    assert _classify_position(50, frame_h) == "upper third"
    assert _classify_position(250, frame_h) == "upper third"
    # Center region (333 <= y <= 666)
    assert _classify_position(450, frame_h) == "center"
    assert _classify_position(550, frame_h) == "center"
    # Bottom region (y > 666)
    assert _classify_position(750, frame_h) == "lower third"
    assert _classify_position(900, frame_h) == "lower third"


def test_ocr_blank_video_no_text():
    """A video with no text must return text_present=False and empty segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "blank.mp4")
        create_synthetic_video(video_path, text="", duration_sec=1.5, fps=10)

        res = extract_video_ocr(video_path, max_frames=10)
        assert res["text_present"] is False
        assert len(res["segments"]) == 0
        assert res["first_text_time"] is None
        assert res["total_text_duration"] == 0.0
        assert res["ocr_text"] == ""


def test_ocr_synthetic_text_recognition():
    """A video with synthetic overlay text must detect text presence and extract segments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = str(Path(tmpdir) / "with_text.mp4")
        create_synthetic_video(video_path, text="VIRALLENS AI", duration_sec=2.0, fps=10)

        res = extract_video_ocr(video_path, max_frames=15)
        assert "ocr_available" in res
        assert "text_present" in res
        assert "segments" in res
        assert "diagnostics" in res
        assert res["diagnostics"]["frames_sampled"] > 0

        # If tesseract is installed in test environment, verify extraction
        if res["ocr_available"]:
            assert res["text_present"] is True
            assert len(res["segments"]) > 0
            seg = res["segments"][0]
            assert "start_time" in seg
            assert "end_time" in seg
            assert "duration" in seg
            assert "confidence" in seg
            assert "position" in seg
            assert "VIRALLENS" in res["ocr_text"].upper() or "VIRAL" in res["ocr_text"].upper()


def test_ocr_graceful_missing_file_handling():
    """Non-existent video file path must fail gracefully without crashing."""
    res = extract_video_ocr("non_existent_path_12345.mp4")
    assert res["text_present"] is False
    assert len(res["segments"]) == 0
    assert res["ocr_text"] == ""
    assert res["first_text_time"] is None
