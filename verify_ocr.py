"""Verification Script: Standalone Video OCR Text Recognition.

Usage:
    python verify_ocr.py <video_path>

Example:
    python verify_ocr.py htdyr.mp4

Outputs:
    - video duration
    - sampled frames
    - detected text
    - timestamps (start_time, end_time)
    - confidence
    - unique text count
    - first text timestamp
    - total text duration
"""
from __future__ import annotations

import sys
from pathlib import Path
from agents.ocr_agent import extract_ocr


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python verify_ocr.py <video_path>")
        return 1

    video_input = sys.argv[1]
    p = Path(video_input)
    if not p.exists():
        # Also check media/ directory
        p_alt = Path("media") / video_input
        if p_alt.exists():
            p = p_alt
        else:
            print(f"Error: Video file not found: {video_input}")
            return 1

    print("=" * 60)
    print(f"VIRALLENS — SPECIALIST VIDEO OCR VERIFICATION")
    print(f"Target Video: {p.resolve()}")
    print("=" * 60)

    res = extract_ocr(p)
    diag = res.get("diagnostics", {})

    print(f"Video Duration:       {diag.get('duration_seconds', 0.0):.2f}s")
    print(f"Sampled Frames:       {diag.get('sampled_frames', 0)}")
    print(f"OCR Engine:           {diag.get('ocr_engine', 'Tesseract')}")
    print(f"Text Present:         {res.get('text_present')}")
    print(f"Unique Text Count:    {len(res.get('unique_text', []))}")
    print(f"First Text Timestamp: {res.get('first_text_time')}s" if res.get('first_text_time') is not None else "First Text Timestamp: None")
    print(f"Total Text Duration:  {res.get('total_text_duration', 0.0):.2f}s")
    print(f"Text Density:         {res.get('text_density', 0.0):.4f}")
    conf_val = float(res.get('ocr_confidence', 0.0) or 0.0)
    conf_str = f"{conf_val:.1f}%" if conf_val > 1.0 else f"{conf_val * 100:.1f}%"
    print(f"Mean Confidence:      {conf_str}")
    print("-" * 60)
    print("DETECTED TEXT SEGMENTS (CHRONOLOGICAL):")

    segments = res.get("segments", [])
    if not segments:
        print("  (No on-screen text detected)")
    else:
        for idx, seg in enumerate(segments, 1):
            s_time = seg.get("start_time", 0.0)
            e_time = seg.get("end_time", 0.0)
            conf = seg.get("confidence", 0.0)
            txt = seg.get("text", "")
            q_tier = seg.get("quality_tier", "usable")
            bbox = seg.get("bbox", [])
            print(f"  [{idx:02d}] {s_time:.2f}s – {e_time:.2f}s (tier: {q_tier}, conf: {conf:.1f}%, bbox: {bbox}): '{txt}'")

    print("-" * 60)
    print("UNIQUE CONSOLIDATED TEXT:")
    for t in res.get("unique_text", []):
        print(f"  • {t}")

    print("=" * 60)
    print("[SUCCESS] OCR VERIFICATION COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
