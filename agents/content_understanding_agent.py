"""Content Understanding Agent - integrates multimodal outputs."""
from __future__ import annotations

from typing import Any
from services.canonical_schema import build_canonical_analysis


def assemble_content_profile(analysis_res: dict[str, Any], tech_signals: dict[str, Any]) -> dict[str, Any]:
    """Combine Twelve Labs understanding and technical signals."""
    analysis = analysis_res.get("analysis") or {}
    canonical = build_canonical_analysis(
        analysis=analysis,
        video=tech_signals.get("video"),
        audio=tech_signals.get("audio"),
        transcript=(tech_signals.get("speech") or {}).get("speech_text"),
        ocr_text=(tech_signals.get("ocr") or {}).get("ocr_text"),
    )
    return canonical
