"""Comparison Agent - compares uploaded Reel against verified competitors."""
from __future__ import annotations

from typing import Any
from services.canonical_schema import normalize_score_0_to_1


def compare(your_reel: dict[str, Any], verified_competitors: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare uploaded Reel features against verified competitors."""
    if not verified_competitors:
        return {
            "status": "no_competitors",
            "comparison_metrics": {},
            "differences": {},
        }

    # Helper for average
    def _avg_metric(accessor) -> float | None:
        vals = []
        for c in verified_competitors:
            val = accessor(c)
            if val is not None:
                vals.append(val)
        return round(sum(vals) / len(vals), 4) if vals else None

    # Competitor averages
    avg_hook = _avg_metric(lambda c: normalize_score_0_to_1((c.get("hook") or {}).get("strength")))
    avg_visual = _avg_metric(lambda c: normalize_score_0_to_1((c.get("visual") or {}).get("visual_impact")))
    avg_duration = _avg_metric(lambda c: (c.get("analysis") or {}).get("duration_seconds") or (c.get("video") or {}).get("duration_sec"))

    # Uploaded Reel values
    your_hook = normalize_score_0_to_1((your_reel.get("hook") or {}).get("strength"))
    your_visual = normalize_score_0_to_1((your_reel.get("visual") or {}).get("visual_impact"))
    your_duration = (your_reel.get("technical_signals") or {}).get("video", {}).get("duration_sec")

    return {
        "status": "complete",
        "verified_competitor_count": len(verified_competitors),
        "metrics": {
            "hook_strength": {
                "your_value": your_hook,
                "competitor_avg": avg_hook,
                "difference": round(your_hook - avg_hook, 4) if your_hook is not None and avg_hook is not None else None,
            },
            "visual_impact": {
                "your_value": your_visual,
                "competitor_avg": avg_visual,
                "difference": round(your_visual - avg_visual, 4) if your_visual is not None and avg_visual is not None else None,
            },
            "duration_seconds": {
                "your_value": your_duration,
                "competitor_avg": avg_duration,
                "difference": round(your_duration - avg_duration, 2) if your_duration is not None and avg_duration is not None else None,
            },
        },
    }
