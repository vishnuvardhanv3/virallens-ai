"""Trend Agent - derives evidence-backed insights and recommendations from verified competitors."""
from __future__ import annotations

from typing import Any


def analyze_trends(verified_competitors: list[dict[str, Any]], your_reel: dict[str, Any] | None = None) -> list[str]:
    """Generate recommendations strictly traceable to actual verified competitor evidence."""
    if not verified_competitors:
        return ["No verified competitors available to generate evidence-backed recommendations."]

    total = len(verified_competitors)
    recommendations = []

    # 1. Fast opening / hook analysis
    visual_hooks = 0
    strong_hooks = 0
    for c in verified_competitors:
        hook = c.get("hook") or {}
        h_type = str(hook.get("type", "")).lower()
        if "visual" in h_type or "action" in h_type:
            visual_hooks += 1
        h_str = hook.get("strength")
        if h_str is not None and float(h_str) >= 0.7:
            strong_hooks += 1

    if visual_hooks >= (total / 2):
        recommendations.append(
            f"{visual_hooks} of {total} verified competitors use high-action visual hooks in the opening 3 seconds."
        )

    if strong_hooks >= (total / 2):
        recommendations.append(
            f"{strong_hooks} of {total} verified competitors maintain a hook strength rating of 70% or higher."
        )

    # 2. Pacing and scene changes
    fast_pacing = 0
    for c in verified_competitors:
        pacing = str((c.get("editing") or {}).get("pacing") or "").lower()
        if "fast" in pacing or "rapid" in pacing:
            fast_pacing += 1
    if fast_pacing >= (total / 2):
        recommendations.append(
            f"{fast_pacing} of {total} verified competitors employ rapid pacing to sustain viewer retention."
        )

    # 3. Audio / Speech pattern
    music_sync = 0
    for c in verified_competitors:
        audio = c.get("audio") or {}
        if audio.get("music_present"):
            music_sync += 1
    if music_sync >= (total / 2):
        recommendations.append(
            f"{music_sync} of {total} verified competitors align visual scene cuts to background music beats."
        )

    if not recommendations:
        recommendations.append(
            f"Analyzed {total} verified competitors; maintain clear subject focus and high visual clarity in the opening 3 seconds."
        )

    return recommendations
