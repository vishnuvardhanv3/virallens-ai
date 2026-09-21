"""Canonical video-analysis schema shared across ML, comparison, and UI.

Single source of truth for normalized Reel features.
All normalized scores/impacts use 0.0 - 1.0 float internally.
Missing values remain None (never fabricated as zero).
"""
from __future__ import annotations

from typing import Any, Mapping


def normalize_score_0_to_1(value: Any) -> float | None:
    """Normalize any numeric representation (0-1, 1-10, or 0-100) to canonical 0.0-1.0 float.

    Preserves None for missing evidence; never fabricates zero.
    Rules:
        None or ""  -> None
        0.8         -> 0.8  (already 0.0-1.0)
        8           -> 0.8  (from 1-10 scale)
        80          -> 0.8  (from 0-100 scale)
        False       -> 0.0
        True        -> 1.0
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None
    if val != val:  # NaN check
        return None
    if val < 0.0:
        return 0.0
    if val <= 1.0:
        return round(val, 4)
    if val <= 10.0:
        return round(val / 10.0, 4)
    if val <= 100.0:
        return round(val / 100.0, 4)
    return 1.0


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def build_canonical_analysis(
    *,
    analysis: Mapping[str, Any] | None = None,
    video: Mapping[str, Any] | None = None,
    audio: Mapping[str, Any] | None = None,
    visual: Mapping[str, Any] | None = None,
    hook: Mapping[str, Any] | None = None,
    structure: Mapping[str, Any] | None = None,
    engagement: Mapping[str, Any] | None = None,
    content: Mapping[str, Any] | None = None,
    transcript: str | None = None,
    ocr_text: str | None = None,
    ocr: Mapping[str, Any] | None = None,
    relevance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the canonical normalized Reel profile dictionary."""
    analysis = analysis or {}
    video = video or {}
    audio = audio or {}
    visual = visual or {}
    hook = hook or {}
    structure = structure or {}
    engagement = engagement or {}
    content = content or {}
    ocr = ocr or {}
    relevance = relevance or {}

    raw_hook_strength = _first(hook, "strength", "score", "hook_score")
    hook_strength = normalize_score_0_to_1(raw_hook_strength)

    raw_visual_impact = _first(visual, "visual_impact", "impact", "score")
    visual_impact = normalize_score_0_to_1(raw_visual_impact)

    raw_audio_score = _first(audio, "quality", "clarity", "score")
    audio_quality = normalize_score_0_to_1(raw_audio_score)

    raw_relevance = _first(relevance, "video_relevance_score", "score", "relevance")
    relevance_score = normalize_score_0_to_1(raw_relevance)

    # Narrative structure
    opening = _first(structure, "opening", "hook")
    development = _first(structure, "development", "body")
    payoff = _first(structure, "payoff", "conclusion", "ending")
    cta = _first(structure, "cta", "call_to_action")

    # Speech and OCR text
    final_transcript = transcript or _first(audio, "transcript", "speech_summary")
    final_ocr_text = _first(ocr, "ocr_text") or ocr_text or _first(visual, "ocr_text", "on_screen_text_summary")
    text_segments = list(ocr.get("segments", []))
    unique_text = list(ocr.get("unique_text", []))
    text_present = bool(ocr.get("text_present") or text_segments or final_ocr_text)
    first_text_time = ocr.get("first_text_time")
    text_duration = float(ocr.get("total_text_duration", 0.0) or 0.0)
    text_density = float(ocr.get("text_density") or _first(visual, "text_density") or 0.0)
    text_change_rate = float(ocr.get("text_change_rate", 0.0) or 0.0)
    ocr_confidence = float(ocr.get("ocr_confidence", 0.0) or 0.0)

    # Speech metrics
    speech_present = bool(_first(audio, "speech_present", "has_speech") or final_transcript)
    word_count = len(final_transcript.split()) if final_transcript else 0
    speech_onset = _first(audio, "speech_onset")
    speech_duration = _first(audio, "speech_duration")
    audio_energy = float(_first(audio, "rms_mean", "audio_energy") or 0.0)
    silence_ratio = _first(audio, "silence_ratio")

    return {
        "niche": _first(analysis, "niche") or _first(content, "niche"),
        "sub_niche": _first(analysis, "sub_niche") or _first(content, "sub_niche"),
        "topic": _first(analysis, "topic") or _first(content, "topic"),
        "primary_entity": _first(analysis, "primary_entity", "primary_subject") or _first(content, "primary_entity"),
        "format": _first(analysis, "format") or _first(content, "format"),
        "style": _first(analysis, "visual_style", "style") or _first(visual, "style"),
        "editing_style": _first(analysis, "editing_style"),
        "tone": _first(analysis, "tone"),
        "language": _first(analysis, "language"),
        "duration_seconds": _first(video, "duration_sec", "duration_seconds"),
        "fps": _first(video, "fps"),
        "scene_changes": _first(video, "scene_change_count", "scene_changes"),
        "motion_mean": _first(video, "motion_mean"),
        "contrast_mean": _first(video, "contrast_mean"),
        "hook": {
            "type": _first(hook, "type", "hook_type"),
            "summary": _first(hook, "summary", "description"),
            "first_3_seconds": _first(hook, "first_3_seconds", "opening"),
            "strength": hook_strength,
        },
        "structure": {
            "opening": opening,
            "development": development,
            "payoff": payoff,
            "cta": cta,
        },
        "visual": {
            "style": _first(visual, "style"),
            "scene_count": _first(visual, "scene_count", "scene_changes") or _first(video, "scene_change_count"),
            "text_density": text_density,
            "visual_impact": visual_impact,
        },
        "editing": {
            "pacing": _first(analysis, "editing_pacing", "pacing"),
            "scene_change_rate": _first(video, "scene_change_rate"),
        },
        "audio": {
            "speech_present": speech_present,
            "music_present": _first(audio, "music_present", "has_music"),
            "tempo": _first(audio, "tempo"),
            "rms_mean": audio_energy,
            "audio_energy": audio_energy,
            "silence_ratio": silence_ratio,
            "transcript": final_transcript,
            "quality": audio_quality,
        },
        "speech": {
            "speech_present": speech_present,
            "transcript": final_transcript,
            "word_count": word_count,
            "speech_onset": speech_onset,
            "speech_duration": speech_duration,
            "vocal_activity": 0.8 if final_transcript else 0.0,
            "confidence": _first(audio, "speech_confidence"),
            "language": _first(audio, "speech_language") or _first(analysis, "language"),
        },
        "ocr": {
            "text_present": text_present,
            "text_presence": text_present,
            "segments": text_segments,
            "text_segments": text_segments,
            "unique_text": unique_text,
            "first_text_time": first_text_time,
            "text_duration": text_duration,
            "total_text_duration": text_duration,
            "text_density": text_density,
            "text_change_rate": text_change_rate,
            "ocr_confidence": ocr_confidence,
            "ocr_text": final_ocr_text,
        },
        "onscreen_text": final_ocr_text,
        "keywords": _first(analysis, "keywords", "discovery_keywords") or [],
        "engagement": {
            "views": _first(engagement, "views"),
            "likes": _first(engagement, "likes"),
            "comments": _first(engagement, "comments"),
            "curiosity": normalize_score_0_to_1(_first(engagement, "curiosity")),
            "novelty": normalize_score_0_to_1(_first(engagement, "novelty")),
            "shareability": normalize_score_0_to_1(_first(engagement, "shareability")),
            "saveability": normalize_score_0_to_1(_first(engagement, "saveability")),
            "comment_potential": normalize_score_0_to_1(_first(engagement, "comment_potential")),
        },
        "relevance": {
            "score": relevance_score,
            "status": _first(relevance, "video_verification_status", "status"),
            "reason": _first(relevance, "video_relevance_reason", "reason"),
        },
    }
