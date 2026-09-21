"""Relevance filtering and Twelve Labs semantic relevance bridge.

Handles:
1. Metadata relevance filtering before expensive MP4 download.
2. Semantic video relevance scoring via Twelve Labs verification.
"""
from __future__ import annotations

from typing import Any, Mapping
from providers.twelvelabs_provider import TwelveLabsProvider
from utils.helpers import sanitize_term


def _candidate_to_dict(cand: Any) -> dict[str, Any]:
    if isinstance(cand, dict):
        return dict(cand)
    d = {}
    for attr in ("id", "url", "shortcode", "caption", "username", "creator",
                 "likes", "comments", "views", "media_url", "thumbnail_url",
                 "source_query", "search_level", "provider", "status"):
        if hasattr(cand, attr):
            d[attr] = getattr(cand, attr)
    return d


def filter_relevant_candidates(
    candidates: list[Any],
    profile: Mapping[str, Any] | dict,
    minimum_score: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank and filter candidates using metadata signals before downloading videos.

    Evaluates the post caption and metadata against the profile concepts.
    Rejects candidates with zero semantic overlap in caption/metadata.
    """
    niche = sanitize_term(profile.get("niche") or "")
    entity = sanitize_term(profile.get("primary_entity") or "")
    topic = sanitize_term(profile.get("topic") or "")
    sub_niche = sanitize_term(profile.get("sub_niche") or "")
    edit_type = sanitize_term(profile.get("edit_type") or "")
    keywords = profile.get("keywords") or []

    target_terms: set[str] = set()
    for item in [niche, entity, topic, sub_niche, edit_type] + list(keywords):
        if item:
            for word in sanitize_term(item).split():
                if len(word) > 2:
                    target_terms.add(word)

    relevant: list[dict[str, Any]] = []
    rejected_count = 0

    for raw_cand in candidates:
        cand = _candidate_to_dict(raw_cand)
        caption = str(cand.get("caption") or "").lower()
        username = str(cand.get("username") or cand.get("creator") or "").lower()

        # Score based strictly on candidate caption / creator overlap
        content_text = f"{caption} {username}".strip()
        matched_terms = [w for w in sorted(target_terms) if w in content_text]

        if not target_terms:
            score = 0.5
        elif matched_terms:
            score = len(matched_terms) / len(target_terms)
            # Bonus if primary entity is explicitly in caption
            if entity and entity in content_text:
                score += 0.35
        else:
            # Zero matching terms in caption/creator metadata
            score = 0.0

        score = min(1.0, round(score, 4))
        cand["relevance_score"] = score
        cand["matched_terms"] = matched_terms

        if score >= minimum_score:
            cand["relevance_reason"] = f"Matched {len(matched_terms)} term(s): {', '.join(matched_terms[:4])}"
            relevant.append(cand)
        else:
            rejected_count += 1
            cand["relevance_reason"] = "No matching profile terms in caption or metadata."

    diagnostics = {
        "raw_candidates": len(candidates),
        "relevant_candidates": len(relevant),
        "rejected_candidates": rejected_count,
        "relevance_threshold": minimum_score,
    }
    return relevant, diagnostics


def score_video_relevance(
    uploaded_analysis: Mapping[str, Any] | dict | None,
    competitor_analysis: Mapping[str, Any] | dict | None,
) -> dict[str, Any]:
    """Bridge function delegating to TwelveLabsProvider semantic relevance verification."""
    provider = TwelveLabsProvider()
    return provider.verify_semantic_relevance(uploaded_analysis, competitor_analysis)
