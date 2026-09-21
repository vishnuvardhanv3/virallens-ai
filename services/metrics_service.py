"""Metrics Service - handles presentation formatting and metric evaluations.

Guarantees:
- Canonical values (0.0-1.0) format as percentages without scaling bugs (e.g. 0.8 -> 80%).
- Never produces 800% or 8% from 0.8 or 8.
- Missing values format gracefully as fallback strings (e.g. 'Unavailable').
"""
from __future__ import annotations

from typing import Any
from services.canonical_schema import normalize_score_0_to_1


def format_percentage(value: Any, fallback: str = "Unavailable") -> str:
    """Safely format a canonical 0.0-1.0 float or raw score as a percentage string.

    Examples:
        0.8  -> "80%"
        8    -> "80%" (normalized first from 1-10)
        80   -> "80%" (normalized first from 0-100)
        0.73 -> "73%"
        None -> fallback ("Unavailable")
    """
    normalized = normalize_score_0_to_1(value)
    if normalized is None:
        return fallback
    return f"{round(normalized * 100):.0f}%"


def calculate_engagement_rate(
    likes: int | None, comments: int | None, views: int | None
) -> float | None:
    """Calculate engagement rate: (likes + comments) / views.

    Returns None if views are missing or zero, or if both likes and comments are missing.
    """
    if views is None or views <= 0:
        return None
    if likes is None and comments is None:
        return None
    total_interactions = (likes or 0) + (comments or 0)
    return round(total_interactions / views, 4)


def format_number(value: int | float | None, fallback: str = "Unavailable") -> str:
    """Format large numbers with commas, e.g. 1,234,567."""
    if value is None or value == "":
        return fallback
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return fallback
