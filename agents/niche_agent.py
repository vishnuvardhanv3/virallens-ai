"""Niche Agent - heuristic niche and topic detection fallback."""
from __future__ import annotations

from typing import Any


def detect_niche(title: str = "", caption: str = "", hashtags: list[str] | None = None) -> dict[str, Any]:
    """Detect niche and topic from metadata text."""
    combined = f"{title} {caption} {' '.join(hashtags or [])}".lower()

    niche_keywords = {
        "superhero": ["spiderman", "batman", "superman", "marvel", "dc", "avengers", "superhero"],
        "fitness": ["workout", "gym", "fitness", "bodybuilding", "exercise", "training"],
        "comedy": ["funny", "skit", "humor", "comedy", "prank", "relatable", "joke"],
        "gaming": ["gameplay", "gaming", "fortnite", "minecraft", "playstation", "xbox"],
        "tech": ["gadget", "ai", "coding", "software", "tech", "technology", "phone"],
    }

    detected_niche = "creative"
    for n, kws in niche_keywords.items():
        if any(kw in combined for kw in kws):
            detected_niche = n
            break

    return {
        "niche": detected_niche,
        "primary_topic": detected_niche,
        "keywords": [w for w in combined.split() if len(w) > 4][:5],
    }
