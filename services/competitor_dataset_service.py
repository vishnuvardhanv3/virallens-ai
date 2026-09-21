"""Competitor Dataset Service - persists structured competitor records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from config.settings import COMPETITORS_DATA_DIR


def save_competitor_dataset(competitors: list[dict[str, Any]]) -> Path:
    """Save structured competitor dataset into data/competitors/."""
    COMPETITORS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    index_file = COMPETITORS_DATA_DIR / "index.json"

    index_entries = []
    for comp in competitors:
        cid = str(comp.get("id") or comp.get("shortcode") or "unknown")
        safe_name = "".join(c for c in cid if c.isalnum() or c in "_-")
        comp_file = COMPETITORS_DATA_DIR / f"{safe_name}.json"

        try:
            with open(comp_file, "w", encoding="utf-8") as f:
                json.dump(comp, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        index_entries.append({
            "id": comp.get("id"),
            "shortcode": comp.get("shortcode"),
            "url": comp.get("url"),
            "creator": comp.get("creator"),
            "niche": comp.get("niche"),
            "topic": comp.get("topic"),
            "verification_status": comp.get("video_verification_status"),
            "file": comp_file.name,
        })

    try:
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_entries, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return index_file
