"""Structure Agent - narrative and hook structure helper."""
from __future__ import annotations

from typing import Any


def parse_structure(analysis: dict[str, Any]) -> dict[str, Any]:
    """Extract narrative structure components."""
    struct = analysis.get("structure") or {}
    return {
        "opening": struct.get("opening"),
        "development": struct.get("development"),
        "payoff": struct.get("payoff"),
        "cta": struct.get("cta"),
    }
