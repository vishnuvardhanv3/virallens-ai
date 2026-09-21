"""Deterministic Competitor Artifact Cache for ViralLens AI.

Provides structured caching for competitor Reels to eliminate duplicate
downloads, redundant Twelve Labs API uploads/analyses, OCR processing, and
attention model inferences across analysis runs.

Cache entries are anchored deterministically to canonical Reel identity:
- Instagram shortcode (preferred)
- Canonical Reel ID / URL
- Content-hash fallback

Emits standard telemetry:
- CACHE_HIT
- CACHE_MISS
- CACHE_WRITE
- CACHE_INVALIDATED
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings

logger = logging.getLogger(__name__)


def get_canonical_reel_id(candidate: Any) -> str:
    """Resolve a stable, canonical identity string for a candidate Reel."""
    if isinstance(candidate, str):
        val = candidate.strip()
        match = re.search(r"/(?:reel|p|tv)/([A-Za-z0-9_-]+)", val)
        if match:
            return match.group(1)
        return re.sub(r"[^A-Za-z0-9_-]", "_", val)

    if not isinstance(candidate, dict):
        shortcode = getattr(candidate, "shortcode", None)
        cand_id = getattr(candidate, "id", None)
        url = getattr(candidate, "url", None) or getattr(candidate, "media_url", None)
    else:
        shortcode = candidate.get("shortcode")
        cand_id = candidate.get("id")
        url = candidate.get("url") or candidate.get("media_url")

    if shortcode and str(shortcode).strip() and str(shortcode).strip() != "unknown":
        return str(shortcode).strip()

    code = candidate.get("code") if isinstance(candidate, dict) else getattr(candidate, "code", None)
    if code and str(code).strip() and str(code).strip() != "unknown":
        return str(code).strip()

    if url and str(url).strip():
        match = re.search(r"/(?:reel|p|tv)/([A-Za-z0-9_-]+)", str(url))
        if match:
            return match.group(1)

    if cand_id and str(cand_id).strip() and str(cand_id).strip() != "unknown":
        return str(cand_id).strip()

    return "unknown"


class CompetitorArtifactCache:
    """Local deterministic cache for competitor Reel analyses, OCR, and attention predictions."""

    def __init__(self, cache_dir: Optional[Path | str] = None, ttl_seconds: Optional[int] = None) -> None:
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR / "competitor_artifacts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.hits: int = 0
        self.misses: int = 0
        self.writes: int = 0
        self.invalidations: int = 0

    def _cache_path(self, canonical_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", canonical_id)
        return self.cache_dir / f"comp_{safe_id}.json"

    def get(self, candidate_or_id: Any, profiler: Any = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached competitor analysis if valid and available."""
        canonical_id = get_canonical_reel_id(candidate_or_id)
        if not canonical_id or canonical_id == "unknown":
            self.misses += 1
            if profiler:
                try:
                    profiler.increment_metric("cache_misses", 1)
                except Exception:
                    pass
            return None

        c_file = self._cache_path(canonical_id)
        if not c_file.exists():
            self.misses += 1
            logger.debug(f"[CACHE_MISS] Competitor artifact cache miss for [{canonical_id}]")
            if profiler:
                try:
                    profiler.increment_metric("cache_misses", 1)
                except Exception:
                    pass
            return None

        try:
            with open(c_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data or not isinstance(data, dict) or not data.get("canonical_id"):
                self.invalidate(canonical_id)
                self.misses += 1
                return None

            local_path = data.get("local_media_path")
            if local_path and not os.path.exists(local_path):
                data["local_media_path"] = None

            self.hits += 1
            logger.info(f"[CACHE_HIT] Competitor artifact cache hit for [{canonical_id}]")
            if profiler:
                try:
                    profiler.increment_metric("cache_hits", 1)
                except Exception:
                    pass
            return data
        except Exception as e:
            logger.warning(f"Failed to read cache for [{canonical_id}]: {e}")
            self.invalidate(canonical_id)
            self.misses += 1
            if profiler:
                try:
                    profiler.increment_metric("cache_misses", 1)
                except Exception:
                    pass
            return None

    def set(
        self,
        candidate_or_id: Any,
        data: Dict[str, Any],
        profiler: Any = None,
    ) -> bool:
        """Store competitor analysis and derived signals in cache."""
        canonical_id = get_canonical_reel_id(candidate_or_id)
        if not canonical_id or canonical_id == "unknown":
            return False

        if not data or not isinstance(data, dict):
            return False

        payload = dict(data)
        payload["canonical_id"] = canonical_id
        payload["cached_at"] = time.time()

        c_file = self._cache_path(canonical_id)
        try:
            temp_file = c_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            temp_file.replace(c_file)
            self.writes += 1
            logger.info(f"[CACHE_WRITE] Competitor artifact cached for [{canonical_id}]")
            return True
        except Exception as e:
            logger.warning(f"Failed to write cache for [{canonical_id}]: {e}")
            return False

    def invalidate(self, candidate_or_id: Any) -> bool:
        """Invalidate and remove a cache entry."""
        canonical_id = get_canonical_reel_id(candidate_or_id)
        if not canonical_id or canonical_id == "unknown":
            return False

        c_file = self._cache_path(canonical_id)
        if c_file.exists():
            try:
                c_file.unlink()
                self.invalidations += 1
                logger.info(f"[CACHE_INVALIDATED] Competitor artifact invalidated for [{canonical_id}]")
                return True
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for [{canonical_id}]: {e}")
        return False
