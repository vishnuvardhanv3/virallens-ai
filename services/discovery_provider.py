"""Discovery Provider Contract and Canonical Data Model for ViralLens AI.

Formal abstraction layer separating discovery providers (Apify, Agent-Reach, etc.)
from downstream analytical, ranking, and verification engines.

Strict Production Invariance Rules:
1. Missing values = None.
2. Never use 0 to represent unknown reach.
3. Never fabricate media URLs or download URLs.
4. Never infer views from likes, comments, or followers.
5. Never infer timestamps or creator identity.
6. Downstream ranking, semantic verification, and intelligence remain invariant.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse, parse_qs


class ProviderHealthState(str, Enum):
    """Normalized health state of a discovery provider."""
    HEALTHY = "healthy"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


class QueryExecutionState(str, Enum):
    """Execution state of an individual discovery query."""
    SUCCESS = "success"
    BLOCKED = "blocked"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ProviderCapabilities:
    """Explicitly reported capabilities of a discovery provider.
    
    Reports actual capabilities based on current provider implementation,
    never aspirational or hardcoded assumptions.
    """
    can_search: bool = False
    can_fetch_metadata: bool = False
    can_extract_reach: bool = False
    can_extract_media_url: bool = False
    can_paginate: bool = False
    requires_auth: bool = False
    supports_profile_lookup: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


_CANONICAL_REEL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def extract_canonical_shortcode(url_or_id: Any) -> str:
    """Extract standard Instagram shortcode from URL or string identifier.
    
    Handles:
    - /reel/ABC/
    - /p/ABC/
    - /tv/ABC/
    - ?utm_source=... query stripping
    - Raw shortcodes (5 to 35 alphanum characters)
    """
    if not url_or_id:
        return ""
    text = str(url_or_id).strip()
    match = _CANONICAL_REEL_RE.search(text)
    if match:
        return match.group(1)
    clean = re.sub(r"[^A-Za-z0-9_-]", "", text)
    return clean if 5 <= len(clean) <= 35 else ""


def build_canonical_reel_url(url_or_id: Any) -> str:
    """Normalize any Instagram post/reel URL or shortcode to standard /reel/ URL."""
    shortcode = extract_canonical_shortcode(url_or_id)
    if shortcode:
        return f"https://www.instagram.com/reel/{shortcode}/"
    return ""


def safe_int_reach(val: Any) -> Optional[int]:
    """Strict reach extraction: returns positive integer views/plays or None.
    
    Never converts missing/zero to fabricated reach.
    Never uses likes, comments, or followers.
    """
    if val is None or val == "":
        return None
    try:
        iv = int(val)
        return iv if iv > 0 else None
    except (ValueError, TypeError):
        return None


@dataclass
class Candidate:
    """Canonical Reel discovery candidate.
    
    Adheres strictly to the ViralLens canonical contract:
    - missing values = None
    - reach = int views or None (never 0, never synthetic)
    - media_url = verified URL or None (never fabricated)
    - timestamp = genuine timestamp or None
    - creator = username or None
    """
    reel_id: str
    reel_url: str
    creator: Optional[str] = None
    caption: Optional[str] = None
    reach: Optional[int] = None
    timestamp: Optional[str] = None
    media_url: Optional[str] = None
    source_provider: str = "unknown"
    source_query: Optional[str] = None
    metadata_completeness: float = 0.0
    raw_provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce canonical Reel ID & URL
        shortcode = extract_canonical_shortcode(self.reel_id) or extract_canonical_shortcode(self.reel_url)
        if shortcode:
            self.reel_id = shortcode
            self.reel_url = f"https://www.instagram.com/reel/{shortcode}/"
        elif not self.reel_url.startswith("http"):
            self.reel_url = build_canonical_reel_url(self.reel_id) or self.reel_url

        # Reach integrity: strictly None if not a positive integer
        self.reach = safe_int_reach(self.reach)

        # Sanitize creator
        if self.creator and self.creator.strip().lower() in {"unknown", "none", "null", ""}:
            self.creator = None

        # Calculate metadata completeness if not provided
        if self.metadata_completeness <= 0.0:
            self.metadata_completeness = self.compute_completeness()

    def compute_completeness(self) -> float:
        """Compute proportion of core metadata fields present [0.0 - 1.0]."""
        fields = [
            bool(self.reel_id),
            bool(self.reel_url),
            bool(self.creator),
            bool(self.caption),
            self.reach is not None,
            bool(self.timestamp),
            bool(self.media_url),
        ]
        return round(sum(fields) / len(fields), 3)

    def to_dict(self) -> Dict[str, Any]:
        """Produce backward-compatible dictionary representation for downstream agents.
        
        Ensures 100% backward compatibility with existing tests, relevance filters,
        and competitor intelligence algorithms expecting legacy keys.
        """
        shortcode = self.reel_id
        views = self.reach
        src_query = self.source_query or ""
        src_queries = self.raw_provenance.get("source_queries", [src_query] if src_query else [])
        src_providers = self.raw_provenance.get("source_providers", [self.source_provider])

        return {
            "id": shortcode,
            "reel_id": shortcode,
            "url": self.reel_url,
            "shortcode": shortcode,
            "creator": self.creator or "unknown",
            "username": self.creator or "unknown",
            "caption": self.caption,
            "reach": views,
            "reach_value": views,
            "reach_metric": "views" if views is not None else "unavailable",
            "views": views,
            "plays": views,
            "likes": self.raw_provenance.get("likes"),
            "comments": self.raw_provenance.get("comments"),
            "media_url": self.media_url,
            "thumbnail_url": self.raw_provenance.get("thumbnail_url"),
            "timestamp": self.timestamp,
            "source_query": src_query or None,
            "source_queries": src_queries,
            "discovery_query": src_query or None,
            "source_provider": self.source_provider,
            "source_providers": src_providers,
            "provider": self.source_provider,
            "source": self.source_provider,
            "search_level": self.raw_provenance.get("search_level", "unknown"),
            "source_search_level": self.raw_provenance.get("search_level", "unknown"),
            "metadata_completeness": self.metadata_completeness,
            "raw_provenance": self.raw_provenance,
            "status": "discovered",
            "media_status": "not_fetched" if not self.media_url else "available",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], provider_name: str = "unknown") -> Candidate:
        """Create a Candidate instance from a legacy dictionary."""
        r_id = str(data.get("reel_id") or data.get("shortcode") or data.get("id") or "")
        r_url = str(data.get("url") or data.get("reel_url") or "")
        creator = data.get("creator") or data.get("username")
        caption = data.get("caption")
        reach = (
            data.get("reach")
            if data.get("reach") is not None
            else data.get("reach_value")
            if data.get("reach_value") is not None
            else data.get("views")
            if data.get("views") is not None
            else data.get("plays")
        )
        timestamp = data.get("timestamp")
        media_url = data.get("media_url") or data.get("video_url") or data.get("videoUrl")
        source_provider = str(data.get("source_provider") or data.get("provider") or provider_name)
        source_query = data.get("source_query") or data.get("discovery_query")

        provenance = dict(data.get("raw_provenance") or {})
        if "likes" not in provenance and data.get("likes") is not None:
            provenance["likes"] = data.get("likes")
        if "comments" not in provenance and data.get("comments") is not None:
            provenance["comments"] = data.get("comments")
        if "thumbnail_url" not in provenance and data.get("thumbnail_url") is not None:
            provenance["thumbnail_url"] = data.get("thumbnail_url")
        if "search_level" not in provenance and data.get("search_level") is not None:
            provenance["search_level"] = data.get("search_level")
        if "source_queries" not in provenance:
            provenance["source_queries"] = data.get("source_queries", [source_query] if source_query else [])
        if "source_providers" not in provenance:
            provenance["source_providers"] = data.get("source_providers", [source_provider])

        return cls(
            reel_id=r_id,
            reel_url=r_url,
            creator=str(creator) if creator else None,
            caption=str(caption) if caption is not None else None,
            reach=safe_int_reach(reach),
            timestamp=str(timestamp) if timestamp else None,
            media_url=str(media_url) if media_url else None,
            source_provider=source_provider,
            source_query=str(source_query) if source_query else None,
            metadata_completeness=float(data.get("metadata_completeness", 0.0)),
            raw_provenance=provenance,
        )


@dataclass
class QueryExecutionResult:
    """Execution telemetry and candidate container for a single query."""
    provider: str
    query: str
    state: QueryExecutionState
    duration_ms: float
    raw_count: int
    normalized_count: int
    cache_hit: bool
    attempt: int
    fallback: bool
    error_type: Optional[str] = None
    actor_id: Optional[str] = None
    run_id: Optional[str] = None
    dataset_id: Optional[str] = None
    candidates: List[Candidate] = field(default_factory=list)
    message: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None

    def to_telemetry(self) -> Dict[str, Any]:
        """Convert to standard telemetry payload matching Section 14."""
        return {
            "provider": self.provider,
            "query": self.query,
            "state": self.state.value,
            "duration_ms": round(self.duration_ms, 1),
            "raw_count": self.raw_count,
            "normalized_count": self.normalized_count,
            "cache_hit": self.cache_hit,
            "attempt": self.attempt,
            "fallback": self.fallback,
            "error_type": self.error_type,
            "actor_id": self.actor_id,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        res = self.to_telemetry()
        res["candidates"] = [c.to_dict() for c in self.candidates]
        res["message"] = self.message
        return res


class DiscoveryProvider(ABC):
    """Abstract Base Class defining the formal Discovery Provider interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'apify', 'agent_reach')."""
        raise NotImplementedError

    @abstractmethod
    def health(self) -> ProviderHealthState:
        """Derive real provider health state from configuration and operational status."""
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return genuine supported capabilities of this provider."""
        raise NotImplementedError

    @abstractmethod
    def execute_query(
        self,
        query: str,
        limit: int = 25,
        search_level: str = "unknown",
        target_profile_id: str = "",
        profiler: Any = None,
        **kwargs: Any,
    ) -> QueryExecutionResult:
        """Execute a single search query returning canonical QueryExecutionResult."""
        raise NotImplementedError

    @abstractmethod
    def discover_reels(
        self,
        queries: List[Any],
        limit: int = 100,
        profiler: Any = None,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute multi-query discovery returning candidate dictionaries and diagnostics."""
        raise NotImplementedError
