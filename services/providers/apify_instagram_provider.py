"""Apify Instagram Discovery Provider Adapter.

Wraps the production-tested ApifyProvider (apify/instagram-search-scraper) to implement
the canonical DiscoveryProvider interface while preserving 100% production invariance:
- smart blocked-query handling
- unretried blocked queries
- bounded retries on transient network failures
- deterministic caching
- non-fabrication of reach/media URLs
- exact candidate ordering and caps
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from instagram.providers.apify import ApifyProvider
from services.discovery_provider import (
    Candidate,
    DiscoveryProvider,
    ProviderCapabilities,
    ProviderHealthState,
    QueryExecutionResult,
    QueryExecutionState,
)

logger = logging.getLogger(__name__)


class ApifyInstagramProvider(DiscoveryProvider):
    """Production Instagram Reel discovery provider backed by Apify."""

    def __init__(
        self,
        token: Optional[str] = None,
        actor_id: Optional[str] = None,
        cache_dir: Optional[Path | str] = None,
        cache_ttl_seconds: int = 3600,
        enable_cache: Optional[bool] = None,
    ) -> None:
        self._provider = ApifyProvider(
            token=token,
            actor_id=actor_id,
            cache_dir=cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
            enable_cache=enable_cache,
        )

    @property
    def name(self) -> str:
        return "apify"

    @property
    def underlying_provider(self) -> ApifyProvider:
        return self._provider

    def health(self) -> ProviderHealthState:
        """Derive genuine provider health state from configuration and environment."""
        token = self._provider._custom_token or settings.APIFY_API_TOKEN.strip()
        if not token:
            return ProviderHealthState.FAILED
        return ProviderHealthState.HEALTHY

    def capabilities(self) -> ProviderCapabilities:
        """Report actual verified capabilities of the Apify Instagram Search Scraper."""
        return ProviderCapabilities(
            can_search=True,
            can_fetch_metadata=True,
            can_extract_reach=True,
            can_extract_media_url=True,
            can_paginate=True,
            requires_auth=False,
            supports_profile_lookup=True,
        )

    def execute_query(
        self,
        query: str,
        limit: int = 25,
        search_level: str = "unknown",
        target_profile_id: str = "",
        profiler: Any = None,
        **kwargs: Any,
    ) -> QueryExecutionResult:
        """Execute a single search query against Apify returning a QueryExecutionResult."""
        t_start = time.perf_counter()
        
        # Build standard query info struct for underlying ApifyProvider
        q_item = {
            "query": query,
            "search_level": search_level,
            "target_profile_id": target_profile_id,
        }

        # Use the underlying client directly for single query execution
        try:
            from apify_client import ApifyClient
        except ImportError:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            return QueryExecutionResult(
                provider=self.name,
                query=query,
                state=QueryExecutionState.ERROR,
                duration_ms=dur_ms,
                raw_count=0,
                normalized_count=0,
                cache_hit=False,
                attempt=1,
                fallback=False,
                error_type="library_missing",
                message="apify_client package not installed",
            )

        token = self._provider._custom_token or settings.APIFY_API_TOKEN.strip()
        if not token:
            dur_ms = (time.perf_counter() - t_start) * 1000.0
            return QueryExecutionResult(
                provider=self.name,
                query=query,
                state=QueryExecutionState.ERROR,
                duration_ms=dur_ms,
                raw_count=0,
                normalized_count=0,
                cache_hit=False,
                attempt=1,
                fallback=False,
                error_type="unauthorized",
                message="Apify API token not configured",
            )

        client = ApifyClient(token)
        actor_id = self._provider._custom_actor_id or settings.APIFY_INSTAGRAM_REELS_ACTOR.strip()
        
        import json
        try:
            template = json.loads(settings.APIFY_INSTAGRAM_REELS_ACTOR_INPUT_JSON)
        except Exception:
            template = {"search": "{query}", "searchType": "popular", "searchLimit": limit}

        q_state, raw_candidates = self._provider._execute_single_query(
            client=client,
            actor_id=actor_id,
            template=template,
            search_term=query,
            search_level=search_level,
            effective_search_limit=limit,
            max_retries=2,
            q_idx=kwargs.get("q_idx", 1),
            profiler=profiler,
            target_profile_id=target_profile_id,
        )

        dur_ms = (time.perf_counter() - t_start) * 1000.0
        status_str = str(q_state.get("status", "error")).lower()

        state_map = {
            "success": QueryExecutionState.SUCCESS,
            "blocked": QueryExecutionState.BLOCKED,
            "empty": QueryExecutionState.EMPTY,
            "timeout": QueryExecutionState.TIMEOUT,
            "error": QueryExecutionState.ERROR,
        }
        exec_state = state_map.get(status_str, QueryExecutionState.ERROR)

        canonical_cands = [
            Candidate.from_dict(c, provider_name=self.name) for c in raw_candidates
        ]

        error_type = None
        if exec_state == QueryExecutionState.BLOCKED:
            error_type = "platform_block"
        elif exec_state == QueryExecutionState.TIMEOUT:
            error_type = "timeout"
        elif exec_state == QueryExecutionState.ERROR:
            error_type = q_state.get("error") or "actor_error"

        return QueryExecutionResult(
            provider=self.name,
            query=query,
            state=exec_state,
            duration_ms=dur_ms,
            raw_count=len(raw_candidates),
            normalized_count=len(canonical_cands),
            cache_hit=bool(q_state.get("telemetry", {}).get("cache_hit", False)),
            attempt=q_state.get("attempt", 1),
            fallback=False,
            error_type=error_type,
            candidates=canonical_cands,
            message=q_state.get("error"),
            actor_id=q_state.get("actor_id"),
            run_id=q_state.get("run_id"),
            dataset_id=q_state.get("dataset_id"),
            raw_payload=q_state,
        )

    def discover_reels(
        self,
        queries: List[Any],
        limit: int = 100,
        profiler: Any = None,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Delegate directly to underlying ApifyProvider for complete behavioral invariance."""
        return self._provider.discover_reels(
            queries=queries,
            limit=limit,
            profiler=profiler,
            progress_callback=progress_callback,
        )
