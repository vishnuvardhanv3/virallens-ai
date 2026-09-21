"""Discovery Orchestrator for ViralLens AI.

Orchestrates multi-query Instagram competitor discovery across providers (Apify, Agent-Reach)
with strict architectural guarantees:
- Configurable modes: 'apify' (default), 'agent_reach', 'fallback'
- Cache isolation across providers
- Canonical shortcode / URL deduplication
- Cross-provider provenance merging (source_providers, source_queries)
- Non-fabrication of reach (views/plays or None)
- Explicit, structured fallback logging
- Complete telemetry reporting
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import settings
from services.discovery_provider import (
    Candidate,
    DiscoveryProvider,
    ProviderHealthState,
    QueryExecutionResult,
    QueryExecutionState,
    build_canonical_reel_url,
    extract_canonical_shortcode,
    safe_int_reach,
)
from services.providers.agent_reach_instagram_provider import AgentReachInstagramProvider
from services.providers.apify_instagram_provider import ApifyInstagramProvider

logger = logging.getLogger(__name__)


class DiscoveryOrchestrator:
    """Orchestrates discovery across providers with strict isolation, caching, and fallback."""

    def __init__(
        self,
        mode: Optional[str] = None,
        cache_dir: Optional[Path | str] = None,
        enable_cache: Optional[bool] = None,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self.mode = (mode or getattr(settings, "DISCOVERY_PROVIDER", "apify")).strip().lower()
        if self.mode not in {"apify", "agent_reach", "fallback"}:
            logger.warning(f"Unrecognized discovery mode '{self.mode}', defaulting to 'apify'")
            self.mode = "apify"

        base_cache = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir = base_cache / "discovery_orchestrator"
        self.cache_ttl_seconds = cache_ttl_seconds

        if enable_cache is None:
            # Disable cache in pytest unless explicit directory is provided
            if "PYTEST_CURRENT_TEST" in os.environ and cache_dir is None:
                self.enable_cache = False
            else:
                self.enable_cache = True
        else:
            self.enable_cache = enable_cache

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Providers
        self._apify_provider = ApifyInstagramProvider()
        self._agent_reach_provider = AgentReachInstagramProvider()

        # Telemetry and fallback logs
        self.fallback_events: List[Dict[str, Any]] = []

    def get_provider(self, name: str) -> Optional[DiscoveryProvider]:
        """Resolve a provider instance by name."""
        clean = name.strip().lower()
        if clean == "apify":
            return self._apify_provider
        if clean in {"agent_reach", "agentreach", "opencli"}:
            return self._agent_reach_provider
        return None

    def get_active_providers(self) -> List[DiscoveryProvider]:
        """Return the list of providers active under current mode."""
        if self.mode == "agent_reach":
            return [self._agent_reach_provider]
        if self.mode == "fallback":
            return [self._apify_provider, self._agent_reach_provider]
        return [self._apify_provider]

    def _build_isolated_cache_key(
        self,
        provider_name: str,
        query: str,
        search_level: str,
        limit: int,
        target_profile_id: str = "",
    ) -> str:
        """Construct deterministic cache key strictly isolating providers."""
        norm_query = re.sub(r"\s+", " ", query.strip().lower())
        key_payload = {
            "provider": provider_name.strip().lower(),
            "query": norm_query,
            "search_level": search_level.strip().lower(),
            "limit": limit,
            "profile_id": target_profile_id.strip(),
        }
        raw_bytes = json.dumps(key_payload, sort_keys=True).encode("utf-8")
        return f"{provider_name}_{hashlib.sha256(raw_bytes).hexdigest()[:20]}"

    def _get_cached_query(self, cache_key: str) -> Optional[QueryExecutionResult]:
        """Retrieve cached successful query result if valid and within TTL."""
        if not self.enable_cache:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("cached_at", 0) > self.cache_ttl_seconds:
                return None
            
            cands = [Candidate.from_dict(c) for c in data.get("candidates", [])]
            return QueryExecutionResult(
                provider=data.get("provider", "unknown"),
                query=data.get("query", ""),
                state=QueryExecutionState(data.get("state", "success")),
                duration_ms=data.get("duration_ms", 0.0),
                raw_count=data.get("raw_count", len(cands)),
                normalized_count=len(cands),
                cache_hit=True,
                attempt=data.get("attempt", 1),
                fallback=data.get("fallback", False),
                error_type=data.get("error_type"),
                candidates=cands,
                message=data.get("message"),
                raw_payload=data.get("raw_payload"),
            )
        except Exception as exc:
            logger.debug(f"[Discovery Cache Read Error] {exc}")
            return None

    def _save_cached_query(self, cache_key: str, result: QueryExecutionResult) -> None:
        """Cache only genuine successful results with non-empty candidates."""
        if not self.enable_cache:
            return
        if result.state != QueryExecutionState.SUCCESS or not result.candidates:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            payload = {
                "cached_at": time.time(),
                "provider": result.provider,
                "query": result.query,
                "state": result.state.value,
                "duration_ms": result.duration_ms,
                "raw_count": result.raw_count,
                "normalized_count": result.normalized_count,
                "attempt": result.attempt,
                "fallback": result.fallback,
                "error_type": result.error_type,
                "candidates": [c.to_dict() for c in result.candidates],
                "message": result.message,
                "raw_payload": result.raw_payload,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.debug(f"[Discovery Cache Write Error] {exc}")

    def deduplicate_and_merge_candidates(
        self,
        pool: Dict[str, Candidate],
        new_candidates: List[Candidate],
        query: str,
        provider_name: str,
    ) -> None:
        """Deduplicate candidates by canonical Reel ID/shortcode and merge provenance.
        
        Section 10 & 11:
        - Resolves /reel/ABC/, /p/ABC/, /tv/ABC/ to one canonical identity
        - Merges source_providers and source_queries
        - Never overwrites useful non-null fields with null
        - Preserves raw provider metadata
        """
        for cand in new_candidates:
            canonical_id = extract_canonical_shortcode(cand.reel_id) or cand.reel_id
            if not canonical_id:
                continue

            if canonical_id in pool:
                existing = pool[canonical_id]
                
                # Merge provenance
                ex_prov = existing.raw_provenance
                new_prov = cand.raw_provenance

                # Merge providers list
                providers = set(ex_prov.get("source_providers", [existing.source_provider]))
                providers.add(provider_name)
                providers.update(new_prov.get("source_providers", []))
                ex_prov["source_providers"] = sorted(list(providers))

                # Merge queries list
                queries = set(ex_prov.get("source_queries", [existing.source_query] if existing.source_query else []))
                if query:
                    queries.add(query)
                queries.update(new_prov.get("source_queries", []))
                ex_prov["source_queries"] = sorted(list(queries))

                # Never overwrite useful non-null fields with null
                if not existing.creator and cand.creator:
                    existing.creator = cand.creator
                if not existing.caption and cand.caption:
                    existing.caption = cand.caption
                if existing.reach is None and cand.reach is not None:
                    existing.reach = cand.reach
                if not existing.timestamp and cand.timestamp:
                    existing.timestamp = cand.timestamp
                if not existing.media_url and cand.media_url:
                    existing.media_url = cand.media_url

                # Merge raw dictionary snapshots
                ex_prov.setdefault("provider_snapshots", {})[provider_name] = new_prov
                existing.metadata_completeness = existing.compute_completeness()
            else:
                cand.reel_id = canonical_id
                cand.source_provider = provider_name
                cand.source_query = query
                cand.raw_provenance.setdefault("source_providers", [provider_name])
                cand.raw_provenance.setdefault("source_queries", [query] if query else [])
                pool[canonical_id] = cand

    def discover_reels(
        self,
        queries: List[Any],
        limit: int = 100,
        profiler: Any = None,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Multi-query discovery dispatching to providers according to configured mode.
        
        Preserves 100% production invariance for mode='apify':
        Directly delegates to the ApifyProvider, matching all existing tests and behaviors.
        """
        # When running in pure apify mode, directly delegate to ApifyProvider
        # to ensure 100% invariance with existing tests and behavior.
        if self.mode == "apify":
            return self._apify_provider.discover_reels(
                queries=queries,
                limit=limit,
                profiler=profiler,
                progress_callback=progress_callback,
                **kwargs,
            )

        if self.mode == "agent_reach":
            return self._agent_reach_provider.discover_reels(
                queries=queries,
                limit=limit,
                profiler=profiler,
                progress_callback=progress_callback,
                **kwargs,
            )

        # Mode == 'fallback': primary (Apify) first, secondary (Agent-Reach) only on explicit fallback
        primary = self._apify_provider
        secondary = self._agent_reach_provider
        target_candidates = getattr(settings, "TARGET_DISCOVERY_CANDIDATES", 30)

        # 1. Execute Primary
        p_cands, p_diag = primary.discover_reels(
            queries=queries,
            limit=limit,
            profiler=profiler,
            progress_callback=progress_callback,
            **kwargs,
        )

        p_health = p_diag.get("discovery_health", {})
        p_health_str = p_health.get("discovery_health", "unknown") if isinstance(p_health, dict) else str(p_health)
        should_fallback = False
        fallback_trigger = ""

        # Check explicit fallback triggers (Section 9)
        # Note: A single blocked query is NOT sufficient by itself to switch!
        if primary.health() == ProviderHealthState.FAILED:
            should_fallback = True
            fallback_trigger = "primary_provider_failed"
        elif p_health_str == "failed" and len(p_cands) == 0:
            should_fallback = True
            fallback_trigger = "primary_discovery_zero_results"
        elif len(p_cands) < target_candidates and p_health_str in {"partial", "failed"}:
            should_fallback = True
            fallback_trigger = "primary_candidate_budget_exhausted"

        if not should_fallback:
            return p_cands, p_diag

        # 2. Execute Fallback to Secondary
        if progress_callback:
            try:
                progress_callback(f"[Discovery Fallback] Switching from {primary.name} to {secondary.name} ({fallback_trigger})")
            except Exception:
                pass

        fb_event = {
            "primary_provider": primary.name,
            "secondary_provider": secondary.name,
            "trigger": fallback_trigger,
            "candidate_count_before_fallback": len(p_cands),
            "timestamp": time.time(),
        }

        s_cands, s_diag = secondary.discover_reels(
            queries=queries,
            limit=limit,
            profiler=profiler,
            progress_callback=progress_callback,
            **kwargs,
        )

        fb_event["candidate_count_after_fallback"] = len(p_cands) + len(s_cands)
        self.fallback_events.append(fb_event)

        # Merge candidate pools with canonical deduplication & provenance
        merged_pool: Dict[str, Candidate] = {}
        for c_dict in p_cands:
            cand = Candidate.from_dict(c_dict, provider_name=primary.name)
            self.deduplicate_and_merge_candidates(
                merged_pool, [cand], query=cand.source_query or "", provider_name=primary.name
            )

        for c_dict in s_cands:
            cand = Candidate.from_dict(c_dict, provider_name=secondary.name)
            self.deduplicate_and_merge_candidates(
                merged_pool, [cand], query=cand.source_query or "", provider_name=secondary.name
            )

        final_cands = [c.to_dict() for c in merged_pool.values()]
        
        # Sort by reach descending (strictly preserving actual views/plays)
        final_cands.sort(
            key=lambda c: (1 if (c.get("reach_value") or 0) > 0 else 0, c.get("reach_value") or 0),
            reverse=True,
        )
        final_cands = final_cands[:limit]

        # Combine diagnostics
        diag = dict(p_diag)
        diag["fallback_triggered"] = True
        diag["fallback_event"] = fb_event
        diag["candidates_returned"] = len(final_cands)
        diag["valid_reels"] = len(final_cands)
        diag["providers_used"] = [primary.name, secondary.name]
        
        return final_cands, diag
