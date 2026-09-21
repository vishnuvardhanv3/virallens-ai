"""Router that dispatches discovery requests to configured providers."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from config import settings
from instagram.providers.apify import ApifyProvider
from instagram.providers.base import DiscoveryResult
from services.discovery_orchestrator import DiscoveryOrchestrator
from services.providers.agent_reach_instagram_provider import AgentReachInstagramProvider


class InstagramProviderRouter:
    """Dispatches discovery requests to active providers."""

    def __init__(self, orchestrator: DiscoveryOrchestrator | None = None) -> None:
        self.providers = {
            "apify": ApifyProvider(),
            "agent_reach": AgentReachInstagramProvider(),
        }
        self.orchestrator = orchestrator or DiscoveryOrchestrator()

    def get_provider(self, name: str | None = None):
        target = (name or getattr(settings, "DISCOVERY_PROVIDER", None) or settings.INSTAGRAM_PROVIDER_ORDER.split(",")[0]).strip().lower()
        if target == "fallback":
            return self.providers.get("apify")
        return self.providers.get(target) or self.providers.get("apify")

    def search_with_diagnostics(
        self,
        queries: List[Any],
        limit: int | None = None,
        profiler: Any = None,
        progress_callback: Any = None,
    ) -> DiscoveryResult:
        max_limit = limit or settings.INSTAGRAM_MAX_RESULTS
        mode = getattr(settings, "DISCOVERY_PROVIDER", "apify").strip().lower()
        
        if mode == "fallback":
            candidates, diag = self.orchestrator.discover_reels(
                queries, max_limit, profiler=profiler, progress_callback=progress_callback
            )
            provider_name = diag.get("provider", "apify")
            provider_obj_name = "fallback"
        else:
            provider = self.get_provider()
            if not provider:
                return (
                    [],
                    {
                        "success": False,
                        "status": "PROVIDER_NOT_FOUND",
                        "error": "Configured Instagram discovery provider not available",
                        "candidates_returned": 0,
                        "raw_candidates": 0,
                    },
                )

            try:
                candidates, diag = provider.discover_reels(
                    queries, max_limit, profiler=profiler, progress_callback=progress_callback
                )
            except TypeError:
                try:
                    candidates, diag = provider.discover_reels(queries, max_limit, profiler=profiler)
                except TypeError:
                    candidates, diag = provider.discover_reels(queries, max_limit)
            provider_name = getattr(provider, "name", "apify")
            provider_obj_name = provider_name

        disc_health = diag.get("discovery_health") or {}
        diagnostics = {
            "status": diag.get("status", "SUCCESS" if candidates else "NO_RESULTS"),
            "provider": provider_name,
            "queries_attempted": queries,
            "raw_candidates": diag.get("raw_posts_found", len(candidates)),
            "candidates_returned": len(candidates),
            "valid_reels": len(candidates),
            "discovery_health": disc_health.get("discovery_health", "healthy" if candidates else "failed") if isinstance(disc_health, dict) else str(disc_health),
            "discovery_health_details": disc_health,
            "query_results": diag.get("query_results", []),
            "successful_queries": disc_health.get("successful_queries", []) if isinstance(disc_health, dict) else [],
            "blocked_queries": disc_health.get("blocked_queries", []) if isinstance(disc_health, dict) else [],
            "failed_queries": disc_health.get("failed_queries", []) if isinstance(disc_health, dict) else [],
            "provider_diagnostics": {
                provider_obj_name: {
                    "attempted": True,
                    "status": diag.get("status", "UNKNOWN"),
                    "candidates_returned": len(candidates),
                    "error": diag.get("error"),
                    "query_results": diag.get("query_results", []),
                    "discovery_health": disc_health,
                }
            },
            "error": diag.get("error"),
        }
        if diag.get("fallback_triggered"):
            diagnostics["fallback_triggered"] = True
            diagnostics["fallback_event"] = diag.get("fallback_event")
        return candidates, diagnostics
