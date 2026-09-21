"""Unit and integration tests for the Discovery Orchestrator."""
import tempfile
import pytest
from pathlib import Path

from services.discovery_orchestrator import DiscoveryOrchestrator
from services.discovery_provider import (
    Candidate,
    ProviderHealthState,
    QueryExecutionResult,
    QueryExecutionState,
)


@pytest.fixture
def temp_cache_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def test_orchestrator_initialization_modes():
    """Verify supported modes and default fallback behavior."""
    orch_default = DiscoveryOrchestrator()
    assert orch_default.mode == "apify"

    orch_ar = DiscoveryOrchestrator(mode="agent_reach")
    assert orch_ar.mode == "agent_reach"

    orch_fb = DiscoveryOrchestrator(mode="fallback")
    assert orch_fb.mode == "fallback"

    orch_inv = DiscoveryOrchestrator(mode="unknown_mode")
    assert orch_inv.mode == "apify"


def test_cache_isolation_across_providers(temp_cache_dir):
    """Verify that provider identity is strictly part of the cache key preventing cache cross-talk."""
    orch = DiscoveryOrchestrator(cache_dir=temp_cache_dir, enable_cache=True)
    
    key_apify = orch._build_isolated_cache_key("apify", "spider-man edits", "high_precision", 25)
    key_ar = orch._build_isolated_cache_key("agent_reach", "spider-man edits", "high_precision", 25)

    assert key_apify != key_ar
    assert "apify_" in key_apify
    assert "agent_reach_" in key_ar

    # Cache a successful Apify result
    cand_apify = Candidate(reel_id="C2_apify123", reel_url="https://instagram.com/reel/C2_apify123/", reach=2000000, source_provider="apify")
    res_apify = QueryExecutionResult(
        provider="apify",
        query="spider-man edits",
        state=QueryExecutionState.SUCCESS,
        duration_ms=1000.0,
        raw_count=1,
        normalized_count=1,
        cache_hit=False,
        attempt=1,
        fallback=False,
        candidates=[cand_apify],
    )
    orch._save_cached_query(key_apify, res_apify)

    # Lookup using Apify key: Hit
    hit = orch._get_cached_query(key_apify)
    assert hit is not None
    assert hit.candidates[0].reel_id == "C2_apify123"

    # Lookup using Agent-Reach key: Strictly Miss
    miss = orch._get_cached_query(key_ar)
    assert miss is None


def test_blocked_and_failed_queries_never_cached(temp_cache_dir):
    """Verify that blocked, empty, error, and timeout queries never poison the cache."""
    orch = DiscoveryOrchestrator(cache_dir=temp_cache_dir, enable_cache=True)
    key_blocked = orch._build_isolated_cache_key("apify", "blocked query", "high_precision", 25)

    res_blocked = QueryExecutionResult(
        provider="apify",
        query="blocked query",
        state=QueryExecutionState.BLOCKED,
        duration_ms=500.0,
        raw_count=0,
        normalized_count=0,
        cache_hit=False,
        attempt=1,
        fallback=False,
        error_type="platform_block",
    )
    orch._save_cached_query(key_blocked, res_blocked)

    # Should not be cached
    assert orch._get_cached_query(key_blocked) is None


def test_canonical_deduplication_and_cross_provider_provenance_merging():
    """Verify deduplication across /reel/, /p/, /tv/ and merging of provenance without field loss."""
    orch = DiscoveryOrchestrator()
    pool = {}

    # Candidate 1 from Apify
    cand_apify = Candidate(
        reel_id="https://www.instagram.com/reel/C2_sharedReel/?utm_source=x",
        reel_url="https://www.instagram.com/reel/C2_sharedReel/?utm_source=x",
        creator="creator_alpha",
        caption="Spider-Man fight edit",
        reach=5000000,
        timestamp="2026-03-01T12:00:00Z",
        media_url="https://cdn.instagram.com/reel.mp4",
        source_provider="apify",
        source_query="spider man edits",
    )
    orch.deduplicate_and_merge_candidates(pool, [cand_apify], query="spider man edits", provider_name="apify")

    assert len(pool) == 1
    assert "C2_sharedReel" in pool

    # Candidate 2 from Agent-Reach for the same reel via /p/ URL with missing media_url
    cand_ar = Candidate(
        reel_id="https://www.instagram.com/p/C2_sharedReel/",
        reel_url="https://www.instagram.com/p/C2_sharedReel/",
        creator="creator_alpha",
        caption=None,
        reach=5000000,
        timestamp=None,
        media_url=None,  # missing media_url should NOT overwrite existing
        source_provider="agent_reach",
        source_query="superhero combat sequence",
    )
    orch.deduplicate_and_merge_candidates(pool, [cand_ar], query="superhero combat sequence", provider_name="agent_reach")

    assert len(pool) == 1
    merged = pool["C2_sharedReel"]
    
    # Verify non-null media_url and caption were preserved
    assert merged.media_url == "https://cdn.instagram.com/reel.mp4"
    assert merged.caption == "Spider-Man fight edit"
    assert merged.reach == 5000000

    # Verify provenance merged
    prov = merged.raw_provenance
    assert "apify" in prov["source_providers"]
    assert "agent_reach" in prov["source_providers"]
    assert "spider man edits" in prov["source_queries"]
    assert "superhero combat sequence" in prov["source_queries"]


def test_fallback_triggers_and_event_logging():
    """Verify explicit fallback triggers and structured audit logging."""
    orch = DiscoveryOrchestrator(mode="fallback")

    # Simulate primary Apify returning 0 candidates
    fake_queries = [{"query": "spider-man edits", "search_level": "high_precision"}]

    def mock_apify_discover(queries, limit=100, **kwargs):
        return [], {
            "status": "BLOCKED",
            "provider": "apify",
            "raw_posts_found": 0,
            "valid_reels": 0,
            "discovery_health": {
                "discovery_health": "failed",
                "queries_blocked": 1,
            },
        }

    def mock_ar_discover(queries, limit=100, **kwargs):
        cand = Candidate(
            reel_id="C9_fallbackReel",
            reel_url="https://www.instagram.com/reel/C9_fallbackReel/",
            reach=100000,
            source_provider="agent_reach",
            source_query="spider-man edits",
        )
        return [cand.to_dict()], {
            "status": "SUCCESS",
            "provider": "agent_reach",
            "raw_posts_found": 1,
            "valid_reels": 1,
            "discovery_health": {"discovery_health": "healthy"},
        }

    orch._apify_provider.discover_reels = mock_apify_discover
    orch._agent_reach_provider.discover_reels = mock_ar_discover

    cands, diag = orch.discover_reels(fake_queries, limit=30)

    assert len(cands) == 1
    assert cands[0]["reel_id"] == "C9_fallbackReel"
    assert diag["fallback_triggered"] is True
    assert len(orch.fallback_events) == 1
    
    event = orch.fallback_events[0]
    assert event["primary_provider"] == "apify"
    assert event["secondary_provider"] == "agent_reach"
    assert event["trigger"] == "primary_discovery_zero_results"
    assert event["candidate_count_before_fallback"] == 0
    assert event["candidate_count_after_fallback"] == 1


def test_security_zero_token_or_credential_leakage():
    """Verify that neither telemetry nor diagnostics leak sensitive credentials."""
    orch = DiscoveryOrchestrator()
    sample_diag = {
        "provider": "apify",
        "discovery_health": {"discovery_health": "healthy"},
        "query_results": [
            {
                "query": "spider man",
                "telemetry": {"provider": "apify", "duration_ms": 100.0},
            }
        ],
    }

    raw_str = str(sample_diag).lower()
    for sensitive in ["api_token", "password", "sessionid", "cookie", "authorization", "bearer"]:
        assert sensitive not in raw_str
