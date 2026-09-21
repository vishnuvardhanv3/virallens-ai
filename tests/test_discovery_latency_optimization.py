"""
Tests for Discovery Latency Optimization and Resilience:
1. Stop after sufficient candidates
2. Blocked query moves to next query
3. Transient error can retry
4. Confirmed block does not receive unnecessary retry
5. Candidate deduplication still works
6. Query provenance remains intact
7. All generated queries remain derived from the actual profile
8. Creator diversity remains enforced
9. Raw reach ordering remains descending
10. Final verified competitor limit remains unchanged
"""

import pytest
from unittest.mock import MagicMock, patch
from instagram.providers.apify import ApifyProvider, StatusStr
from agents.discovery_agent import DiscoveryAgent
from services.discovery_profile import build_discovery_profile


def _make_dummy_item(shortcode: str, username: str, views: int = 100000, likes: int = 5000):
    return {
        "id": f"id_{shortcode}",
        "reel_id": f"id_{shortcode}",
        "shortcode": shortcode,
        "shortCode": shortcode,
        "creator": username,
        "owner_username": username,
        "ownerUsername": username,
        "views": views,
        "plays": views,
        "reach_value": views,
        "videoViewCount": views,
        "videoPlayCount": views,
        "likes": likes,
        "likesCount": likes,
        "comments": 100,
        "commentsCount": 100,
        "url": f"https://www.instagram.com/reel/{shortcode}/",
        "taken_at": "2026-03-01T12:00:00Z",
    }


def test_1_stop_after_sufficient_candidates():
    """1. Test that discovery stops early as soon as candidate pool reaches sufficiency."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper")
    
    # Generate 32 items with high reach and diverse creators
    items_q1 = [_make_dummy_item(f"sc_{i}", f"user_{i}", views=50000 + i*1000) for i in range(32)]
    
    queries = [
        {"query": "first query", "search_level": "CORE", "provenance": "entity"},
        {"query": "second query", "search_level": "NICHE", "provenance": "niche"},
        {"query": "third query", "search_level": "FALLBACK", "provenance": "style"},
    ]
    
    with patch.object(provider, "_execute_single_query") as mock_exec:
        q_state = {
            "query": "first query",
            "search_level": "CORE",
            "status": StatusStr("SUCCESS"),
            "attempt": 1,
            "attempt_count": 1,
            "results_count": len(items_q1),
            "raw_items": len(items_q1),
            "normalized_items": len(items_q1),
            "duration_seconds": 1.2,
            "telemetry": {},
        }
        mock_exec.return_value = (q_state, items_q1)
        
        res, meta = provider.discover_reels(queries, limit=30)
        
        # Must only call _execute_single_query ONCE
        assert mock_exec.call_count == 1
        assert meta.get("early_stopped") is True
        assert len(meta.get("skipped_queries", [])) == 2
        assert "second query" in meta.get("skipped_queries")
        assert len(res) == 30  # capped at overall_max/limit


def test_2_blocked_query_moves_to_next_query():
    """2. Test that a blocked query moves smoothly to the next query without halting discovery."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper")
    
    items_q2 = [_make_dummy_item(f"sc_{i}", f"user_{i}", views=20000) for i in range(15)]
    
    queries = [
        {"query": "blocked query", "search_level": "CORE", "provenance": "entity"},
        {"query": "succeeding query", "search_level": "NICHE", "provenance": "niche"},
    ]
    
    with patch.object(provider, "_execute_single_query") as mock_exec:
        mock_exec.side_effect = [
            (
                {
                    "query": "blocked query",
                    "search_level": "CORE",
                    "status": StatusStr("BLOCKED"),
                    "attempt": 1,
                    "attempt_count": 1,
                    "results_count": 0,
                    "raw_items": 0,
                    "normalized_items": 0,
                    "duration_seconds": 0.5,
                    "telemetry": {},
                },
                [],
            ),
            (
                {
                    "query": "succeeding query",
                    "search_level": "NICHE",
                    "status": StatusStr("SUCCESS"),
                    "attempt": 1,
                    "attempt_count": 1,
                    "results_count": 15,
                    "raw_items": 15,
                    "normalized_items": 15,
                    "duration_seconds": 1.1,
                    "telemetry": {},
                },
                items_q2,
            ),
        ]
        
        res, meta = provider.discover_reels(queries, limit=30)
        
        assert mock_exec.call_count == 2
        health_str = meta.get("discovery_health", {}).get("discovery_health") if isinstance(meta.get("discovery_health"), dict) else meta.get("discovery_health")
        assert health_str in {"partial", "healthy"}
        assert len(res) == 15
        assert meta["discovery_health_obj"]["queries_blocked"] == 1
        assert meta["discovery_health_obj"]["queries_succeeded"] == 1


def test_3_transient_error_can_retry():
    """3. Test that a genuinely transient error (e.g. rate_limit or server timeout) can retry."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper")
    
    mock_client = MagicMock()
    # First actor call throws a retryable transient error, second succeeds
    mock_run_obj = {
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds_123",
        "statusMessage": "Finished!",
    }
    
    # Mock _call_actor_safely to fail once then succeed
    with patch.object(provider, "_call_actor_safely") as mock_call, \
         patch.object(provider, "_fetch_dataset_safely") as mock_fetch, \
         patch("time.sleep", return_value=None):
        
        mock_call.side_effect = [
            (None, "Connection reset by peer: socket error"),
            (mock_run_obj, None),
        ]
        
        dummy_items = [_make_dummy_item("retry_ok", "creator_a")]
        mock_fetch.return_value = (dummy_items, None)
        
        q_state, cands = provider._execute_single_query(
            client=mock_client,
            actor_id="apify/instagram-search-scraper",
            template={"search": "{query}"},
            search_term="retry query",
            search_level="CORE",
            effective_search_limit=10,
            max_retries=2,
            q_idx=1,
        )
        
        assert q_state["status"] == "SUCCESS"
        assert q_state["retry_count"] == 1
        assert len(cands) == 1


def test_4_confirmed_block_does_not_receive_unnecessary_retry():
    """4. Test that a confirmed platform block (e.g., login required / checkpoint in log) does NOT retry."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper")
    mock_client = MagicMock()
    
    failed_run = {
        "id": "run_blocked_1",
        "status": "FAILED",
        "statusMessage": "Finished! Total 1 requests: 1 succeeded, 0 failed.",
        "defaultDatasetId": "ds_empty",
    }
    
    with patch.object(provider, "_call_actor_safely", return_value=(failed_run, None)) as mock_call, \
         patch.object(provider, "_fetch_dataset_safely", return_value=([], None)), \
         patch.object(provider, "_get_run_log_text", return_value="Search scraper failed to find any results because it was blocked by Instagram"), \
         patch("time.sleep", return_value=None):
        
        q_state, cands = provider._execute_single_query(
            client=mock_client,
            actor_id="apify/instagram-search-scraper",
            template={"search": "{query}"},
            search_term="blocked query",
            search_level="CORE",
            effective_search_limit=10,
            max_retries=2,
            q_idx=1,
        )
        
        assert q_state["status"] == "BLOCKED"
        # Must NOT have retried: exactly 1 attempt
        assert q_state["retry_count"] == 0
        assert q_state["attempt"] == 1
        assert mock_call.call_count == 1


def test_5_candidate_deduplication_still_works():
    """5. Test that duplicate candidates across queries are deduplicated while preserving provenance."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper")
    
    dup_item = _make_dummy_item("duplicate_sc", "creator_shared", views=100000)
    unique_item_1 = _make_dummy_item("unique_1", "creator_1", views=80000)
    unique_item_2 = _make_dummy_item("unique_2", "creator_2", views=90000)
    
    queries = [
        {"query": "q1", "search_level": "CORE", "provenance": "entity"},
        {"query": "q2", "search_level": "NICHE", "provenance": "niche"},
    ]
    
    with patch.object(provider, "_execute_single_query") as mock_exec:
        mock_exec.side_effect = [
            ({"query": "q1", "status": StatusStr("SUCCESS"), "duration_seconds": 1.0, "telemetry": {}}, [dup_item, unique_item_1]),
            ({"query": "q2", "status": StatusStr("SUCCESS"), "duration_seconds": 1.0, "telemetry": {}}, [dup_item, unique_item_2]),
        ]
        
        res, meta = provider.discover_reels(queries, limit=30)
        
        # 3 unique reels in total
        assert len(res) == 3
        sc_list = [c["shortcode"] for c in res]
        assert len(set(sc_list)) == 3
        
        # Find the duplicate and verify it tracked both source queries
        dup_cand = next(c for c in res if c["shortcode"] == "duplicate_sc")
        assert len(dup_cand.get("source_queries", [])) == 2
        assert "q1" in dup_cand["source_queries"]
        assert "q2" in dup_cand["source_queries"]


def test_6_query_provenance_remains_intact():
    """6. Test that candidate records retain query provenance and search level."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper")
    item = _make_dummy_item("sc_prov", "creator_prov")
    item["query_provenance"] = "topic:cars"
    item["search_level"] = "NICHE"
    
    queries = [
        {"query": "provenance query", "search_level": "NICHE", "provenance": "topic:cars"},
    ]
    
    with patch.object(provider, "_execute_single_query") as mock_exec:
        mock_exec.return_value = (
            {"query": "provenance query", "status": StatusStr("SUCCESS"), "duration_seconds": 1.0, "telemetry": {}},
            [item],
        )
        res, _ = provider.discover_reels(queries)
        assert len(res) == 1
        cand = res[0]
        assert cand.get("search_level") == "NICHE"
        assert cand.get("query_provenance") == "topic:cars"
        assert "provenance query" in cand.get("source_queries", [])


def test_7_all_generated_queries_remain_derived_from_actual_profile():
    """7. Test that generated discovery queries strictly originate from actual profile without hard-coded fallbacks."""
    profile = build_discovery_profile({
        "primary_entity": "ferrari 488 pista",
        "entity_type": "automobile",
        "entity_usable_for_discovery": True,
        "entity_confidence": 0.95,
        "topic": "supercar exhaust sound",
        "topic_confidence": 0.9,
        "niche": "automotive enthusiast",
        "sub_niche": "exhaust revs",
        "sub_niche_confidence": 0.85,
        "edit_type": "cinematic car edit",
        "edit_type_confidence": 0.8,
        "content_format": "reel sound showcase",
        "style": "high energy automotive",
    })
    
    res = DiscoveryAgent.generate_queries(profile, max_queries=5)
    queries = res.get("queries", [])
    assert len(queries) > 0
    
    # Assert none of the queries contain hardcoded Spiderman or benchmark keywords
    for q in queries:
        query_text = q["query"].lower()
        assert "spider" not in query_text
        assert "peter" not in query_text
        # Every query must relate to Ferrari, supercar, car edit, or automotive
        matches = any(k in query_text for k in ["ferrari", "supercar", "automotive", "exhaust", "car"])
        assert matches, f"Query '{query_text}' did not derive from profile"


def test_8_creator_diversity_remains_enforced():
    """8. Test that candidate sufficiency checks require creator diversity, not just 30 posts from 1 creator."""
    provider = ApifyProvider(token="mock-token")
    
    # 30 posts but ALL from the same creator
    single_creator_pool = [_make_dummy_item(f"sc_{i}", "monopolist_creator", views=100000) for i in range(30)]
    
    is_sufficient, reason, metrics = provider.evaluate_candidate_sufficiency(
        single_creator_pool, target_count=30, min_unique_creators=5
    )
    # Must NOT be sufficient because unique_creators is 1 (< 5)
    assert is_sufficient is False
    assert metrics["unique_creators"] == 1
    
    # Now create pool with 10 diverse creators
    diverse_pool = [_make_dummy_item(f"sc_{i}", f"creator_{i % 10}", views=100000) for i in range(30)]
    is_sufficient_div, reason_div, metrics_div = provider.evaluate_candidate_sufficiency(
        diverse_pool, target_count=30, min_unique_creators=5
    )
    assert is_sufficient_div is True
    assert metrics_div["unique_creators"] == 10


def test_9_raw_reach_ordering_remains_descending():
    """9. Test that merged candidate pool is strictly sorted by raw reach descending."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper")
    
    items = [
        _make_dummy_item("low", "user_1", views=1000),
        _make_dummy_item("high", "user_2", views=500000),
        _make_dummy_item("med", "user_3", views=50000),
    ]
    
    queries = [{"query": "test", "search_level": "CORE"}]
    with patch.object(provider, "_execute_single_query") as mock_exec:
        mock_exec.return_value = (
            {"query": "test", "status": StatusStr("SUCCESS"), "duration_seconds": 1.0, "telemetry": {}},
            items,
        )
        res, _ = provider.discover_reels(queries, limit=30)
        
        reaches = [c.get("views") or c.get("reach_value", 0) for c in res]
        assert reaches == [500000, 50000, 1000]


def test_10_final_verified_competitor_limit_remains_unchanged():
    """10. Test that the final candidate return cap matches overall_max / settings."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper")
    
    # 40 items returned from query
    items = [_make_dummy_item(f"sc_{i}", f"user_{i}", views=100000 + i) for i in range(40)]
    queries = [{"query": "test", "search_level": "CORE"}]
    
    with patch.object(provider, "_execute_single_query") as mock_exec:
        mock_exec.return_value = (
            {"query": "test", "status": StatusStr("SUCCESS"), "duration_seconds": 1.0, "telemetry": {}},
            items,
        )
        res, _ = provider.discover_reels(queries, limit=25)
        
        # Must be capped at 25
        assert len(res) == 25


def test_11_safe_discovery_caching(tmp_path):
    """11. Test that successful query is cached and returned on repeat with cache_hit=True."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper", cache_dir=str(tmp_path))
    
    raw_items = [
        {"id": f"cached_{i}", "code": f"cached_{i}", "caption": "test", "playCount": 80000, "ownerUsername": f"user_{i}"}
        for i in range(5)
    ]
    query = {"query": "safe test query", "search_level": "CORE", "target_profile_id": "test_profile_1"}
    
    mock_call = {"id": "run-id-123", "status": "SUCCEEDED", "defaultDatasetId": "ds-123"}
    
    with patch("apify_client.ApifyClient"):
        # First call: execute and cache
        with patch.object(provider, "_call_actor_safely", return_value=(mock_call, None)) as mock_call_act, \
             patch.object(provider, "_fetch_dataset_safely", return_value=(raw_items, None)) as mock_ds:
            
            cands, diag = provider.discover_reels([query], limit=10)
            assert diag["success"] is True
            assert len(diag["query_results"]) == 1
            assert diag["query_results"][0]["status"] == "SUCCESS"
            assert diag["query_results"][0]["telemetry"]["cache_hit"] is False
            assert len(cands) == 5
            assert mock_call_act.call_count == 1
        
        # Second call: should hit cache, NOT calling _call_actor_safely or _fetch_dataset_safely
        with patch.object(provider, "_call_actor_safely") as mock_call_act, \
             patch.object(provider, "_fetch_dataset_safely") as mock_ds:
            cands2, diag2 = provider.discover_reels([query], limit=10)
            assert diag2["success"] is True
            assert len(diag2["query_results"]) == 1
            assert diag2["query_results"][0]["status"] == "SUCCESS"
            assert diag2["query_results"][0]["telemetry"]["cache_hit"] is True
            assert diag2["query_results"][0]["telemetry"]["cache_lookup_duration_sec"] is not None
            assert diag2["query_results"][0]["telemetry"]["cache_lookup_duration_sec"] < 0.5
            assert len(cands2) == 5
            assert cands2[0]["shortcode"] == "cached_0"
            assert mock_call_act.call_count == 0
            assert mock_ds.call_count == 0


def test_12_cache_isolation_across_different_queries_and_profiles(tmp_path):
    """12. Test that different queries or different profiles do not hit each other's cache."""
    provider = ApifyProvider(token="mock-token", actor_id="apify/instagram-search-scraper", cache_dir=str(tmp_path))
    
    query_a = {"query": "profile a query", "search_level": "CORE", "target_profile_id": "profile_A"}
    query_b = {"query": "profile b query", "search_level": "CORE", "target_profile_id": "profile_B"}
    query_a_diff_profile = {"query": "profile a query", "search_level": "CORE", "target_profile_id": "profile_C"}
    
    raw_a = [{"id": "item_a", "code": "item_a", "caption": "a", "playCount": 1000, "ownerUsername": "user_a"}]
    raw_b = [{"id": "item_b", "code": "item_b", "caption": "b", "playCount": 2000, "ownerUsername": "user_b"}]
    
    mock_call = {"id": "run-id-123", "status": "SUCCEEDED", "defaultDatasetId": "ds-123"}
    
    with patch("apify_client.ApifyClient"):
        with patch.object(provider, "_call_actor_safely", return_value=(mock_call, None)), \
             patch.object(provider, "_fetch_dataset_safely", return_value=(raw_a, None)):
            cands_a, diag_a = provider.discover_reels([query_a], limit=10)
            assert diag_a["query_results"][0]["status"] == "SUCCESS"
            assert diag_a["query_results"][0]["telemetry"]["cache_hit"] is False
            
        # query_b should NOT hit query_a's cache
        with patch.object(provider, "_call_actor_safely", return_value=(mock_call, None)) as mock_call_act, \
             patch.object(provider, "_fetch_dataset_safely", return_value=(raw_b, None)):
            cands_b, diag_b = provider.discover_reels([query_b], limit=10)
            assert diag_b["query_results"][0]["status"] == "SUCCESS"
            assert diag_b["query_results"][0]["telemetry"]["cache_hit"] is False
            assert mock_call_act.call_count == 1
            
        # query_a with different profile ID should NOT hit profile_A's cache
        with patch.object(provider, "_call_actor_safely", return_value=(mock_call, None)) as mock_call_act, \
             patch.object(provider, "_fetch_dataset_safely", return_value=(raw_a, None)):
            cands_c, diag_c = provider.discover_reels([query_a_diff_profile], limit=10)
            assert diag_c["query_results"][0]["status"] == "SUCCESS"
            assert diag_c["query_results"][0]["telemetry"]["cache_hit"] is False
            assert mock_call_act.call_count == 1


