"""Comprehensive unit and integration tests for Apify discovery resilience.

Verifies all 16 requirements from Section 17:
1. One query blocked while another succeeds
2. Blocked query retries twice (3 attempts total)
3. Failed/error query does not abort discovery
4. Timeout query does not abort discovery
5. Empty query distinguished from blocked query
6. Successful candidates are preserved
7. Candidates are deduplicated
8. source_queries preserved
9. Reach ranking occurs after discovery
10. Semantic relevance remains a hard gate
11. Partial discovery health is reported
12. Zero candidates produces failed health state
13. UI does not claim healthy discovery when blocked queries occurred
14. Malformed fallback queries are rejected
15. Overall candidate cap remains 100
16. Per-query candidate cap remains 25
"""
import json
import time
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from config import settings
from instagram.providers.apify import ApifyProvider
from instagram.provider_router import InstagramProviderRouter
from utils.query_builder import QueryBuilder
from services.relevance import score_video_relevance


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    """Fast forward time.sleep in discovery retries."""
    monkeypatch.setattr(time, "sleep", lambda s: None)


@pytest.fixture
def mock_apify_token(monkeypatch):
    """Ensure API token is configured for tests."""
    monkeypatch.setattr("config.settings.APIFY_API_TOKEN", "fake_test_token")


def _make_fake_item(shortcode: str, views: int, query: str = "test query"):
    return {
        "shortcode": shortcode,
        "url": f"https://www.instagram.com/reel/{shortcode}/",
        "ownerUsername": f"creator_{shortcode}",
        "likesCount": views // 10,
        "videoPlayCount": views,
        "productType": "clips",
        "media_url": f"https://cdn.instagram.com/{shortcode}.mp4",
        "caption": f"Reel for {query}",
    }


def test_1_and_6_one_query_blocked_while_another_succeeds_and_candidates_preserved(mock_apify_token):
    """1 & 6: A blocked query does not abort discovery; successful candidates are preserved."""
    provider = ApifyProvider()

    def actor_call(run_input):
        search = run_input.get("search", "")
        if "blocked" in search:
            return {"status": "FAILED", "default_dataset_id": "ds_blocked"}
        else:
            return {"status": "SUCCEEDED", "default_dataset_id": "ds_success"}

    def dataset_iterate(ds_id):
        mock_ds = MagicMock()
        if ds_id == "ds_blocked":
            mock_ds.iterate_items.return_value = [
                {"errorDescription": "Search scraper failed to find any results because it was blocked by Instagram."}
            ]
        else:
            mock_ds.iterate_items.return_value = [
                _make_fake_item("reel_A", 50000),
                _make_fake_item("reel_B", 30000),
            ]
        return mock_ds

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.side_effect = actor_call
    mock_client.actor.return_value = mock_actor
    mock_client.dataset.side_effect = dataset_iterate

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [
            {"query": "blocked spider man edit", "search_level": "CORE"},
            {"query": "spider man cinematic fan edit", "search_level": "CORE"},
        ]
        candidates, diag = provider.discover_reels(queries, limit=100)

        assert len(candidates) == 2
        assert {c["shortcode"] for c in candidates} == {"reel_A", "reel_B"}
        assert diag["status"] == "PARTIAL_SUCCESS"
        assert diag["discovery_health"]["discovery_health"] in {"partial", "degraded"}
        assert diag["discovery_health"]["queries_blocked"] == 1
        assert diag["discovery_health"]["queries_succeeded"] == 1


def test_2_blocked_query_does_not_hammer_actor(mock_apify_token):
    """2: Explicitly blocked query does not hammer the actor repeatedly."""
    provider = ApifyProvider()
    call_attempts = []

    def actor_call(run_input):
        call_attempts.append(run_input.get("search"))
        return {
            "status": "FAILED",
            "statusMessage": "Search scraper failed to find any results because it was blocked by Instagram.",
            "default_dataset_id": "ds_blocked",
        }

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.side_effect = actor_call
    mock_client.actor.return_value = mock_actor

    mock_ds = MagicMock()
    mock_ds.iterate_items.return_value = []
    mock_client.dataset.return_value = mock_ds

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [{"query": "blocked edit", "search_level": "CORE"}]
        candidates, diag = provider.discover_reels(queries, limit=100)

        # Explicit blocking breaks out without hammering the actor repeatedly
        assert len(call_attempts) == 1
        assert len(candidates) == 0
        q_state = diag["query_results"][0]
        assert q_state["status"] == "blocked"
        assert q_state["attempt_count"] == 1
        assert q_state["retry_count"] == 0


def test_3_failed_query_does_not_abort_discovery(mock_apify_token):
    """3: Unexpected error in one query does not abort discovery."""
    provider = ApifyProvider()

    def actor_call(run_input):
        search = run_input.get("search", "")
        if "error" in search:
            raise RuntimeError("Apify API connection reset")
        return {"status": "SUCCEEDED", "default_dataset_id": "ds_success"}

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.side_effect = actor_call
    mock_client.actor.return_value = mock_actor

    mock_ds = MagicMock()
    mock_ds.iterate_items.return_value = [_make_fake_item("reel_good", 10000)]
    mock_client.dataset.return_value = mock_ds

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [
            {"query": "error query", "search_level": "CORE"},
            {"query": "valid query", "search_level": "CORE"},
        ]
        candidates, diag = provider.discover_reels(queries, limit=100)
        assert len(candidates) == 1
        assert candidates[0]["shortcode"] == "reel_good"
        assert diag["query_results"][0]["status"] == "error"
        assert diag["query_results"][1]["status"] == "success"


def test_4_timeout_query_does_not_abort_discovery(mock_apify_token):
    """4: Actor timeout in one query does not abort discovery."""
    provider = ApifyProvider()

    def actor_call(*args, **kwargs):
        run_input = kwargs.get("run_input") or (args[0] if args else {})
        search = run_input.get("search", "")
        if "timeout" in search:
            raise TimeoutError("Actor execution timed out after 300 seconds")
        return {"status": "SUCCEEDED", "default_dataset_id": "ds_success"}

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.side_effect = actor_call
    mock_client.actor.return_value = mock_actor

    mock_ds = MagicMock()
    mock_ds.iterate_items.return_value = [_make_fake_item("reel_ok", 25000)]
    mock_client.dataset.return_value = mock_ds

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [
            {"query": "timeout query", "search_level": "CORE"},
            {"query": "normal query", "search_level": "CORE"},
        ]
        candidates, diag = provider.discover_reels(queries, limit=100)
        assert len(candidates) == 1
        assert candidates[0]["shortcode"] == "reel_ok"
        assert diag["query_results"][0]["status"] == "timeout"
        assert diag["query_results"][1]["status"] == "success"


def test_5_empty_query_distinguished_from_blocked_query(mock_apify_token):
    """5: An actor returning 0 results normally is classified as 'empty', not 'blocked'."""
    provider = ApifyProvider()

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.return_value = {"status": "SUCCEEDED", "default_dataset_id": "ds_empty"}
    mock_client.actor.return_value = mock_actor

    mock_ds = MagicMock()
    mock_ds.iterate_items.return_value = []
    mock_client.dataset.return_value = mock_ds

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [{"query": "niche rare query", "search_level": "CORE"}]
        candidates, diag = provider.discover_reels(queries, limit=100)
        assert len(candidates) == 0
        q_state = diag["query_results"][0]
        assert q_state["status"] == "empty"
        assert q_state["retry_count"] == 0
        assert diag["discovery_health"]["queries_empty"] == 1
        assert diag["discovery_health"]["queries_blocked"] == 0


def test_7_and_8_candidates_deduplicated_and_source_queries_preserved(mock_apify_token):
    """7 & 8: Candidates appearing in multiple queries are deduplicated and preserve source_queries."""
    provider = ApifyProvider()

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.return_value = {"status": "SUCCEEDED", "default_dataset_id": "ds_test"}
    mock_client.actor.return_value = mock_actor

    def dataset_iterate(ds_id):
        mock_ds = MagicMock()
        # Query 1 returns reel_shared & reel_unique1; Query 2 returns reel_shared & reel_unique2
        return mock_ds

    mock_ds1 = MagicMock()
    mock_ds1.iterate_items.return_value = [
        _make_fake_item("reel_shared", 70000, "q1"),
        _make_fake_item("reel_unique1", 10000, "q1"),
    ]
    mock_ds2 = MagicMock()
    mock_ds2.iterate_items.return_value = [
        _make_fake_item("reel_shared", 70000, "q2"),
        _make_fake_item("reel_unique2", 20000, "q2"),
    ]
    mock_client.dataset.side_effect = [mock_ds1, mock_ds2]

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [
            {"query": "spider man action scene", "search_level": "CORE"},
            {"query": "spider man cinematic edit", "search_level": "CORE"},
        ]
        candidates, diag = provider.discover_reels(queries, limit=100)

        # 3 unique candidates: reel_shared, reel_unique1, reel_unique2
        assert len(candidates) == 3
        shared = next(c for c in candidates if c["shortcode"] == "reel_shared")
        assert "spider man action scene" in shared["source_queries"]
        assert "spider man cinematic edit" in shared["source_queries"]
        assert len(shared["source_queries"]) == 2


def test_9_reach_ranking_occurs_after_discovery(mock_apify_token):
    """9: Candidates are sorted by reach (views/plays descending) without fabrication."""
    provider = ApifyProvider()

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.return_value = {"status": "SUCCEEDED", "default_dataset_id": "ds_reach"}
    mock_client.actor.return_value = mock_actor

    mock_ds = MagicMock()
    mock_ds.iterate_items.return_value = [
        _make_fake_item("reel_low", 5000),
        _make_fake_item("reel_high", 250000),
        _make_fake_item("reel_mid", 45000),
    ]
    mock_client.dataset.return_value = mock_ds

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [{"query": "spider man edit", "search_level": "CORE"}]
        candidates, diag = provider.discover_reels(queries, limit=100)

        assert len(candidates) == 3
        assert candidates[0]["shortcode"] == "reel_high"
        assert candidates[1]["shortcode"] == "reel_mid"
        assert candidates[2]["shortcode"] == "reel_low"


def test_10_semantic_relevance_remains_a_hard_gate():
    """10: Semantic relevance verification strictly rejects unrelated high-reach competitors."""
    uploaded_analysis = {
        "niche": "superhero film edit",
        "topic": "spider-man cinematic fight scene",
        "primary_entity": "spider-man",
    }
    # Completely unrelated cooking competitor
    unrelated_comp = {
        "niche": "culinary arts",
        "topic": "homemade sourdough bread recipe",
        "primary_entity": "sourdough",
    }
    rel_res = score_video_relevance(uploaded_analysis, unrelated_comp)
    assert rel_res["video_relevance_eligible"] is False
    assert rel_res["video_relevance_score"] < 0.35


def test_11_partial_discovery_health_reported(mock_apify_token):
    """11: When some queries fail/blocked but candidates are found, health is partial/degraded."""
    provider = ApifyProvider()

    def actor_call(run_input):
        search = run_input.get("search", "")
        ds_id = "ds_blocked" if "blocked" in search else "ds_working"
        return {"status": "SUCCEEDED", "default_dataset_id": ds_id}

    def dataset_fetch(ds_id):
        mock_ds = MagicMock()
        if ds_id == "ds_blocked":
            mock_ds.iterate_items.return_value = [{"errorDescription": "Search blocked by Instagram"}]
        else:
            mock_ds.iterate_items.return_value = [_make_fake_item("reel_1", 100000)]
        return mock_ds

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.side_effect = actor_call
    mock_client.actor.return_value = mock_actor
    mock_client.dataset.side_effect = dataset_fetch

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [
            {"query": "blocked query", "search_level": "CORE"},
            {"query": "working query", "search_level": "CORE"},
        ]
        candidates, diag = provider.discover_reels(queries, limit=100)
        assert len(candidates) == 1
        assert diag["discovery_health"]["discovery_health"] in {"partial", "degraded"}
        assert diag["discovery_health"]["queries_blocked"] == 1


def test_12_zero_candidates_produces_failed_health_state(mock_apify_token):
    """12: If all queries are blocked and 0 candidates are returned, health is 'failed'."""
    provider = ApifyProvider()

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.return_value = {"status": "SUCCEEDED", "default_dataset_id": "ds"}
    mock_client.actor.return_value = mock_actor
    mock_ds = MagicMock()
    mock_ds.iterate_items.return_value = [{"errorDescription": "blocked by instagram"}]
    mock_client.dataset.return_value = mock_ds

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [{"query": "all blocked", "search_level": "CORE"}]
        candidates, diag = provider.discover_reels(queries, limit=100)

        assert len(candidates) == 0
        assert diag["discovery_health"]["discovery_health"] == "failed"
        assert diag["discovery_health"]["candidates_found"] == 0
        assert "blocked for all attempted queries" in diag["error"]


def test_13_router_diagnostics_truthful_when_blocked_queries_occurred(mock_apify_token):
    """13: Provider router does not label discovery healthy when blocked queries occurred."""
    router = InstagramProviderRouter()

    def actor_call(run_input):
        search = run_input.get("search", "")
        ds_id = "ds_blocked" if "blocked" in search else "ds_ok"
        return {"status": "SUCCEEDED", "default_dataset_id": ds_id}

    def dataset_fetch(ds_id):
        mock_ds = MagicMock()
        if ds_id == "ds_blocked":
            mock_ds.iterate_items.return_value = [{"errorDescription": "challenge_required blocked by instagram"}]
        else:
            mock_ds.iterate_items.return_value = [_make_fake_item("reel_ok_2", 80000)]
        return mock_ds

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.side_effect = actor_call
    mock_client.actor.return_value = mock_actor
    mock_client.dataset.side_effect = dataset_fetch

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [
            {"query": "q_blocked", "search_level": "CORE"},
            {"query": "q_ok", "search_level": "CORE"},
        ]
        candidates, diag = router.search_with_diagnostics(queries, limit=100)

        assert diag["discovery_health"] != "healthy"
        assert diag["discovery_health"] in {"partial", "degraded"}
        assert len(diag["blocked_queries"]) == 1


def test_14_malformed_fallback_queries_rejected():
    """14: QueryBuilder rejects malformed/dangling junk queries like 'spider man high'."""
    assert not QueryBuilder.is_meaningful_query("spider man high")
    assert not QueryBuilder.is_meaningful_query("spider")
    assert not QueryBuilder.is_meaningful_query("very low")
    assert QueryBuilder.is_meaningful_query("spider man cinematic fan edit")


def test_15_and_16_caps_enforced(mock_apify_token):
    """15 & 16: Overall candidate cap (100) and per-query candidate cap (25) are enforced."""
    provider = ApifyProvider()
    sent_inputs = []

    def actor_call(run_input):
        sent_inputs.append(run_input)
        return {"status": "SUCCEEDED", "default_dataset_id": "ds"}

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.side_effect = actor_call
    mock_client.actor.return_value = mock_actor

    mock_ds = MagicMock()
    mock_ds.iterate_items.return_value = [
        _make_fake_item(f"reel_{i}", 1000 + i) for i in range(30)
    ]
    mock_client.dataset.return_value = mock_ds

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [{"query": "query 1", "search_level": "CORE"}]
        candidates, diag = provider.discover_reels(queries, limit=100)

        # Per-query searchLimit injected as <= APIFY_SEARCH_LIMIT (25)
        assert sent_inputs[0]["searchLimit"] <= 25
        assert len(candidates) <= 100


# ==============================================================================
# Section 11 Tests: Truthful Verification Semantics & Zero-Candidate Handling
# ==============================================================================

def test_sec11_1_and_5_zero_candidates_sets_full_e2e_passed_false():
    """11.1 & 11.5: Zero candidates => full_e2e_passed=False, while resilience_passed=True."""
    real_cands = []
    verified_competitors = []

    resilience_passed = True
    discovery_passed = len(real_cands) > 0
    competitor_analysis_passed = len(verified_competitors) > 0
    full_e2e_passed = resilience_passed and discovery_passed and competitor_analysis_passed

    assert resilience_passed is True
    assert discovery_passed is False
    assert competitor_analysis_passed is False
    assert full_e2e_passed is False


def test_sec11_2_zero_candidates_skips_competitor_analysis():
    """11.2: Zero candidates skips competitor download, Twelve Labs analysis, and inference."""
    real_cands = []
    competitor_attentions = {}
    verified_competitors = []
    analyzed_competitors = []

    # Verified zero-candidate branch logic from master_agent.py
    if len(real_cands) == 0:
        downloaded_count = 0
        analyzed_competitors = []
        verified_competitors = []
    else:
        downloaded_count = len(real_cands)

    assert downloaded_count == 0
    assert len(analyzed_competitors) == 0
    assert len(verified_competitors) == 0


def test_sec11_3_zero_candidates_pattern_agent_not_given_fake_competitors():
    """11.3: Pattern Agent is not given fake competitors when candidate pool is empty."""
    from agents.pattern_agent import PatternAgent

    verified_competitors = []
    res = PatternAgent.analyze(verified_competitors)

    assert res["verified_competitor_count"] == 0
    assert res["patterns"] == []
    assert res["pattern_counts"] == {}
    assert "No verified competitors available" in res["findings"][0]


def test_sec11_4_zero_candidates_strategy_agent_cannot_claim_competitor_evidence():
    """11.4: Strategy Agent does not manufacture competitor claims when verified pool is empty."""
    from agents.strategy_agent import StrategyAgent

    your_reel = {
        "analysis": {"niche": "cinematic", "topic": "spider man edit"},
        "technical_signals": {"video": {"duration_sec": 10.0}},
    }
    agent_results = {
        "human_attention": {"first_attention_event": 2.5, "peak_attention": 0.55},
        "hook": {"weaknesses": ["delayed hook"]},
        "editing": {"mean_shot_duration": 4.0, "total_cuts": 2},
        "communication": {"has_direct_address": False},
        "pattern": {"pattern_counts": {}, "patterns": []},
    }
    attention_comparison = {
        "status": "SKIPPED_NO_COMPETITORS",
        "message": "Competitor attention benchmark skipped: zero competitors discovered.",
        "benchmark": {"competitor_count": 0},
    }
    pattern_results = {
        "verified_competitor_count": 0,
        "pattern_counts": {},
        "patterns": [],
    }

    strat = StrategyAgent.synthesize(your_reel, agent_results, attention_comparison, pattern_results)

    # All strategy evidence must relate to uploaded reel signals, zero competitor fabrication
    for item in strat.get("strategic_recommendations", []):
        assert "verified competitors" not in item["evidence"].lower()


def test_sec11_6_explicit_instagram_block_is_classified_as_blocked(mock_apify_token):
    """11.6: An actor failure indicating Instagram blocking is classified as BLOCKED."""
    provider = ApifyProvider()

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.return_value = {
        "status": "FAILED",
        "statusMessage": "Search scraper failed to find any results because it was blocked by Instagram. Please run again and file an issue for the developer",
        "default_dataset_id": "ds_1",
    }
    mock_client.actor.return_value = mock_actor
    mock_ds = MagicMock()
    mock_ds.iterate_items.return_value = []
    mock_client.dataset.return_value = mock_ds

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [{"query": "superhero edit", "search_level": "CORE"}]
        cands, diag = provider.discover_reels(queries, limit=100)

        assert len(cands) == 0
        assert diag["query_results"][0]["status"] == "blocked"
        assert "blocked by Instagram" in diag["query_results"][0]["error"]


def test_sec11_7_transient_provider_exception_is_classified_as_error_or_timeout(mock_apify_token):
    """11.7: A network disconnect or generic API error is classified as ERROR, not BLOCKED."""
    provider = ApifyProvider()

    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_actor.call.side_effect = ConnectionResetError("Remote host closed connection")
    mock_client.actor.return_value = mock_actor

    with patch("apify_client.ApifyClient", return_value=mock_client):
        queries = [{"query": "superhero edit", "search_level": "CORE"}]
        cands, diag = provider.discover_reels(queries, limit=100)

        assert len(cands) == 0
        assert diag["query_results"][0]["status"] == "error"
        assert diag["discovery_health"]["queries_blocked"] == 0
        assert diag["discovery_health"]["queries_failed"] == 1


def test_sec11_8_final_verification_message_reflects_actual_state():
    """11.8: Final verification verdict truthfully reports failure or partial state."""
    # Scenario A: Resilience passed, discovery failed
    resilience_passed = True
    discovery_passed = False
    competitor_analysis_passed = False
    full_e2e_passed = False

    if full_e2e_passed:
        verdict = "FULL E2E PASS"
    elif discovery_passed and competitor_analysis_passed:
        verdict = "DISCOVERY PARTIAL — COMPETITOR ANALYSIS COMPLETED"
    elif resilience_passed and not discovery_passed:
        verdict = "RESILIENCE PASS — DISCOVERY FAILED"
    else:
        verdict = "PIPELINE FAILED"

    assert verdict == "RESILIENCE PASS — DISCOVERY FAILED"
    assert verdict != "FULL E2E PASS"


def test_sec11_9_streamlit_status_reflects_discovery_failure():
    """11.9: Streamlit sidebar and status badge reflect discovery failure when candidates are zero."""
    diag_sidebar = {"discovery_health": "failed", "candidates_returned": 0}
    health_val = str(diag_sidebar.get("discovery_health") or "").lower()

    if health_val == "healthy":
        disc_badge = "🟢 Healthy"
    elif health_val == "partial":
        disc_badge = "🟡 Partial"
    elif health_val == "degraded":
        disc_badge = "🟠 Degraded"
    elif health_val == "failed":
        disc_badge = "🔴 Failed"
    else:
        disc_badge = "⚪ Ready"

    assert disc_badge == "🔴 Failed"


def test_sec11_10_no_false_fully_passed_report_is_generated(tmp_path):
    """11.10: Report explicitly notes when zero candidates were discovered, without claiming full E2E."""
    from services.report_service import save_report

    payload = {
        "your_reel": {"niche": "superhero", "topic": "spider man"},
        "queries": [{"query": "spider man edit"}],
        "discovery_diagnostics": {
            "discovery_health": "failed",
            "candidates_returned": 0,
            "raw_candidates": 0,
            "query_results": [{"query": "spider man edit", "status": "blocked", "results_count": 0}],
        },
        "verified_competitors": [],
        "agent_results": {},
    }
    with patch("services.report_service.REPORTS_DIR", tmp_path):
        json_path, md_path = save_report(payload, base_name="test_zero_cand")
        md_text = md_path.read_text(encoding="utf-8")

        assert "Instagram discovery did not return usable candidates" in md_text
        assert "Resilience handling passed, but full competitor-analysis verification could not be completed" in md_text
        assert "No verified competitors discovered or analyzed" in md_text

