"""Tests for Discovery Entity/Topic Extraction and Adaptive Apify Search.

Validates all Section 12 requirements:
1. entity retained when confidently detected
2. topic retained
3. generic style cannot replace entity
4. malformed query rejected
5. duplicated phrase rejected
6. missing entity handled safely
7. 3–5 distinct query intents generated
8. blocked query advances to next query
9. candidate target 30
10. overall cap 100
11. per-query cap 25
12. actual Reel profile drives queries
13. no hard-coded Spider-Man
14. source_queries preserved
"""
import pytest
from unittest.mock import MagicMock, patch
from services.discovery_profile import build_discovery_profile, is_generic_entity
from utils.query_builder import QueryBuilder
from agents.discovery_agent import DiscoveryAgent
from instagram.providers.apify import ApifyProvider


# 1. Entity retained when confidently detected
def test_entity_retained_when_confidently_detected():
    profile = {
        "primary_entity": "Spider-Man",
        "topic": "movie scene edit",
        "niche": "superhero",
        "content_format": "cinematic edit",
        "entity_confidence": 0.92,
    }
    disc_res = DiscoveryAgent.generate_queries(profile, max_queries=5)
    queries = disc_res["query_strings"]
    assert len(queries) >= 3
    for q in queries:
        assert "spider" in q.lower(), f"Entity missing in query: '{q}'"


# 2. Topic retained
def test_topic_retained():
    profile = {
        "primary_entity": "Spider-Man",
        "topic": "fight scene",
        "niche": "superhero",
        "content_format": "cinematic edit",
        "entity_confidence": 0.90,
    }
    queries = QueryBuilder.build_structured_queries(profile)
    high_precision_queries = [q["query"] for q in queries if q["search_level"] == "high_precision"]
    assert len(high_precision_queries) > 0
    # Check that topic word 'fight' or 'scene' is retained in high-precision query
    assert any("fight" in q.lower() or "scene" in q.lower() for q in high_precision_queries)


# 3. Generic style cannot replace entity
def test_generic_style_cannot_replace_entity():
    profile = {
        "primary_entity": "cinematic edit",
        "topic": "creative storytelling",
        "niche": "visual edit",
    }
    structured = build_discovery_profile(profile)
    assert structured["primary_entity"] == "", "Generic term was accepted as primary entity!"
    assert structured["entity_confidence"] == 0.0

    # Query generation should not produce repeated generic phrases
    queries = QueryBuilder.build_structured_queries(structured)
    for q in queries:
        assert "cinematic edit creative storytelling cinematic edit" not in q["query"]
        words = q["query"].split()
        assert len(words) == len(set(words)), f"Repeated words in query: '{q['query']}'"


# 4. Malformed query rejected
def test_malformed_query_rejected():
    # Dangling adjective
    assert not QueryBuilder.is_meaningful_query("spider man high")
    assert not QueryBuilder.validate_query("spider man high", "spider man", "", "")

    # Dangling interior adjective preceding edit
    assert not QueryBuilder.is_meaningful_query("superhero high edit")
    assert not QueryBuilder.validate_query("superhero high edit", "", "superhero", "")

    # Single word
    assert not QueryBuilder.is_meaningful_query("spiderman")


# 5. Duplicated phrase rejected
def test_duplicated_phrase_rejected():
    assert not QueryBuilder.is_meaningful_query("cinematic edit creative storytelling cinematic edit")
    assert not QueryBuilder.is_meaningful_query("spider man dramatic spider man edit")
    assert not QueryBuilder.is_meaningful_query("movie scene scene edit")


# 6. Missing entity handled safely (no invented entity)
def test_missing_entity_handled_safely():
    profile = {
        "primary_entity": "",
        "topic": "superhero movie scene",
        "niche": "entertainment",
        "content_format": "cinematic edit",
        "entity_confidence": 0.0,
    }
    disc_res = DiscoveryAgent.generate_queries(profile, max_queries=5)
    queries = disc_res["query_strings"]
    assert len(queries) >= 3
    for q in queries:
        assert "spider" not in q.lower(), "Fabricated Spider-Man into query without evidence!"
        assert "batman" not in q.lower(), "Fabricated Batman into query without evidence!"
        assert "superhero" in q.lower() or "movie" in q.lower() or "entertainment" in q.lower()


# 7. 3–5 distinct query intents generated
def test_distinct_query_intents_generated():
    profile = {
        "primary_entity": "Batman",
        "topic": "action scene",
        "niche": "superhero",
        "content_format": "fan edit",
        "entity_confidence": 0.90,
    }
    disc_res = DiscoveryAgent.generate_queries(profile, max_queries=5)
    queries = disc_res["queries"]
    assert 3 <= len(queries) <= 5
    levels = {q["search_level"] for q in queries}
    assert len(levels) >= 2, f"Queries lacked search level diversity: {levels}"
    query_texts = [q["query"] for q in queries]
    assert len(query_texts) == len(set(query_texts)), "Queries must all be unique"


# 8. Blocked query advances to next query
def test_blocked_query_advances_to_next_query():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    def mock_call(client, actor_id, run_input):
        q = run_input.get("search", "")
        if "query1" in q:
            call_obj = {
                "status": "FAILED",
                "statusMessage": "Search scraper failed to find any results because it was blocked by Instagram.",
            }
            return call_obj, None
        else:
            call_obj = {"status": "SUCCEEDED", "default_dataset_id": "ds2"}
            return call_obj, None

    provider._call_actor_safely = mock_call
    provider._fetch_dataset_safely = MagicMock(return_value=([
        {"id": "c1", "shortCode": "code1", "caption": "query2 post", "videoUrl": "http://v1"}
    ], None))

    queries = [
        {"query": "query1 blocked search", "search_level": "high_precision"},
        {"query": "query2 success search", "search_level": "medium"},
    ]

    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    q_results = diag["query_results"]
    assert len(q_results) == 2
    assert q_results[0]["status"] == "blocked"
    assert q_results[0]["retry_count"] == 0  # Non-hammering
    assert q_results[1]["status"] == "success"
    assert len(cands) == 1


# 9. Candidate target 30
def test_candidate_target_30_stops_iteration():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    # Return 15 unique candidates per query
    def mock_call(client, actor_id, run_input):
        return {"status": "SUCCEEDED", "default_dataset_id": "ds_any"}, None

    items_call_count = 0
    def mock_fetch(client, ds_id):
        nonlocal items_call_count
        items_call_count += 1
        return ([
            {"id": f"cand_{items_call_count}_{i}", "shortCode": f"sc_{items_call_count}_{i}", "videoUrl": "http://v"}
            for i in range(15)
        ], None)

    provider._call_actor_safely = mock_call
    provider._fetch_dataset_safely = mock_fetch

    queries = [
        {"query": "query1 term", "search_level": "high_precision"},
        {"query": "query2 term", "search_level": "high_precision"},
        {"query": "query3 term", "search_level": "medium"},
        {"query": "query4 term", "search_level": "broader"},
    ]

    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=50)

    # 15 from q1 + 15 from q2 = 30 reached target. Queries 3 and 4 should not be executed!
    assert len(diag["query_results"]) == 2
    assert len(cands) == 30


# 10. Overall cap 100
def test_overall_cap_100():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    # If limit is set to 150, provider should cap at settings.INSTAGRAM_MAX_RESULTS (100)
    fake_items = [{"id": f"item_{i}", "shortCode": f"sc_{i}", "videoUrl": "http://v"} for i in range(120)]
    provider._call_actor_safely = MagicMock(return_value=({"status": "SUCCEEDED", "default_dataset_id": "ds"}, None))
    provider._fetch_dataset_safely = MagicMock(return_value=(fake_items, None))

    queries = [{"query": "test query", "search_level": "high_precision"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=150)

    assert len(cands) <= 100


# 11. Per-query cap 25
def test_per_query_cap_25():
    from config import settings
    assert int(settings.APIFY_SEARCH_LIMIT) <= 25


# 12. Actual Reel profile drives queries
def test_actual_reel_profile_drives_queries():
    tech_profile = {
        "primary_entity": "ViralLens",
        "topic": "software testing sequence",
        "niche": "tech",
        "content_format": "test sequence",
        "entity_confidence": 0.90,
    }
    tech_res = DiscoveryAgent.generate_queries(tech_profile)
    for q in tech_res["query_strings"]:
        assert "virallens" in q.lower()

    fitness_profile = {
        "primary_entity": "Calisthenics",
        "topic": "pull up tutorial",
        "niche": "fitness",
        "content_format": "tutorial guide",
        "entity_confidence": 0.88,
    }
    fit_res = DiscoveryAgent.generate_queries(fitness_profile)
    for q in fit_res["query_strings"]:
        assert "calisthenics" in q.lower() or "pull" in q.lower() or "fitness" in q.lower()
        assert "virallens" not in q.lower()


# 13. No hard-coded Spider-Man
def test_no_hardcoded_spiderman():
    abstract_profile = {
        "primary_entity": "",
        "topic": "sunset ocean waves",
        "niche": "nature",
        "content_format": "ambient video",
        "entity_confidence": 0.0,
    }
    res = DiscoveryAgent.generate_queries(abstract_profile)
    for q in res["query_strings"]:
        assert "spider" not in q.lower()
        assert "spiderman" not in q.lower()


# 14. Source queries preserved
def test_source_queries_preserved():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    # Candidate c1 appears in both query1 and query2
    provider._call_actor_safely = MagicMock(return_value=({"status": "SUCCEEDED", "default_dataset_id": "ds"}, None))
    provider._fetch_dataset_safely = MagicMock(side_effect=[
        ([{"id": "c1", "shortCode": "sc1", "videoUrl": "http://v1"}], None),
        ([{"id": "c1", "shortCode": "sc1", "videoUrl": "http://v1"},
          {"id": "c2", "shortCode": "sc2", "videoUrl": "http://v2"}], None),
    ])

    queries = [
        {"query": "search term alpha", "search_level": "high_precision"},
        {"query": "search term beta", "search_level": "medium"},
    ]

    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    c1 = next(c for c in cands if c.get("shortcode") == "sc1" or c.get("id") == "sc1")
    assert "search term alpha" in c1["source_queries"]
    assert "search term beta" in c1["source_queries"]
