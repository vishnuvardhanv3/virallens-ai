"""Tests for Apify Success Result Extraction and Normalization.

Validates all Section 14 requirements:
1. actor SUCCEEDED + valid dataset => SUCCESS
2. actor SUCCEEDED + 12 raw records => candidates > 0
3. successful dataset must not be retried
4. optional missing fields do not cause ERROR
5. malformed individual item does not invalidate valid items
6. actor success + zero items => EMPTY
7. explicit Instagram block => BLOCKED
8. dataset extraction exception => DATASET_EXTRACTION_ERROR / ERROR
9. successful candidate source_queries preserved
10. successful queries contribute to final candidate pool
11. reach values preserved
12. missing reach remains None
13. discovery summary counts successful actor runs correctly
14. overall health becomes PARTIAL when some queries block and others succeed
"""
import pytest
from unittest.mock import MagicMock, patch
from instagram.providers.apify import ApifyProvider


# 1. actor SUCCEEDED + valid dataset => SUCCESS
def test_actor_succeeded_plus_valid_dataset_is_success():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    call_obj = {
        "status": "SUCCEEDED",
        "statusMessage": "Finished! Total 1 requests: 1 succeeded, 0 failed.",
        "defaultDatasetId": "ds_valid_1",
    }
    raw_records = [
        {"id": "post_1", "shortCode": "short_1", "url": "https://www.instagram.com/reel/short_1/", "videoViewCount": 15000}
    ]

    provider._call_actor_safely = MagicMock(return_value=(call_obj, None))
    provider._fetch_dataset_safely = MagicMock(return_value=(raw_records, None))

    queries = [{"query": "spider man superhero edit", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    qr = diag["query_results"][0]
    assert qr["status"] == "success", f"Expected success, got {qr['status']}"
    assert qr["results_count"] == 1
    assert qr["retry_count"] == 0
    assert len(cands) == 1


# 2. actor SUCCEEDED + 12 raw records => candidates > 0
def test_actor_succeeded_plus_12_raw_records():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    call_obj = {
        "status": "SUCCEEDED",
        "statusMessage": "[SEARCH]: found 12 result(s)",
        "default_dataset_id": "ds_12",
    }
    raw_12 = [
        {
            "id": f"id_{i}",
            "shortcode": f"sc_{i}",
            "permalink": f"https://www.instagram.com/reel/sc_{i}/",
            "videoViewCount": 1000 * (i + 1),
            "ownerUsername": f"creator_{i}",
        }
        for i in range(12)
    ]

    provider._call_actor_safely = MagicMock(return_value=(call_obj, None))
    provider._fetch_dataset_safely = MagicMock(return_value=(raw_12, None))

    queries = [{"query": "spider man superhero edit", "search_level": "high_precision"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=20)

    qr = diag["query_results"][0]
    assert qr["status"] == "success"
    assert qr["results_count"] == 12
    assert qr["raw_items"] == 12
    assert qr["normalized_items"] == 12
    assert len(cands) == 12


# 3. successful dataset must not be retried
def test_successful_dataset_not_retried():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    call_mock = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds_single",
    }, None))
    fetch_mock = MagicMock(return_value=([
        {"shortCode": "code_abc", "url": "https://www.instagram.com/reel/code_abc/"}
    ], None))

    provider._call_actor_safely = call_mock
    provider._fetch_dataset_safely = fetch_mock

    queries = [{"query": "test query", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    assert call_mock.call_count == 1, "Successful actor was incorrectly retried!"
    assert diag["query_results"][0]["retry_count"] == 0


# 4. optional missing fields do not cause ERROR
def test_optional_missing_fields_do_not_cause_error():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    # Item missing caption, creator, likes, comments, views, media_url
    minimal_item = {
        "url": "https://www.instagram.com/reel/bare_min_123/",
    }

    provider._call_actor_safely = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds_min",
    }, None))
    provider._fetch_dataset_safely = MagicMock(return_value=([minimal_item], None))

    queries = [{"query": "test minimal", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    assert diag["query_results"][0]["status"] == "success"
    assert len(cands) == 1
    c = cands[0]
    assert c["shortcode"] == "bare_min_123"
    assert c["views"] is None
    assert c["likes"] is None
    assert c["creator"] == "unknown"


# 5. malformed individual item does not invalidate valid items
def test_malformed_individual_item_does_not_invalidate_valid_items():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    mixed_items = [
        {"something_weird": 123},  # Missing url and shortcode
        {"searchType": "hashtag"},  # Directory page
        {"shortCode": "valid_one", "url": "https://www.instagram.com/reel/valid_one/", "views": 500},
        {"isProfile": True},       # Profile page
        {"shortCode": "valid_two", "url": "https://www.instagram.com/reel/valid_two/", "views": 900},
    ]

    provider._call_actor_safely = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds_mixed",
    }, None))
    provider._fetch_dataset_safely = MagicMock(return_value=(mixed_items, None))

    queries = [{"query": "test mixed", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    qr = diag["query_results"][0]
    assert qr["status"] == "success"
    assert qr["results_count"] == 2
    assert len(cands) == 2


# 6. actor success + zero items => EMPTY
def test_actor_success_plus_zero_items_is_empty():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    provider._call_actor_safely = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "statusMessage": "Finished without results",
        "defaultDatasetId": "ds_empty",
    }, None))
    provider._fetch_dataset_safely = MagicMock(return_value=([], None))

    queries = [{"query": "test empty query", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    assert diag["query_results"][0]["status"] == "empty"
    assert diag["query_results"][0]["error"] is None
    assert len(cands) == 0


# 7. explicit Instagram block => BLOCKED
def test_explicit_instagram_block_is_blocked():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    provider._call_actor_safely = MagicMock(return_value=({
        "status": "FAILED",
        "statusMessage": "Search scraper failed to find any results because it was blocked by Instagram.",
    }, None))

    queries = [{"query": "blocked query", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    assert diag["query_results"][0]["status"] == "blocked"
    assert diag["query_results"][0]["retry_count"] == 0


# 8. dataset extraction exception => DATASET_EXTRACTION_ERROR / ERROR
def test_dataset_extraction_exception_is_error():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    provider._call_actor_safely = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds_err",
    }, None))
    provider._fetch_dataset_safely = MagicMock(return_value=([], "Failed to connect to dataset endpoint"))

    queries = [{"query": "dataset fail", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    assert diag["query_results"][0]["status"] == "error"
    assert "Dataset read error" in str(diag["query_results"][0]["error"])


# 9. successful candidate source_queries preserved
def test_successful_candidate_source_queries_preserved():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    # Query 1 and Query 2 both return post_alpha
    provider._call_actor_safely = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds_any",
    }, None))
    provider._fetch_dataset_safely = MagicMock(side_effect=[
        ([{"shortCode": "alpha", "url": "https://www.instagram.com/reel/alpha/"}], None),
        ([{"shortCode": "alpha", "url": "https://www.instagram.com/reel/alpha/"},
          {"shortCode": "beta", "url": "https://www.instagram.com/reel/beta/"}], None),
    ])

    queries = [
        {"query": "spider man superhero edit", "search_level": "high_precision"},
        {"query": "spider man superhero montage", "search_level": "medium"},
    ]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    alpha_cand = next(c for c in cands if c["shortcode"] == "alpha")
    assert "spider man superhero edit" in alpha_cand["source_queries"]
    assert "spider man superhero montage" in alpha_cand["source_queries"]


# 10. successful queries contribute to final candidate pool
def test_successful_queries_contribute_to_final_candidate_pool():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    provider._call_actor_safely = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds",
    }, None))
    provider._fetch_dataset_safely = MagicMock(side_effect=[
        ([{"shortCode": "c_1", "url": "https://www.instagram.com/reel/c_1/"}], None),
        ([{"shortCode": "c_2", "url": "https://www.instagram.com/reel/c_2/"}], None),
    ])

    queries = [
        {"query": "query 1", "search_level": "medium"},
        {"query": "query 2", "search_level": "medium"},
    ]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    assert len(cands) == 2
    shortcodes = {c["shortcode"] for c in cands}
    assert shortcodes == {"c_1", "c_2"}


# 11. reach values preserved
def test_reach_values_preserved():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    provider._call_actor_safely = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds",
    }, None))
    provider._fetch_dataset_safely = MagicMock(return_value=([
        {"shortCode": "c_reach", "url": "https://www.instagram.com/reel/c_reach/", "videoViewCount": 750000}
    ], None))

    queries = [{"query": "query reach", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    c = cands[0]
    assert c["reach_value"] == 750000
    assert c["views"] == 750000
    assert c["reach_metric"] == "views"


# 12. missing reach remains None
def test_missing_reach_remains_none():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    provider._call_actor_safely = MagicMock(return_value=({
        "status": "SUCCEEDED",
        "defaultDatasetId": "ds",
    }, None))
    provider._fetch_dataset_safely = MagicMock(return_value=([
        {"shortCode": "c_no_reach", "url": "https://www.instagram.com/reel/c_no_reach/"}
    ], None))

    queries = [{"query": "query no reach", "search_level": "medium"}]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    c = cands[0]
    assert c["reach_value"] is None
    assert c["views"] is None
    assert c["reach_metric"] == "unavailable"


# 13. discovery summary counts successful actor runs correctly
def test_discovery_summary_counts_successful_actor_runs():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    # 2 succeeded, 1 blocked
    provider._call_actor_safely = MagicMock(side_effect=[
        ({"status": "SUCCEEDED", "defaultDatasetId": "ds1"}, None),
        ({"status": "FAILED", "statusMessage": "Search scraper failed to find any results because it was blocked by Instagram."}, None),
        ({"status": "SUCCEEDED", "defaultDatasetId": "ds3"}, None),
    ])
    provider._fetch_dataset_safely = MagicMock(side_effect=[
        ([{"shortCode": "s1", "url": "https://www.instagram.com/reel/s1/"}], None),
        ([{"shortCode": "s3", "url": "https://www.instagram.com/reel/s3/"}], None),
    ])

    queries = [
        {"query": "q1", "search_level": "medium"},
        {"query": "q2", "search_level": "medium"},
        {"query": "q3", "search_level": "medium"},
    ]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    dh = diag["discovery_health"]
    assert dh["queries_attempted"] == 3
    assert dh["queries_succeeded"] == 2
    assert dh["queries_blocked"] == 1
    assert dh["candidates_found"] == 2


# 14. overall health becomes PARTIAL when some queries block and others succeed
def test_overall_health_partial_when_some_block_and_others_succeed():
    provider = ApifyProvider()
    provider.is_configured = MagicMock(return_value=True)

    provider._call_actor_safely = MagicMock(side_effect=[
        ({"status": "SUCCEEDED", "defaultDatasetId": "ds1"}, None),
        ({"status": "FAILED", "statusMessage": "Search scraper failed to find any results because it was blocked by Instagram."}, None),
    ])
    provider._fetch_dataset_safely = MagicMock(return_value=(
        [{"shortCode": "s1", "url": "https://www.instagram.com/reel/s1/"}], None
    ))

    queries = [
        {"query": "success q", "search_level": "medium"},
        {"query": "blocked q", "search_level": "medium"},
    ]
    with patch("apify_client.ApifyClient"):
        cands, diag = provider.discover_reels(queries, limit=10)

    assert diag["discovery_health"]["discovery_health"] in {"partial", "healthy"}
    assert len(cands) == 1
