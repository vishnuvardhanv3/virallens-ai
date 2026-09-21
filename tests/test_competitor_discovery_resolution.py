"""
Regression tests for Competitor Discovery Resolution.

Validates:
1. Zero-result/empty Apify responses are classified as EMPTY, NOT BLOCKED.
2. Genuine Instagram platform challenges (checkpoint/login/429) are classified as BLOCKED.
3. QueryBuilder bounds query token length to 2-4 words for high-precision matching on Instagram topic index.
4. QueryBuilder retains entity and niche anchors across priority levels.
5. Raw candidate normalization preserves canonical shortcodes, video URLs, and metrics.
6. Discovery telemetry correctly captures actor_id, run_id, and dataset_id.
"""

import pytest
from unittest.mock import MagicMock
from instagram.providers.apify import ApifyProvider
from utils.query_builder import QueryBuilder
from services.discovery_provider import (
    QueryExecutionResult,
    QueryExecutionState,
    Candidate,
)


def test_empty_query_classification_not_blocked():
    """Verify that empty/no_items response from Instagram search scraper is classified as EMPTY, not BLOCKED."""
    provider = ApifyProvider(token="test_token")
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_client.actor.return_value = mock_actor

    # Simulate Apify actor output for an obscure query with zero public posts
    mock_run = {
        "id": "run_test_empty_123",
        "defaultDatasetId": "ds_test_empty_123",
        "actId": "apify/instagram-search-scraper",
        "status": "SUCCEEDED",
    }
    mock_actor.call.return_value = mock_run

    # Dataset contains the actor's no_items indicator
    mock_dataset = MagicMock()
    mock_client.dataset.return_value = mock_dataset
    mock_dataset.iterate_items.return_value = [
        {
            "error": "no_items",
            "errorDescription": "Empty or private data for provided input",
            "search": "mega man megamind action montage fan edits",
        }
    ]

    q_state, candidates = provider._execute_single_query(
        client=mock_client,
        actor_id="apify/instagram-search-scraper",
        template={"search": "{query}", "searchType": "popular", "searchLimit": 10},
        search_term="mega man megamind action montage fan edits",
        search_level="entity_edit",
        effective_search_limit=10,
        max_retries=1,
        q_idx=1,
    )

    assert q_state["status"] == "empty", f"Expected 'empty' but got '{q_state.get('status')}'"
    assert candidates == []
    assert q_state["actor_id"] == "apify/instagram-search-scraper"
    assert q_state["run_id"] == "run_test_empty_123"
    assert q_state["dataset_id"] == "ds_test_empty_123"


def test_genuine_platform_block_classification():
    """Verify that real Instagram challenges (checkpoint, 429, challenge_required) are classified as BLOCKED."""
    provider = ApifyProvider(token="test_token")
    mock_client = MagicMock()
    mock_actor = MagicMock()
    mock_client.actor.return_value = mock_actor

    mock_run = {
        "id": "run_test_blocked_456",
        "defaultDatasetId": "ds_test_blocked_456",
        "actId": "apify/instagram-search-scraper",
        "status": "SUCCEEDED",
    }
    mock_actor.call.return_value = mock_run

    mock_dataset = MagicMock()
    mock_client.dataset.return_value = mock_dataset
    # Dataset contains an actual checkpoint_required challenge
    mock_dataset.iterate_items.return_value = [
        {
            "crawler_status": "failed",
            "message": "Instagram challenge required: checkpoint_required at url https://www.instagram.com",
            "http_code": 429,
        }
    ]

    q_state, candidates = provider._execute_single_query(
        client=mock_client,
        actor_id="apify/instagram-search-scraper",
        template={"search": "{query}", "searchType": "popular", "searchLimit": 10},
        search_term="popular edit",
        search_level="niche",
        effective_search_limit=10,
        max_retries=1,
        q_idx=1,
    )

    assert q_state["status"] == "blocked", f"Expected 'blocked' but got '{q_state.get('status')}'"
    assert candidates == []


def test_query_builder_token_length_bounded():
    """Verify QueryBuilder bounds queries to 2-4 tokens so they resolve reliably on Instagram."""
    profile = {
        "entity": "mega man megamind",
        "niche": "Superhero",
        "sub_niche": "Mega Man Megamind Fan Edits",
        "edit_type": "Action Montage",
    }

    queries = QueryBuilder.build_structured_queries(profile)
    assert len(queries) >= 3, "Should generate at least 3 prioritized queries"

    for q in queries:
        query_text = q["query"]
        token_count = len(query_text.split())
        assert token_count >= 2, f"Query '{query_text}' should have at least 2 tokens"
        assert token_count <= 4, f"Query '{query_text}' has {token_count} tokens; must be <= 4 tokens"


def test_query_builder_anchors_entity_and_niche():
    """Verify QueryBuilder maintains entity and niche anchors across priority tiers."""
    profile = {
        "entity": "megamind",
        "niche": "Superhero",
        "sub_niche": "Supervillain Edits",
        "edit_type": "Velocity Edit",
    }

    queries = QueryBuilder.build_structured_queries(profile)
    query_texts = [q["query"].lower() for q in queries]

    # At least one query should be entity-centric
    assert any("megamind" in qt for qt in query_texts), "Expected entity 'megamind' in query list"
    # At least one query should incorporate edit type or niche
    assert any("edit" in qt or "superhero" in qt for qt in query_texts), "Expected edit or niche anchor"


def test_candidate_normalization_canonical_fields():
    """Verify raw Instagram candidate dictionaries normalize into valid Candidate model objects."""
    raw_item = {
        "id": "3512345678901234567",
        "shortCode": "C9XyZ123abc",
        "caption": "Mega mind transition 🔥 #megamind #edit",
        "videoUrl": "https://scontent-iad3-2.cdninstagram.com/v/t50.2886-16/test.mp4",
        "displayUrl": "https://scontent-iad3-2.cdninstagram.com/v/t51.2885-15/thumb.jpg",
        "ownerUsername": "editor_pro",
        "videoPlayCount": 150200,
        "likesCount": 12500,
        "commentsCount": 320,
        "videoDuration": 18.5,
    }

    provider = ApifyProvider(token="test_token")
    normalized_list = provider._normalize_candidates([raw_item], source_query="megamind edit", search_level="entity_edit")

    assert len(normalized_list) == 1
    normalized = normalized_list[0]
    assert normalized["reel_id"] == "C9XyZ123abc"
    assert normalized["shortcode"] == "C9XyZ123abc"
    assert normalized["media_url"] == "https://scontent-iad3-2.cdninstagram.com/v/t50.2886-16/test.mp4"
    assert normalized["creator"] == "editor_pro"
    assert normalized["reach_value"] == 150200
    assert normalized["likes"] == 12500
    assert normalized["comments"] == 320

    # Test Candidate abstraction
    cand = Candidate.from_dict(normalized, provider_name="apify")
    assert cand.reel_id == "C9XyZ123abc"
    assert cand.reach == 150200
    assert cand.media_url == "https://scontent-iad3-2.cdninstagram.com/v/t50.2886-16/test.mp4"


def test_query_execution_result_telemetry():
    """Verify QueryExecutionResult telemetry serializes actor_id, run_id, and dataset_id."""
    res = QueryExecutionResult(
        provider="apify",
        query="megamind edit",
        state=QueryExecutionState.SUCCESS,
        duration_ms=1234.5,
        raw_count=5,
        normalized_count=5,
        cache_hit=False,
        attempt=1,
        fallback=False,
        actor_id="apify/instagram-search-scraper",
        run_id="run_abc123",
        dataset_id="ds_xyz789",
    )

    d = res.to_dict()
    assert d["actor_id"] == "apify/instagram-search-scraper"
    assert d["run_id"] == "run_abc123"
    assert d["dataset_id"] == "ds_xyz789"

    tel = res.to_telemetry()
    assert tel["actor_id"] == "apify/instagram-search-scraper"
    assert tel["run_id"] == "run_abc123"
    assert tel["dataset_id"] == "ds_xyz789"
