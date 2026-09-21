"""Unit tests for Discovery Provider canonical contract, schemas, and Apify adapter."""
import pytest
from services.discovery_provider import (
    Candidate,
    ProviderCapabilities,
    ProviderHealthState,
    QueryExecutionResult,
    QueryExecutionState,
    build_canonical_reel_url,
    extract_canonical_shortcode,
    safe_int_reach,
)
from services.providers.apify_instagram_provider import ApifyInstagramProvider


def test_canonical_shortcode_extraction():
    """Verify robust extraction of Instagram shortcodes across /reel/, /p/, /tv/, and query params."""
    assert extract_canonical_shortcode("https://www.instagram.com/reel/C2_vRNYhykr/") == "C2_vRNYhykr"
    assert extract_canonical_shortcode("https://instagram.com/reel/C2_vRNYhykr/?utm_source=ig_web_copy_link") == "C2_vRNYhykr"
    assert extract_canonical_shortcode("https://www.instagram.com/p/B9-abc123XYZ/") == "B9-abc123XYZ"
    assert extract_canonical_shortcode("https://instagram.com/tv/D11-tvcode/") == "D11-tvcode"
    assert extract_canonical_shortcode("C2_vRNYhykr") == "C2_vRNYhykr"
    assert extract_canonical_shortcode("") == ""
    assert extract_canonical_shortcode(None) == ""


def test_build_canonical_reel_url():
    """Verify normalization to standard /reel/ URLs."""
    url = build_canonical_reel_url("https://www.instagram.com/p/C2_vRNYhykr/?utm_source=x")
    assert url == "https://www.instagram.com/reel/C2_vRNYhykr/"
    assert build_canonical_reel_url("C2_vRNYhykr") == "https://www.instagram.com/reel/C2_vRNYhykr/"


def test_safe_int_reach_integrity():
    """Strict reach integrity: positive ints or None. Never 0, never synthetic, never likes."""
    assert safe_int_reach(1000) == 1000
    assert safe_int_reach("5000") == 5000
    assert safe_int_reach(0) is None
    assert safe_int_reach(-10) is None
    assert safe_int_reach("0") is None
    assert safe_int_reach(None) is None
    assert safe_int_reach("") is None
    assert safe_int_reach("unavailable") is None


def test_candidate_dataclass_contract_and_dict():
    """Verify Candidate dataclass behavior, reach sanitization, and backward-compatible .to_dict()."""
    c = Candidate(
        reel_id="https://www.instagram.com/reel/C2_vRNYhykr/?utm_source=x",
        reel_url="https://www.instagram.com/reel/C2_vRNYhykr/?utm_source=x",
        creator="creator_alpha",
        caption="Epic Spider-Man montage",
        reach=1500000,
        timestamp="2026-03-01T12:00:00Z",
        media_url="https://cdn.instagram.com/video123.mp4",
        source_provider="apify",
        source_query="spider-man fan edits",
    )

    assert c.reel_id == "C2_vRNYhykr"
    assert c.reel_url == "https://www.instagram.com/reel/C2_vRNYhykr/"
    assert c.reach == 1500000
    assert c.metadata_completeness > 0.8

    # Backward compatible dict export
    d = c.to_dict()
    assert d["id"] == "C2_vRNYhykr"
    assert d["shortcode"] == "C2_vRNYhykr"
    assert d["views"] == 1500000
    assert d["reach_value"] == 1500000
    assert d["url"] == "https://www.instagram.com/reel/C2_vRNYhykr/"
    assert d["source_provider"] == "apify"
    assert "spider-man fan edits" in d["source_queries"]


def test_candidate_missing_values_remain_none():
    """Verify that missing fields strictly remain None and are never fabricated."""
    c = Candidate(
        reel_id="C9_emptyReel",
        reel_url="",
        reach=None,
        creator=None,
        media_url=None,
        caption=None,
    )
    assert c.reach is None
    assert c.creator is None
    assert c.caption is None
    assert c.media_url is None
    d = c.to_dict()
    assert d["views"] is None
    assert d["reach_value"] is None
    assert d["media_url"] is None


def test_query_execution_result_telemetry():
    """Verify QueryExecutionResult telemetry serialization matching Section 14."""
    res = QueryExecutionResult(
        provider="apify",
        query="spider-man fan edits",
        state=QueryExecutionState.BLOCKED,
        duration_ms=12900.0,
        raw_count=0,
        normalized_count=0,
        cache_hit=False,
        attempt=1,
        fallback=False,
        error_type="platform_block",
        message="Search scraper blocked by Instagram",
    )

    t = res.to_telemetry()
    assert t["provider"] == "apify"
    assert t["query"] == "spider-man fan edits"
    assert t["state"] == "blocked"
    assert t["duration_ms"] == 12900.0
    assert t["raw_count"] == 0
    assert t["normalized_count"] == 0
    assert t["cache_hit"] is False
    assert t["attempt"] == 1
    assert t["fallback"] is False
    assert t["error_type"] == "platform_block"


def test_apify_provider_adapter_capabilities_and_health(monkeypatch):
    """Verify Apify adapter capability reporting and health checks."""
    monkeypatch.setattr("config.settings.APIFY_API_TOKEN", "fake_token_abc")
    provider = ApifyInstagramProvider()
    
    # Health
    assert provider.health() == ProviderHealthState.HEALTHY

    # Capabilities
    caps = provider.capabilities()
    assert caps.can_search is True
    assert caps.can_fetch_metadata is True
    assert caps.can_extract_reach is True
    assert caps.can_extract_media_url is True
    assert caps.requires_auth is False

    # Missing token -> FAILED
    monkeypatch.setattr("config.settings.APIFY_API_TOKEN", "")
    provider_no_token = ApifyInstagramProvider(token="")
    assert provider_no_token.health() == ProviderHealthState.FAILED
