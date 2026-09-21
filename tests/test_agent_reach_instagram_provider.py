"""Unit tests for the experimental Agent-Reach Instagram Provider Adapter."""
import pytest
from services.discovery_provider import (
    ProviderCapabilities,
    ProviderHealthState,
    QueryExecutionResult,
    QueryExecutionState,
)
from services.providers.agent_reach_instagram_provider import AgentReachInstagramProvider


def test_agent_reach_truthful_availability_detection():
    """Verify that Agent-Reach accurately detects absence of CLI/session without crashing."""
    provider = AgentReachInstagramProvider(cli_path="C:\\nonexistent\\opencli.exe", session_id="")
    avail, reason = provider.is_available()
    assert avail is False
    assert "not found" in reason.lower() or "session" in reason.lower()


def test_agent_reach_honest_health_and_capabilities():
    """Verify that Agent-Reach reports FAILED health and False capabilities when unavailable."""
    provider = AgentReachInstagramProvider(cli_path="C:\\nonexistent\\opencli.exe")
    assert provider.health() == ProviderHealthState.FAILED
    
    caps = provider.capabilities()
    assert caps.can_search is False
    assert caps.can_fetch_metadata is False
    assert caps.can_extract_reach is False
    assert caps.can_extract_media_url is False
    assert caps.requires_auth is True


def test_agent_reach_structured_execution_failure():
    """Verify that executing a query against an unavailable Agent-Reach returns structured failure without crashing."""
    provider = AgentReachInstagramProvider(cli_path="C:\\nonexistent\\opencli.exe")
    res = provider.execute_query("spider-man fan edits", limit=25)

    assert isinstance(res, QueryExecutionResult)
    assert res.provider == "agent_reach"
    assert res.state == QueryExecutionState.ERROR
    assert res.error_type == "provider_unavailable"
    assert res.raw_count == 0
    assert res.normalized_count == 0
    assert len(res.candidates) == 0
    assert res.cache_hit is False
    assert res.duration_ms >= 0.0


def test_agent_reach_multi_query_discovery_degradation():
    """Verify multi-query discovery handles unavailable Agent-Reach cleanly."""
    provider = AgentReachInstagramProvider(cli_path="C:\\nonexistent\\opencli.exe")
    cands, diag = provider.discover_reels([{"query": "spider man", "search_level": "high_precision"}], limit=30)

    assert cands == []
    assert diag["provider"] == "agent_reach"
    assert diag["status"] == "PROVIDER_UNAVAILABLE"
    assert diag["discovery_health"]["discovery_health"] == "failed"
    assert diag["valid_reels"] == 0


def test_agent_reach_strict_security_and_compliance():
    """Verify no forbidden scraping tools (Playwright, Selenium, Instaloader, Instagrapi) are imported."""
    import ast
    from pathlib import Path

    file_path = Path("services/providers/agent_reach_instagram_provider.py")
    tree = ast.parse(file_path.read_text(encoding="utf-8"))

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.lower())

    forbidden_tokens = ["playwright", "selenium", "instaloader", "instagrapi"]
    for token in forbidden_tokens:
        assert not any(token in mod for mod in imported_modules), f"Forbidden import '{token}' detected in Agent-Reach provider adapter"
