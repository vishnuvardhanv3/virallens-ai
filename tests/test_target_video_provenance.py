"""Regression tests ensuring target video identity preservation and anti-fabrication.

Verifies:
1. Input video identity 'htdyr.mp4' cannot silently become 'test_reel.mp4'.
2. Synthetic/generic test sequences cannot produce a hard-coded Spider-Man entity or query.
3. Multimodal analysis of htdyr.mp4 produces genuine Spider-Man queries from real video signals.
4. Target video path is strictly recorded in provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from verify_real_reel import resolve_target_video, PROJECT_ROOT
from services.discovery_profile import build_discovery_profile
from agents.discovery_agent import DiscoveryAgent


def test_input_video_identity_not_swapped():
    """Verify input video identity is strictly preserved and never silently swapped."""
    # Test htdyr.mp4 resolution
    resolved_htdyr = resolve_target_video("htdyr.mp4")
    assert resolved_htdyr.name == "htdyr.mp4"
    assert "test_reel.mp4" not in str(resolved_htdyr)
    assert resolved_htdyr.exists()

    # Test test_reel.mp4 resolution
    resolved_test = resolve_target_video("media/test_reel.mp4")
    assert resolved_test.name == "test_reel.mp4"
    assert "htdyr.mp4" not in str(resolved_test)
    assert resolved_test.exists()

    # Verify distinct file paths
    assert resolved_htdyr != resolved_test


def test_generic_test_sequence_cannot_produce_spiderman_query():
    """Verify that a synthetic/generic test sequence cannot produce hard-coded Spider-Man entity or queries."""
    # Test sequence analysis fixture (mimicking test_reel.mp4)
    test_sequence_analysis = {
        "niche": "tech",
        "sub_niche": "software testing",
        "topic": "ViralLens test frame sequence",
        "primary_entity": "ViralLens",
        "format": "test sequence",
        "visual_style": "minimalist, solid color backgrounds with white text",
        "editing_style": "simple cuts between static frames",
        "tone": "neutral",
        "language": "English",
        "hook": {
            "summary": "The video opens with a green screen displaying 'ViralLens Test Frame 0'.",
            "type": "visual hook",
            "first_3_seconds": "Green background with white text",
            "strength": 0.3,
        },
        "discovery_keywords": [
            "ViralLens test",
            "software testing",
            "test frame sequence",
            "tech demo",
        ],
    }

    profile = build_discovery_profile(test_sequence_analysis)

    # 1. Primary entity must NOT be Spider-Man
    assert "spider" not in str(profile.get("primary_entity", "")).lower()
    assert "superhero" not in str(profile.get("niche", "")).lower()

    # 2. ViralLens application artifact must be filtered out as unusable
    assert profile.get("entity_usable_for_discovery") is False

    # 3. Generated queries must NOT contain Spider-Man or superhero
    disc_res = DiscoveryAgent.generate_queries(profile, max_queries=5)
    queries = disc_res.get("queries", [])
    assert len(queries) >= 2

    for q in queries:
        query_str = q["query"].lower()
        assert "spider" not in query_str, f"Query '{query_str}' fabricated Spider-Man!"
        assert "superhero" not in query_str, f"Query '{query_str}' fabricated superhero!"
        # Must reflect tech/testing/format
        assert any(k in query_str for k in ["tech", "storytelling", "test", "sequence", "edit", "mixed"]), (
            f"Query '{query_str}' does not reflect target profile!"
        )


def test_htdyr_produces_spiderman_queries_from_multimodal_analysis():
    """Verify that htdyr.mp4 produces Spider-Man queries because Twelve Labs genuinely identified it."""
    cache_htdyr = PROJECT_ROOT / "cache" / "tl_d4fb2b3e9d548baf.json"
    if not cache_htdyr.exists():
        pytest.skip("Cached Twelve Labs analysis for htdyr.mp4 not found")

    data = json.loads(cache_htdyr.read_text(encoding="utf-8"))
    analysis = data.get("analysis", {})

    profile = build_discovery_profile(analysis)

    # htdyr.mp4 analysis has Spider-Man as the primary entity
    assert "spider" in str(profile.get("primary_entity", "")).lower()
    assert str(profile.get("niche", "")).lower() == "superhero"
    assert profile.get("entity_usable_for_discovery") is True

    disc_res = DiscoveryAgent.generate_queries(profile, max_queries=5)
    queries = disc_res.get("queries", [])
    assert len(queries) >= 3

    # Every primary query should contain spider man because it was genuinely extracted
    assert any("spider man" in q["query"].lower() for q in queries)


def test_criteria_keys_completeness():
    """Verify all 11 required E2E audit criteria (A through K) are accounted for."""
    expected_criteria = [
        "A_target_video_loaded",
        "B_twelvelabs_analysis_completed",
        "C_discovery_profile_derived",
        "D_queries_derived_from_profile",
        "E_apify_executed_queries",
        "F_competitors_from_queries",
        "G_competitor_mp4s_downloaded",
        "H_competitor_mp4s_analyzed",
        "I_semantic_relevance_verified",
        "J_pattern_analysis_used_verified",
        "K_strategy_used_evidence",
    ]
    # Check that verify_real_reel script checks these exact criteria
    import inspect
    from verify_real_reel import run_verification
    src = inspect.getsource(run_verification)
    for crit in expected_criteria:
        assert crit in src, f"Criterion {crit} is missing from verify_real_reel.py!"


def test_resolve_target_video_missing_file_raises_error():
    """Verify that specifying a non-existent video raises FileNotFoundError without silent fallback."""
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_target_video("non_existent_arbitrary_video_12345.mp4")
    assert "not found" in str(exc_info.value).lower()


def test_resolve_target_video_none_defaults_to_htdyr():
    """Verify that omitting video argument defaults to canonical htdyr.mp4 when present."""
    default_resolved = resolve_target_video(None)
    assert default_resolved.name == "htdyr.mp4"
    assert "test_reel.mp4" not in str(default_resolved)


def test_real_test_reel_multimodal_cache_produces_zero_spiderman():
    """Verify that the actual Twelve Labs multimodal analysis of test_reel.mp4 yields zero Spider-Man signals."""
    cache_test_reel = PROJECT_ROOT / "cache" / "tl_c2961d8b715578cd.json"
    if not cache_test_reel.exists():
        pytest.skip("Cached Twelve Labs analysis for test_reel.mp4 not found")

    data = json.loads(cache_test_reel.read_text(encoding="utf-8"))
    analysis = data.get("analysis", {})

    profile = build_discovery_profile(analysis)

    # Must be recognized as tech test sequence
    assert "spider" not in str(profile.get("primary_entity", "")).lower()
    assert "superhero" not in str(profile.get("niche", "")).lower()
    assert profile.get("entity_usable_for_discovery") is False

    disc_res = DiscoveryAgent.generate_queries(profile, max_queries=5)
    queries = disc_res.get("queries", [])
    assert len(queries) >= 2

    for q in queries:
        q_lower = q["query"].lower()
        assert "spider" not in q_lower
        assert "superhero" not in q_lower
