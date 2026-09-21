"""Tests for single ML pattern discovery layer."""
from ml.pattern_discovery import PatternDiscovery


def test_ml_guard_requires_at_least_4_competitors():
    ml = PatternDiscovery()

    # Case 1: 3 competitors (< 4) must return insufficient_competitor_analyses
    competitors_3 = [
        {"hook": {"strength": 0.8}, "visual": {"visual_impact": 0.7}, "views": 10000},
        {"hook": {"strength": 0.75}, "visual": {"visual_impact": 0.8}, "views": 25000},
        {"hook": {"strength": 0.9}, "visual": {"visual_impact": 0.85}, "views": 50000},
    ]
    res_3 = ml.discover(competitors_3)
    assert res_3["status"] == "insufficient_competitor_analyses"
    assert res_3["cluster_count"] == 0
    assert len(res_3["clusters"]) == 0

    # Case 2: 4 competitors (>= 4) executes KMeans clustering without error
    competitors_4 = competitors_3 + [
        {"hook": {"strength": 0.65}, "visual": {"visual_impact": 0.6}, "views": 5000}
    ]
    res_4 = ml.discover(competitors_4, k=2)
    assert res_4["status"] == "complete"
    assert res_4["cluster_count"] > 0
    assert len(res_4["clusters"]) > 0
