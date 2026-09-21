"""Unit tests for ViralityAgent."""
from agents.virality_agent import ViralityAgent


def test_virality_agent_evaluation():
    your_reel = {
        "analysis": {
            "niche": "superhero",
            "topic": "spiderman action edit",
            "primary_entity": "spiderman",
            "duration_seconds": 12.5,
            "hook": {
                "strength": 8.5,
                "first_3_seconds": "Fast visual action cut with Spider-Man landing",
            },
            "visual": {
                "visual_impact": 9.0,
                "scene_count": 14,
            },
        },
        "technical_signals": {
            "video": {
                "duration_sec": 12.5,
                "scene_change_count": 14,
            },
            "audio": {
                "has_audio": True,
                "rms_mean": 0.08,
            },
        },
    }

    verified_competitors = [
        {
            "views": 150000,
            "likes": 12000,
            "hook": {"strength": 0.80},
            "visual": {"visual_impact": 0.85},
            "video": {"duration_sec": 11.0},
        },
        {
            "views": 85000,
            "likes": 7500,
            "hook": {"strength": 0.75},
            "visual": {"visual_impact": 0.80},
            "video": {"duration_sec": 13.0},
        },
    ]

    res = ViralityAgent.evaluate(your_reel, verified_competitors)

    assert "current_virality_score" in res
    assert "projected_virality_score" in res
    assert 0.0 <= res["current_virality_score"] <= 1.0
    assert 0.0 <= res["projected_virality_score"] <= 1.0
    assert res["projected_virality_score"] >= res["current_virality_score"]
    assert res["verdict"] in ["HIGH_VIRAL", "MODERATE_VIRAL", "LOW_VIRAL"]
    assert len(res["viral_blueprint"]) >= 3
    assert "retention_leaks" in res
    assert "pillar_breakdown" in res
    assert "hook_velocity" in res["pillar_breakdown"]


def test_virality_agent_empty_competitors():
    your_reel = {
        "analysis": {
            "hook": {"strength": 0.5},
            "visual": {"visual_impact": 0.5},
        }
    }
    res = ViralityAgent.evaluate(your_reel, [])
    assert 0.0 <= res["current_virality_score"] <= 1.0
    assert len(res["viral_blueprint"]) >= 1
