"""Tests for ReportService JSON and Markdown generation."""
from services.report_service import save_report


def test_save_report_generates_json_and_md(tmp_path):
    payload = {
        "your_reel": {
            "niche": "superhero",
            "topic": "spiderman cinematic edit",
            "primary_entity": "spiderman",
            "hook": {"strength": 0.85, "first_3_seconds": "Opening jump"},
            "visual": {"visual_impact": 0.8},
        },
        "queries": [{"query": "superhero spiderman cinematic edit", "search_level": "high_precision"}],
        "discovery_diagnostics": {"raw_candidates": 20, "selected_competitors": 5},
        "verified_competitors": [
            {
                "id": "comp1",
                "creator": "spidey_fan",
                "video_relevance_score": 0.92,
                "hook": {"strength": 0.8},
                "views": 250000,
            }
        ],
        "rejected_competitors": [],
        "patterns": {"status": "complete", "cluster_count": 1, "clusters": []},
        "recommendations": ["Use high-action visual hook in opening second."],
    }

    json_p, md_p = save_report(payload, base_name="test_report")
    assert json_p.exists()
    assert md_p.exists()
    assert json_p.stat().st_size > 0
    assert md_p.stat().st_size > 0

    content = md_p.read_text(encoding="utf-8")
    assert "ViralLens AI" in content
    assert "superhero" in content
    assert "85%" in content
