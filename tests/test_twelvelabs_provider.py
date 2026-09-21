"""Tests for TwelveLabsProvider parsing and semantic relevance verification."""
from providers.twelvelabs_provider import TwelveLabsProvider


def test_parse_json_defensive():
    provider = TwelveLabsProvider()

    # Raw JSON
    assert provider._parse_json_defensive('{"niche": "fitness", "hook": {"strength": 8}}') == {
        "niche": "fitness",
        "hook": {"strength": 8},
    }

    # Wrapped in markdown code fence
    fence = """Here is the analysis:
```json
{
  "niche": "comedy",
  "topic": "relatable skit",
  "hook": {"strength": 0.85}
}
```
Hope this helps!"""
    parsed = provider._parse_json_defensive(fence)
    assert parsed["niche"] == "comedy"
    assert parsed["hook"]["strength"] == 0.85

    # Invalid text returns None defensively
    assert provider._parse_json_defensive("Not a json at all") is None


def test_semantic_relevance_verification():
    provider = TwelveLabsProvider()

    uploaded = {
        "niche": "superhero",
        "primary_entity": "spiderman",
        "topic": "peter parker action edit",
        "format": "cinematic edit",
    }

    # Case A: Related competitor
    related_competitor = {
        "niche": "superhero",
        "primary_entity": "spiderman",
        "topic": "spiderman vs villains fight",
        "format": "cinematic edit",
    }
    res_a = provider.verify_semantic_relevance(uploaded, related_competitor)
    assert res_a["video_relevance_eligible"] is True
    assert res_a["relevance_status"] == "verified_relevant"
    assert res_a["video_relevance_score"] >= 0.7

    # Case B: Completely unrelated competitor
    unrelated_competitor = {
        "niche": "beauty",
        "primary_entity": "skincare brand",
        "topic": "morning makeup routine",
        "format": "tutorial",
    }
    res_b = provider.verify_semantic_relevance(uploaded, unrelated_competitor)
    assert res_b["video_relevance_eligible"] is False
    assert res_b["relevance_status"] == "analyzed_not_relevant"
    assert res_b["video_relevance_score"] < 0.45
