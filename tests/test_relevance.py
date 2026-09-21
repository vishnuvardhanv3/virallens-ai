"""Tests for metadata relevance and video semantic verification."""
from services.relevance import filter_relevant_candidates, score_video_relevance


def test_metadata_filtering_preserves_matches_and_rejects_unrelated():
    profile = {
        "niche": "superhero",
        "primary_entity": "spiderman",
        "topic": "spiderman action scenes",
        "keywords": ["marvel", "cinematic", "peter"],
    }
    candidates = [
        {"id": "c1", "caption": "Epic Spiderman swing scene #marvel #superhero", "source_query": "superhero spiderman"},
        {"id": "c2", "caption": "Peter Parker movie highlights", "source_query": "superhero spiderman edit"},
        {"id": "c3", "caption": "Delicious chocolate cake recipe #baking", "source_query": "superhero spiderman"},
    ]

    relevant, diag = filter_relevant_candidates(candidates, profile, minimum_score=0.10)
    assert len(relevant) >= 2
    rel_ids = [c["id"] for c in relevant]
    assert "c1" in rel_ids
    assert "c2" in rel_ids
    # Unrelated candidate rejected
    assert diag["rejected_candidates"] >= 1


def test_video_semantic_relevance():
    uploaded = {
        "niche": "fitness",
        "primary_entity": "bodybuilder",
        "topic": "hypertrophy chest workout",
        "format": "tutorial",
    }
    comp_relevant = {
        "niche": "fitness",
        "primary_entity": "bodybuilder",
        "topic": "chest and triceps hypertrophy workout",
        "format": "tutorial",
    }
    comp_unrelated = {
        "niche": "gaming",
        "primary_entity": "streamer",
        "topic": "minecraft speedrun",
        "format": "gameplay",
    }

    res_rel = score_video_relevance(uploaded, comp_relevant)
    assert res_rel["video_relevance_eligible"] is True
    assert res_rel["relevance_status"] == "verified_relevant"

    res_unrel = score_video_relevance(uploaded, comp_unrelated)
    assert res_unrel["video_relevance_eligible"] is False
    assert res_unrel["relevance_status"] == "analyzed_not_relevant"
