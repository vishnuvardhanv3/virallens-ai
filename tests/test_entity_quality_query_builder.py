"""Unit & integration tests for entity quality, artifact filtering, and QueryBuilder templates."""
import pytest
from services.discovery_profile import (
    build_discovery_profile,
    is_artifact_entity,
    is_generic_entity,
    classify_entity_type,
)
from utils.query_builder import QueryBuilder, is_meaningful_query
from agents.discovery_agent import DiscoveryAgent


def test_real_content_entity_classification_and_usability():
    """Confidently identified entities should be usable for entity-first discovery."""
    analysis = {
        "primary_entity": "Spider-Man",
        "topic": "rooftop leap",
        "niche": "superhero",
        "format": "cinematic edit",
        "keywords": ["spiderman", "peter parker", "action"],
    }
    profile = build_discovery_profile(analysis)
    assert profile["entity_usable_for_discovery"] is True
    assert profile["primary_entity"].lower() in {"spider-man", "spider man"}
    assert profile["entity_type"] in {"media_entity", "content_entity"}
    assert profile["entity_confidence"] >= 0.50

    queries = QueryBuilder.build_structured_queries(profile)
    assert 3 <= len(queries) <= 5
    # High precision query must contain the entity
    hp_query = next((q["query"] for q in queries if q["search_level"].lower() == "high_precision"), None)
    assert hp_query is not None
    assert any(term in hp_query.lower() for term in ["spider man", "spiderman", "spider-man"])


def test_artifact_rejection_app_names_and_watermarks():
    """Application names and watermarks must be rejected as artifacts and never used for discovery."""
    test_artifacts = [
        "ViralLens",
        "virallens ai",
        "CapCut",
        "created with capcut",
        "Premiere Pro",
        "Adobe After Effects",
        "@creator_handle",
        "test sequence",
        "demo footage",
        "sample sequence",
    ]
    for art in test_artifacts:
        assert is_artifact_entity(art) is True, f"Expected '{art}' to be recognized as artifact"
        ent_type = classify_entity_type(art)
        assert ent_type == "artifact", f"Expected '{art}' to be classified as artifact, got {ent_type}"

    # Verify build_discovery_profile zeroes out artifact entities
    analysis = {
        "primary_entity": "ViralLens AI",
        "topic": "app tutorial",
        "niche": "tech",
        "format": "screen record",
    }
    profile = build_discovery_profile(analysis)
    assert profile["entity_usable_for_discovery"] is False
    assert profile["primary_entity"] == ""
    assert profile["entity_type"] == "artifact"


def test_generic_phrase_rejection():
    """Generic phrases must be rejected from being treated as primary entities."""
    generic_terms = [
        "cinematic edit",
        "creative storytelling",
        "cinematic edit creative storytelling",
        "visual effects",
        "video edit",
        "reel",
        "viral video",
    ]
    for term in generic_terms:
        assert is_generic_entity(term) is True, f"Expected '{term}' to be recognized as generic"

    # Query builder rejection of generic queries
    assert is_meaningful_query("cinematic edit creative storytelling cinematic edit") is False
    assert is_meaningful_query("cinematic edit") is False
    assert is_meaningful_query("video") is False
    assert is_meaningful_query("spider man high") is False


def test_fallback_queries_when_no_usable_entity():
    """When no entity is usable, query builder must generate clean Topic + Niche fallbacks."""
    profile = {
        "primary_entity": "",
        "entity_usable_for_discovery": False,
        "topic": "portrait photography tips",
        "niche": "photography",
        "format": "tutorial",
        "style": "lighting guide",
        "keywords": ["camera", "lighting", "portrait"],
    }
    queries = QueryBuilder.build_structured_queries(profile)
    assert 3 <= len(queries) <= 5

    for q in queries:
        q_text = q["query"]
        words = q_text.split()
        # Must be multi-word
        assert len(words) >= 2, f"Query '{q_text}' must have at least 2 words"
        # No single-word or empty queries
        assert len(q_text.strip()) > 3
        # Must not contain generic repeats
        assert len(words) == len(set(words)), f"Repeated words in query '{q_text}'"
        # Must contain either niche or topic keywords
        assert any(term in q_text.lower() for term in ["photo", "portrait", "tips", "camera", "tutorial"])


def test_discovery_agent_end_to_end_generation():
    """DiscoveryAgent.generate_queries() returns validated queries with diagnostics."""
    profile = {
        "primary_entity": "Lionel Messi",
        "entity_usable_for_discovery": True,
        "entity_confidence": 0.95,
        "entity_type": "person_entity",
        "topic": "free kick goal",
        "niche": "football",
        "format": "sports highlights",
        "keywords": ["soccer", "inter miami", "argentina"],
    }
    res = DiscoveryAgent.generate_queries(profile, max_queries=4)
    assert "queries" in res
    assert 3 <= len(res["queries"]) <= 4
    for q in res["queries"]:
        assert "query" in q
        assert "search_level" in q
        assert len(q["query"].split()) >= 2
