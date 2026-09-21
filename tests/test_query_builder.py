"""Tests for authoritative QueryBuilder."""
from utils.query_builder import QueryBuilder


def test_niche_anchored_multi_word_queries():
    profile = {
        "niche": "superhero",
        "sub_niche": "cinematic edit",
        "primary_entity": "spiderman",
        "topic": "fight scene",
        "format": "cinematic edit",
        "keywords": ["marvel", "avengers", "action"],
    }
    queries = QueryBuilder.build_structured_queries(profile)

    # Must produce 3 to 5 queries
    assert 3 <= len(queries) <= 5

    # Check search levels
    levels = [q["search_level"] for q in queries]
    assert "high_precision" in levels
    assert "medium" in levels

    # CRITICAL: Every query MUST retain the niche or compound niche+entity anchor
    for q in queries:
        query_text = q["query"]
        words = query_text.split()
        # Must be multi-word (at least 2 words)
        assert len(words) >= 2, f"Query '{query_text}' must have at least 2 words"
        # Must contain niche or entity anchor
        assert "superhero" in query_text or "spiderman" in query_text, f"Query '{query_text}' lost anchor"


def test_query_builder_deduplication():
    profile = {
        "niche": "fitness",
        "primary_entity": "workout",
        "topic": "gym routine",
        "keywords": ["fitness", "gym", "workout", "workout"],
    }
    queries = QueryBuilder.build_structured_queries(profile)
    q_texts = [q["query"] for q in queries]
    assert len(q_texts) == len(set(q_texts)), "Queries must be unique"
    for q_text in q_texts:
        words = q_text.split()
        assert len(words) == len(set(words)), f"Repeated words in query: '{q_text}'"
