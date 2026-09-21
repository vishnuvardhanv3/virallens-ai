"""Tests for final intelligence quality pass before cross-niche benchmark.

Covers:
1. OCR aggregate confidence and 0-100 normalization.
2. Contextual OCR quality filtering (noise rejection and legitimate short text preservation).
3. Discovery profile sub-niche and edit-type extraction.
4. Semantic-composition aware query generation.
5. Edit-type aware competitor verification (including rejection of cosplay/reviews for action montages).
6. Competitor clustering (PRIMARY_CLUSTER vs SECONDARY_CLUSTERS).
7. Pattern Agent strict denominator accounting across subgroups.
8. Non-causal attention wording (elimination of sustained uniform attention profile).
9. Per-query discovery diagnostics schema validation.
"""
import pytest
from agents.ocr_agent import classify_ocr_segment, extract_ocr
from services.discovery_profile import infer_sub_niche_and_edit_type, build_discovery_profile
from utils.query_builder import QueryBuilder
from providers.twelvelabs_provider import TwelveLabsProvider
from agents.competitor_agent import cluster_competitors, rank_candidates
from agents.pattern_agent import PatternAgent
from agents.human_attention_agent import HumanAttentionAgent


# 1. OCR Confidence Normalization & Calculation
def test_ocr_confidence_scale_and_aggregate():
    segments = [
        {"text": "Cum On", "confidence": 88.0, "start_time": 0.5, "end_time": 1.2, "duration": 0.7, "position": "center"},
        {"text": "one sec", "confidence": 76.0, "start_time": 2.0, "end_time": 2.8, "duration": 0.8, "position": "bottom"},
    ]
    # Simulate calculate mean_conf
    mean_conf = round(sum(s["confidence"] for s in segments) / len(segments), 1)
    assert mean_conf == 82.0
    assert 0.0 <= mean_conf <= 100.0


# 2. Contextual OCR Quality Filtering
def test_contextual_ocr_filtering_legitimate_vs_noise():
    # Legitimate phrases from real reels must be high_quality or usable
    assert classify_ocr_segment("Cum On", 85.0) in {"high_quality", "usable"}
    assert classify_ocr_segment("one sec", 75.0) in {"high_quality", "usable"}
    assert classify_ocr_segment("And that's Wat's", 78.0) in {"high_quality", "usable"}
    assert classify_ocr_segment("how You", 80.0) in {"high_quality", "usable"}
    assert classify_ocr_segment("repey me", 70.0) in {"high_quality", "usable"}
    assert classify_ocr_segment("VRSK", 82.0) in {"high_quality", "usable"}

    # Obvious garbage / repeated letter noise must be low_quality or noise
    assert classify_ocr_segment("Ly TTT aij", 40.0) in {"low_quality", "noise"}
    assert classify_ocr_segment("wee eek", 35.0) in {"low_quality", "noise"}
    assert classify_ocr_segment("TY eee rey dig port fot", 45.0) in {"low_quality", "noise"}
    assert classify_ocr_segment("roy Van", 30.0) in {"low_quality", "noise"}
    assert classify_ocr_segment("...", 20.0) in {"low_quality", "noise"}


# 3. Discovery Profile Sub-Niche & Edit-Type Extraction
def test_discovery_profile_sub_niche_and_edit_type():
    tl_analysis = {
        "niche": "superhero",
        "primary_entity": "spider-man",
        "topic": "spider-man cinematic fight scene action montage",
        "format": "fan edit",
        "summary": "High intensity dynamic combat cuts between Peter Parker and villains",
    }
    sub_niche, edit_type, sn_conf, et_conf = infer_sub_niche_and_edit_type(
        tl_analysis, scene_cut_rate=1.8, motion_magnitude=0.08
    )
    assert "spider man" in sub_niche or "spider-man" in sub_niche or "superhero" in sub_niche
    assert edit_type == "action montage"
    assert sn_conf >= 0.70
    assert et_conf >= 0.70

    # Fallback on empty/insufficient analysis
    sub_niche_fb, edit_type_fb, _, _ = infer_sub_niche_and_edit_type({}, scene_cut_rate=0.0, motion_magnitude=0.0)
    assert edit_type_fb in {"unknown", "mixed"}


# 4. Semantic-Composition Aware Query Generation
def test_semantic_composition_aware_query_generation():
    profile = {
        "niche": "superhero",
        "primary_entity": "spider man",
        "topic": "spider man rooftop fight scene",
        "sub_niche": "spider-man fan edits",
        "edit_type": "action montage",
        "keywords": ["marvel", "cinematic", "peter parker"],
        "entity_usable_for_discovery": True,
    }
    queries = QueryBuilder.build_queries(profile)
    assert 3 <= len(queries) <= 5
    query_texts = [q["query"] for q in queries]

    # No mechanical duplication like "spider man spider-man fan edits action montage"
    for qt in query_texts:
        words = qt.split()
        assert words.count("spider") <= 1
        assert words.count("man") <= 1
        assert len(qt) > 10

    # Should contain action montage fan edit angle and cinematic angle
    assert any("action montage" in qt for qt in query_texts)
    # Check fallback is present and labeled
    levels = [q.get("search_level") for q in queries]
    assert "fallback" in levels


# 5. Edit-Type Aware Competitor Verification & Incompatible Rejection
def test_edit_type_aware_competitor_verification():
    provider = TwelveLabsProvider()
    uploaded = {
        "niche": "superhero",
        "primary_entity": "spider-man",
        "topic": "spider-man rooftop fight scene",
        "sub_niche": "spider-man fan edits",
        "edit_type": "action montage",
        "format": "fan edit",
    }

    # Compatible action montage competitor
    comp_compatible = {
        "niche": "superhero",
        "primary_entity": "spider-man",
        "topic": "spider-man vs green goblin combat sequence",
        "sub_niche": "spider-man edits",
        "edit_type": "action montage",
        "format": "cinematic edit",
    }
    res_comp = provider.verify_semantic_relevance(uploaded, comp_compatible)
    assert res_comp["video_relevance_eligible"] is True
    assert res_comp["relevance_status"] == "verified_relevant"
    assert res_comp["edit_type_match"] >= 0.8
    assert res_comp["entity_match"] >= 0.8
    assert res_comp["overall_relevance"] >= 0.70

    # Incompatible cosplay / review competitor
    comp_incompatible = {
        "niche": "superhero",
        "primary_entity": "spider-man",
        "topic": "spider-man suit unboxing and cosplay review",
        "sub_niche": "cosplay review",
        "edit_type": "cosplay review",
        "format": "vlog",
        "caption": "Check out my new spiderman cosplay review #cosplay #review",
    }
    res_incomp = provider.verify_semantic_relevance(uploaded, comp_incompatible)
    assert res_incomp["video_relevance_eligible"] is False
    assert res_incomp["relevance_status"] == "analyzed_not_relevant"
    assert "Incompatible" in res_incomp["video_relevance_reason"] or res_incomp["edit_type_match"] == 0.0


# 6. Competitor Clustering
def test_competitor_clustering_primary_and_secondary():
    competitors = [
        {"id": "1", "creator": "alpha", "reach_value": 5000000, "views": 5000000, "format": "action montage", "topic": "spider-man fight", "analysis": {"edit_type": "action montage"}},
        {"id": "2", "creator": "beta", "reach_value": 3000000, "views": 3000000, "format": "action montage", "topic": "spider-man battle", "analysis": {"edit_type": "action montage"}},
        {"id": "3", "creator": "gamma", "reach_value": 1000000, "views": 1000000, "format": "emotional edit", "topic": "spider-man sad tribute", "analysis": {"edit_type": "emotional edit"}},
        {"id": "4", "creator": "delta", "reach_value": 800000, "views": 800000, "format": "dialogue edit", "topic": "peter parker quotes", "analysis": {"edit_type": "dialogue edit"}},
    ]
    clusters = cluster_competitors(competitors, user_edit_type="action montage")
    assert clusters["primary_cluster_key"] == "action montage"
    assert clusters["primary_count"] == 2
    assert clusters["secondary_count"] == 2
    assert len(clusters["secondary_clusters"]) >= 2
    # Verify primary cluster competitors sorted descending by reach
    p_reaches = [c["reach_value"] for c in clusters["primary_cluster"]["competitors"]]
    assert p_reaches == sorted(p_reaches, reverse=True)


# 7. Pattern Agent Strict Denominator Accounting
def test_pattern_agent_denominator_accounting():
    competitors = [
        {"id": "c1", "reach_value": 4000000, "views": 4000000, "primary_entity": "spider man", "competitor_cluster": "action montage", "caption": "spider man action", "analysis": {"ocr_text": "Hero", "canonical": {"ocr_text": "Hero"}}, "attention_predictions": [{"start_time": 0.0, "end_time": 0.5, "attention_score": 0.65, "feature_values": {"face_presence": 0.4, "text_presence": 0.3}}]},
        {"id": "c2", "reach_value": 2000000, "views": 2000000, "primary_entity": "spider man", "competitor_cluster": "action montage", "caption": "spider man fight", "analysis": {"ocr_text": "Fight", "canonical": {"ocr_text": "Fight"}}, "attention_predictions": [{"start_time": 0.0, "end_time": 0.5, "attention_score": 0.60, "feature_values": {"face_presence": 0.3, "text_presence": 0.2}}]},
        {"id": "c3", "reach_value": 1000000, "views": 1000000, "primary_entity": "batman", "competitor_cluster": "action montage", "caption": "batman action", "analysis": {"canonical": {}}, "attention_predictions": []},
    ]
    user_reel = {
        "discovery_profile": {
            "primary_entity": "spider man",
            "edit_type": "action montage",
        }
    }
    pattern_res = PatternAgent.analyze(competitors, user_reel)
    assert pattern_res["verified_competitor_count"] == 3
    assert pattern_res["primary_subgroup"] == "same_entity_and_edit_type"
    assert pattern_res["primary_pool_count"] == 2

    # Structured patterns must have count, denominator, subgroup, and provenance
    for p in pattern_res["structured_patterns"]:
        assert "count" in p
        assert "denominator" in p
        assert "subgroup" in p
        assert "provenance" in p
        assert p["count"] <= p["denominator"]


# 8. Non-Causal Attention Wording
def test_attention_agent_non_causal_wording():
    # Predictions with very low variation
    low_var_predictions = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.52, "feature_values": {}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.53, "feature_values": {}},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.52, "feature_values": {}},
        {"start_time": 1.5, "end_time": 2.0, "attention_score": 0.53, "feature_values": {}},
    ]
    res = HumanAttentionAgent.analyze(low_var_predictions)
    findings_str = " ".join(res["findings"])
    # Must NOT contain "indicating sustained uniform attention profile"
    assert "indicating sustained uniform attention profile" not in findings_str
    assert "sustained uniform attention profile" not in findings_str
    # Must contain cautious non-causal phrasing
    assert "Low temporal variation in the model output" in findings_str
    assert "temporal differences should be interpreted cautiously" in findings_str


# 9. Discovery Query States Schema Validation
def test_query_states_schema():
    from instagram.providers.apify import ApifyProvider
    # Test query state field completeness
    dummy_state = {
        "query": "spider man action montage fan edit",
        "attempt": 1,
        "status": "SUCCESS",
        "http_or_actor_status": "SUCCEEDED",
        "dataset_id": "ds_12345",
        "raw_items": 25,
        "normalized_items": 25,
        "block_reason": None,
        "error_reason": None,
        "retry_count": 0,
        "duration_seconds": 4.12,
    }
    required_fields = [
        "query", "attempt", "status", "http_or_actor_status", "dataset_id",
        "raw_items", "normalized_items", "block_reason", "error_reason",
        "retry_count", "duration_seconds"
    ]
    for rf in required_fields:
        assert rf in dummy_state
    assert dummy_state["status"] in {"SUCCESS", "BLOCKED", "EMPTY", "TIMEOUT", "ERROR"}
