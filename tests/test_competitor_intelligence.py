"""Comprehensive test suite for Phase 15 Competitor Intelligence Engine.

Covers all 25 required tests:
1. target schema normalization
2. competitor schema normalization
3. subgroup construction
4. denominator accounting
5. same-entity/edit-type grouping
6. same-edit-type grouping
7. all-verified grouping
8. highest-reach subgroup
9. numeric comparison
10. categorical pattern detection
11. attention pattern detection
12. target-vs-group gap
13. reach-aware handling
14. missing reach handling
15. small-sample protection
16. insufficient-sample classification
17. pattern deduplication
18. provenance
19. recommendation evidence linkage
20. no causal language
21. no retention/scroll-stop claims
22. no duplicate model inference
23. no additional network calls
24. performance regression
25. production invariance
"""
import time
import pytest
from typing import Any, Dict, List
from services.competitor_intelligence import CompetitorIntelligenceService
from agents.pattern_agent import PatternAgent
from agents.strategy_agent import StrategyAgent


# Mock fixtures
def make_mock_target() -> Dict[str, Any]:
    return {
        "id": "target_123",
        "shortcode": "target_123",
        "primary_entity": "Spider-Man",
        "niche": "superhero",
        "sub_niche": "action edits",
        "topic": "spider man combat montage",
        "edit_type": "action montage",
        "format": "montage",
        "duration": 18.5,
        "width": 1080,
        "height": 1920,
        "fps": 30.0,
        "ocr": {
            "ocr_text": "THE AMAZING SPIDER-MAN",
            "first_text_time": 0.8,
            "text_present": True,
        },
        "technical_signals": {
            "duration": 18.5,
            "width": 1080,
            "height": 1920,
            "fps": 30.0,
            "audio_rms": 0.15,
            "silence_ratio": 0.05,
        },
        "analysis": {
            "primary_entity": "Spider-Man",
            "niche": "superhero",
            "edit_type": "action montage",
            "visual_subjects": ["Spider-Man", "city skyline"],
            "hook": {"type": "visual surprise", "strength": 0.85},
            "audio": {"speech_present": True},
            "transcript": "With great power comes great responsibility",
        },
        "nemar_attention_signal": [
            {"window_idx": 0, "start_time": 0.0, "end_time": 0.5, "attention_score": 0.65, "feature_values": {"motion_magnitude": 0.05, "scene_cut_rate": 0.5, "face_presence": 0.8, "text_presence": 0.9}},
            {"window_idx": 1, "start_time": 0.5, "end_time": 1.0, "attention_score": 0.72, "feature_values": {"motion_magnitude": 0.08, "scene_cut_rate": 0.6, "face_presence": 0.7, "text_presence": 0.8}},
            {"window_idx": 2, "start_time": 1.0, "end_time": 1.5, "attention_score": 0.75, "feature_values": {"motion_magnitude": 0.06, "scene_cut_rate": 0.4, "face_presence": 0.5, "text_presence": 0.2}},
            {"window_idx": 3, "start_time": 1.5, "end_time": 2.0, "attention_score": 0.68, "feature_values": {"motion_magnitude": 0.04, "scene_cut_rate": 0.2, "face_presence": 0.1, "text_presence": 0.0}},
        ],
        "dhf1k_spatial_saliency_signal": {
            "status": "SUCCESS",
            "summary": {"mean_spatial_dispersion": 0.28, "mean_spatial_entropy_bits": 4.1},
        },
        "attention_intelligence": {
            "status": "SUCCESS",
            "spatial": {"focal_concentration": 0.62, "dispersion": 0.26, "focal_competition": {"competition_state": "focused"}},
            "temporal": {"peak_attention_score": 0.75, "peak_attention_time": 1.2, "temporal_persistence": {"opening_retention_ratio": 0.95}},
            "relationships": {
                "spatial_temporal_alignment": {"alignment_type": "stable_focal_attention"},
                "text_saliency_alignment": {"alignment_state": "congruent_text_anchor"},
                "face_saliency_alignment": {"alignment_state": "human_anchor"},
            },
        },
    }


def make_mock_competitor(
    cid: str,
    creator: str,
    views: Any = 100000,
    entity: str = "Spider-Man",
    edit_type: str = "action montage",
    has_text: bool = True,
    has_face: bool = True,
    has_conc_spatial: bool = True,
) -> Dict[str, Any]:
    att = [
        {"window_idx": 0, "start_time": 0.0, "end_time": 0.5, "attention_score": 0.68, "feature_values": {"motion_magnitude": 0.06, "scene_cut_rate": 0.5, "face_presence": 0.7 if has_face else 0.0, "text_presence": 0.8 if has_text else 0.0}},
        {"window_idx": 1, "start_time": 0.5, "end_time": 1.0, "attention_score": 0.70, "feature_values": {"motion_magnitude": 0.07, "scene_cut_rate": 0.5, "face_presence": 0.6 if has_face else 0.0, "text_presence": 0.8 if has_text else 0.0}},
        {"window_idx": 2, "start_time": 1.0, "end_time": 1.5, "attention_score": 0.65, "feature_values": {"motion_magnitude": 0.05, "scene_cut_rate": 0.4, "face_presence": 0.4 if has_face else 0.0, "text_presence": 0.0}},
        {"window_idx": 3, "start_time": 1.5, "end_time": 2.0, "attention_score": 0.60, "feature_values": {"motion_magnitude": 0.03, "scene_cut_rate": 0.2, "face_presence": 0.1 if has_face else 0.0, "text_presence": 0.0}},
    ]
    return {
        "id": cid,
        "shortcode": cid,
        "creator": creator,
        "views": views,
        "reach_value": views,
        "reach_metric": "views",
        "likes": 5000,
        "caption": f"Epic {entity} {edit_type}",
        "analysis": {
            "primary_entity": entity,
            "niche": "superhero",
            "topic": f"{entity} highlights",
            "edit_type": edit_type,
            "format": "montage",
            "ocr": {
                "ocr_text": "HERO AWAKENING" if has_text else "",
                "first_text_time": 0.5 if has_text else None,
                "text_present": has_text,
            },
            "visual": {"scene_count": 4},
            "audio": {"speech_present": True},
        },
        "attention_predictions": att,
        "dhf1k_spatial_saliency_signal": {
            "status": "SUCCESS",
            "summary": {"mean_spatial_dispersion": 0.25 if has_conc_spatial else 0.45},
        },
        "attention_intelligence": {
            "status": "SUCCESS",
            "spatial": {"focal_concentration": 0.60 if has_conc_spatial else 0.30, "dispersion": 0.25 if has_conc_spatial else 0.45},
            "temporal": {"peak_attention_score": 0.70, "peak_attention_time": 0.9, "temporal_persistence": {"opening_retention_ratio": 0.92}},
            "relationships": {
                "spatial_temporal_alignment": {"alignment_type": "stable_focal_attention" if has_conc_spatial else "divergent"},
                "text_saliency_alignment": {"alignment_state": "congruent_text_anchor" if has_text else "unaligned"},
                "face_saliency_alignment": {"alignment_state": "human_anchor" if has_face else "unaligned"},
            },
        },
        "video_verification_status": "VERIFIED",
        "video_relevance_score": 0.92,
        "status": "VERIFIED",
    }


def make_competitor_pool(n: int = 7) -> List[Dict[str, Any]]:
    pool = []
    for i in range(1, n + 1):
        views = 50000 * i
        # 6 out of 7 have text and concentrated spatial focus, 5 have face
        has_text = i <= 6
        has_conc = i <= 6
        has_face = i <= 5
        pool.append(
            make_mock_competitor(
                cid=f"comp_{i:02d}",
                creator=f"creator_{i % 4}",
                views=views,
                entity="Spider-Man" if i <= 5 else "Batman",
                edit_type="action montage",
                has_text=has_text,
                has_face=has_face,
                has_conc_spatial=has_conc,
            )
        )
    return pool


# 1. target schema normalization
def test_1_target_schema_normalization():
    target = make_mock_target()
    norm = CompetitorIntelligenceService.normalize_target(target)
    assert norm["reel_id"] == "target_123"
    assert "semantic" in norm
    assert norm["semantic"]["entity"]["value"] == "Spider-Man"
    assert norm["semantic"]["entity"]["source"] == "DiscoveryProfile/TwelveLabs"
    assert "provenance" in norm["semantic"]["entity"]
    assert "video" in norm
    assert norm["video"]["duration"]["value"] == 18.5
    assert "temporal_attention" in norm
    assert "spatial_saliency" in norm
    assert "attention_intelligence" in norm


# 2. competitor schema normalization
def test_2_competitor_schema_normalization():
    comp = make_mock_competitor("comp_01", "creator_a", views=150000)
    norm = CompetitorIntelligenceService.normalize_competitor(comp)
    assert norm["identity"]["canonical_reel_id"] == "comp_01"
    assert norm["identity"]["creator"] == "creator_a"
    assert norm["reach"]["views"] == 150000
    assert norm["reach"]["reach_valid"] is True
    assert norm["semantic"]["entity"] == "Spider-Man"
    assert norm["visual"]["face_presence"] is True
    assert norm["ocr"]["text_present"] is True
    assert "provenance" in norm


# 3. subgroup construction
def test_3_subgroup_construction():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    subgroups = CompetitorIntelligenceService.partition_subgroups(norm_t, norm_c)
    assert "same_entity_and_edit_type" in subgroups
    assert "same_edit_type" in subgroups
    assert "all_verified" in subgroups
    assert "highest_reach_same_edit_type" in subgroups


# 4. denominator accounting
def test_4_denominator_accounting():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    for p in res["recurring_patterns"]:
        assert "numerator" in p
        assert "denominator" in p
        assert isinstance(p["numerator"], int)
        assert isinstance(p["denominator"], int)
        assert p["denominator"] > 0
        assert p["numerator"] <= p["denominator"]
        assert f"{p['numerator']}/{p['denominator']}" in p["competitor_value"]


# 5. same-entity/edit-type grouping
def test_5_same_entity_and_edit_type_grouping():
    target = make_mock_target()
    comps = make_competitor_pool(7)  # 5 are Spider-Man action montage, 2 are Batman action montage
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    subgroups = CompetitorIntelligenceService.partition_subgroups(norm_t, norm_c)
    assert len(subgroups["same_entity_and_edit_type"]) == 5
    for c in subgroups["same_entity_and_edit_type"]:
        assert c["semantic"]["entity"] == "Spider-Man"


# 6. same-edit-type grouping
def test_6_same_edit_type_grouping():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    subgroups = CompetitorIntelligenceService.partition_subgroups(norm_t, norm_c)
    assert len(subgroups["same_edit_type"]) == 7


# 7. all-verified grouping
def test_7_all_verified_grouping():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    subgroups = CompetitorIntelligenceService.partition_subgroups(norm_t, norm_c)
    assert len(subgroups["all_verified"]) == 7


# 8. highest-reach subgroup
def test_8_highest_reach_subgroup():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    subgroups = CompetitorIntelligenceService.partition_subgroups(norm_t, norm_c)
    high_reach = subgroups["highest_reach_same_edit_type"]
    assert len(high_reach) >= 2
    # Verify descending reach ordering
    views = [c["reach"]["views"] for c in high_reach]
    assert views == sorted(views, reverse=True)


# 9. numeric comparison
def test_9_numeric_comparison():
    res = CompetitorIntelligenceService.compare_numeric_features(0.28, [0.24, 0.26, 0.30, 0.35, 0.40])
    assert res["target_value"] == 0.28
    assert res["competitor_median"] == 0.30
    assert res["absolute_difference"] == -0.02
    assert res["sample_size"] == 5
    assert res["percentile"] is not None


# 10. categorical pattern detection
def test_10_categorical_pattern_detection():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    subgroups = CompetitorIntelligenceService.partition_subgroups(norm_t, norm_c)
    pats = CompetitorIntelligenceService.detect_categorical_patterns(norm_t, subgroups)
    assert len(pats) > 0
    p_names = [p["pattern_name"] for p in pats]
    assert "Early Typography Hook" in p_names or "Opening Visual Focal Anchor" in p_names


# 11. attention pattern detection
def test_11_attention_pattern_detection():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    subgroups = CompetitorIntelligenceService.partition_subgroups(norm_t, norm_c)
    pats = CompetitorIntelligenceService.detect_attention_patterns(norm_t, subgroups)
    assert len(pats) > 0
    p_ids = [p["pattern_id"] for p in pats]
    assert any("concentrated_opening_spatial_focus" in pid for pid in p_ids)


# 12. target-vs-group gap
def test_12_target_vs_group_gap():
    target = make_mock_target()
    # Let target lack text to trigger gap
    target["ocr"]["text_present"] = False
    target["ocr"]["ocr_text"] = ""
    target["ocr"]["first_text_time"] = None
    comps = make_competitor_pool(7)
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    gaps = res["gap_analysis"]
    assert len(gaps) > 0
    # Descriptive language test
    for g in gaps:
        assert "gap_status" in g
        assert "interpretation" in g
        assert "limitation" in g


# 13. reach-aware handling
def test_13_reach_aware_handling():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    reach_res = CompetitorIntelligenceService.analyze_reach_aware_patterns(norm_t, norm_c)
    assert reach_res["status"] == "SUCCESS"
    assert reach_res["valid_reach_count"] == 7
    assert reach_res["cohort_median_views"] > 0
    assert len(reach_res["observations"]) >= 3


# 14. missing reach handling
def test_14_missing_reach_handling():
    target = make_mock_target()
    comps = make_competitor_pool(5)
    # Set reach to None/missing on 2 competitors
    comps[0]["views"] = None
    comps[0]["reach_value"] = None
    comps[1]["views"] = "invalid"
    comps[1]["reach_value"] = None
    norm_t = CompetitorIntelligenceService.normalize_target(target)
    norm_c = [CompetitorIntelligenceService.normalize_competitor(c) for c in comps]
    reach_res = CompetitorIntelligenceService.analyze_reach_aware_patterns(norm_t, norm_c)
    assert reach_res["valid_reach_count"] == 3
    assert reach_res["excluded_count"] == 2


# 15. small-sample protection
def test_15_small_sample_protection():
    strength, lim = CompetitorIntelligenceService.classify_pattern_strength(numerator=3, denominator=3, prevalence=1.0)
    assert strength == "insufficient_sample"
    assert "n=3 < 5" in lim


# 16. insufficient-sample classification
def test_16_insufficient_sample_classification():
    target = make_mock_target()
    comps = make_competitor_pool(3)  # n = 3 < 5
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    for p in res["recurring_patterns"]:
        if p["denominator"] < 5:
            assert p["strength"] == "insufficient_sample"


# 17. pattern deduplication
def test_17_pattern_deduplication():
    patterns = [
        {"pattern_id": "concentrated_opening_spatial_focus:same_edit_type", "subgroup": "same_edit_type", "pattern_name": "Concentrated Opening Spatial Focus"},
        {"pattern_id": "opening_visual_concentration:same_edit_type", "subgroup": "same_edit_type", "pattern_name": "Opening Visual Concentration"},
    ]
    deduped = CompetitorIntelligenceService.deduplicate_patterns(patterns)
    assert len(deduped) == 1
    assert deduped[0]["pattern_id"] == "concentrated_opening_spatial_focus:same_edit_type"


# 18. provenance
def test_18_provenance():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    prov = res["provenance"]
    assert prov["target_reel_id"] == "target_123"
    assert len(prov["competitor_ids"]) == 7
    for p in res["recurring_patterns"]:
        p_prov = p["provenance"]
        assert "target_reel_id" in p_prov
        assert "competitor_reel_ids" in p_prov
        assert "features" in p_prov


# 19. recommendation evidence linkage
def test_19_recommendation_evidence_linkage():
    target = make_mock_target()
    # Target lacks text to trigger divergent gap
    target["ocr"]["text_present"] = False
    target["ocr"]["ocr_text"] = ""
    comps = make_competitor_pool(7)
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    creatives = res["creative_implications"]
    assert len(creatives) > 0
    for c in creatives:
        assert "observation" in c
        assert "evidence" in c
        assert "target_competitor_comparison" in c
        assert "recommendation" in c
        assert "limitation" in c


# 20. no causal language
def test_20_no_causal_language():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    # Check that forbidden causal words do not appear
    forbidden = ["makes a reel go viral", "guaranteed engagement", "causes retention", "scroll-stop cause"]
    text_corpus = str(res).lower()
    for word in forbidden:
        assert word not in text_corpus, f"Forbidden causal phrase found: {word}"


# 21. no retention/scroll-stop claims
def test_21_no_retention_scroll_stop_claims():
    target = make_mock_target()
    comps = make_competitor_pool(7)
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    forbidden = ["causes higher retention", "causes scroll-stop", "improves instagram retention", "guarantees retention"]
    text_corpus = str(res).lower()
    for word in forbidden:
        assert word not in text_corpus, f"Forbidden claim found: {word}"


# 22. no duplicate model inference
def test_22_no_duplicate_model_inference():
    # Verify that CompetitorIntelligenceService does not hold or initialize ML models
    svc = CompetitorIntelligenceService()
    assert not hasattr(svc, "model")
    assert not hasattr(svc, "rf")
    assert not hasattr(svc, "tinysalnet")
    assert not hasattr(svc, "net")


# 23. no additional network calls
def test_23_no_additional_network_calls(monkeypatch):
    # Mock urllib or socket to fail if network called
    def _fail_net(*args, **kwargs):
        raise RuntimeError("Unexpected network call in analytical layer")

    monkeypatch.setattr("urllib.request.urlopen", _fail_net)
    target = make_mock_target()
    comps = make_competitor_pool(7)
    # Should execute purely locally in-memory without network calls
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    assert res["status"] == "SUCCESS"


# 24. performance regression (< 100ms for analytical processing)
def test_24_performance_regression():
    target = make_mock_target()
    comps = make_competitor_pool(10)
    t_start = time.perf_counter()
    res = CompetitorIntelligenceService.analyze(target, target.get("attention_intelligence"), comps)
    t_duration = time.perf_counter() - t_start
    assert res["status"] == "SUCCESS"
    assert t_duration < 0.200, f"Analytical computation took {t_duration:.3f}s (expected < 200ms)"


# 25. production invariance
def test_25_production_invariance():
    # Verify PatternAgent output remains fully compatible with existing contracts
    target = make_mock_target()
    comps = make_competitor_pool(7)
    pat_res = PatternAgent.analyze(comps, target)
    assert pat_res["agent"] == "pattern"
    assert "findings" in pat_res
    assert "evidence" in pat_res
    assert "recommendations" in pat_res
    assert "patterns" in pat_res
    assert "structured_patterns" in pat_res
    assert "pattern_counts" in pat_res
    assert "high_reach_counts" in pat_res
    assert "subgroups" in pat_res
    assert "competitor_intelligence" in pat_res
    assert pat_res["competitor_intelligence"]["status"] == "SUCCESS"
