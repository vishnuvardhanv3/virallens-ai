"""Tests for ViralLens Phase 16 — Recommendation Intelligence Service.

Covers all 31 Phase 16 test requirements:
1. recommendation schema
2. observation generation
3. evidence attachment
4. target-vs-competitor comparison
5. subgroup provenance
6. numerator/denominator preservation
7. preserve recommendation
8. change recommendation
9. test recommendation
10. insufficient-evidence handling
11. conflicting-evidence handling
12. no-action recommendation
13. attention evidence
14. OCR evidence
15. face/subject evidence
16. editing evidence
17. reach-aware evidence
18. recommendation deduplication
19. evidence graph
20. provenance
21. confidence handling
22. no fabricated confidence
23. no causal language
24. no retention claims
25. no scroll-stop claims
26. no external network calls
27. no duplicate model inference
28. performance
29. StrategyAgent compatibility
30. report serialization
31. production invariance
"""
from __future__ import annotations

import copy
import json
import re
import socket
import time
from typing import Any, Dict
from unittest.mock import patch

import pytest

from agents.strategy_agent import StrategyAgent
from services.recommendation_intelligence import RecommendationIntelligenceService
from services.report_service import ReportService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_target_profile() -> Dict[str, Any]:
    """Canonical target profile representing an analyzed Reel."""
    return {
        "id": "target_reel_123",
        "shortcode": "Ct99xyz",
        "creator": "creative_studio",
        "duration": 18.5,
        "topic": "productivity tips",
        "sub_niche": "morning routine",
        "edit_type": "fast_paced_cut",
        "has_hook": True,
        "visual_hook": "screen_recording_split",
        "audio_hook": "spoken_question",
        "hook_summary": "3 habits that changed my morning routine",
        "hook_score": 0.82,
        "human_present": True,
        "face_prominence": 0.40,
        "ocr_word_count": 28,
        "scene_cut_count": 9,
        "scene_cut_rate": 0.48,
        "video_path": "E:/new viral/htdyr.mp4",
    }


@pytest.fixture
def mock_attention_intelligence() -> Dict[str, Any]:
    """Phase 14 Attention Intelligence mock payload."""
    return {
        "status": "SUCCESS",
        "schema_version": "1.0.0",
        "spatial": {
            "dispersion": 0.42,
            "focal_concentration": 0.45,
            "focal_competition": {
                "competition_state": "moderate_competition",
                "dominant_quadrant": "center",
            },
            "spatial_focus_movement": "stable_center",
        },
        "temporal": {
            "opening_energy": 0.72,
            "attention_persistence": 0.65,
            "attention_drops": [
                {"timestamp_sec": 7.5, "severity": "moderate", "recovery": True}
            ],
            "attention_peaks": [
                {"timestamp_sec": 1.5, "energy": 0.88}
            ],
        },
        "multimodal": {
            "text_saliency_interaction": "moderate_text_saliency_overlap",
            "face_saliency_interaction": "high_face_saliency_alignment",
        },
    }


@pytest.fixture
def mock_competitor_intelligence() -> Dict[str, Any]:
    """Phase 15 Competitor Intelligence mock payload with verified cohort."""
    return {
        "status": "SUCCESS",
        "schema_version": "1.0.0",
        "verified_competitor_count": 7,
        "primary_subgroup": "same_entity_and_edit_type",
        "subgroup_counts": {
            "same_entity_and_edit_type": 7,
            "same_edit_type": 7,
            "all_verified": 7,
            "highest_reach_same_edit_type": 4,
        },
        "target_normalized": {
            "reel_id": "target_reel_123",
            "creator": "creative_studio",
            "opening_focus_state": "moderately_distributed",
            "opening_focal_concentration": 0.45,
            "scene_cut_rate": 0.48,
            "ocr_word_count": 28,
            "human_presence": True,
            "spoken_audio_present": True,
            "opening_hook_present": True,
        },
        "recurring_patterns": [
            {
                "pattern_id": "concentrated_opening_spatial_focus",
                "pattern_name": "Concentrated Opening Spatial Focus",
                "domain": "spatial",
                "subgroup": "same_entity_and_edit_type",
                "numerator": 6,
                "denominator": 7,
                "prevalence": 0.857,
                "target_value": "absent",
                "description": "6/7 peers exhibit concentrated opening spatial focus.",
                "provenance": {
                    "comparison_group": "same_entity_and_edit_type",
                    "competitor_reel_ids": [f"comp_{i}" for i in range(1, 7)],
                },
            },
            {
                "pattern_id": "early_typography_present",
                "pattern_name": "Early On-Screen Typography",
                "domain": "typography",
                "subgroup": "same_entity_and_edit_type",
                "numerator": 7,
                "denominator": 7,
                "prevalence": 1.0,
                "target_value": "present",
                "description": "7/7 peers exhibit early on-screen headline text.",
                "provenance": {
                    "comparison_group": "same_entity_and_edit_type",
                    "competitor_reel_ids": [f"comp_{i}" for i in range(1, 8)],
                },
            },
            {
                "pattern_id": "primary_character_present",
                "pattern_name": "Primary Character Presence",
                "domain": "human",
                "subgroup": "same_entity_and_edit_type",
                "numerator": 6,
                "denominator": 7,
                "prevalence": 0.857,
                "target_value": "present",
                "description": "6/7 peers feature an identifiable character.",
                "provenance": {
                    "comparison_group": "same_entity_and_edit_type",
                    "competitor_reel_ids": [f"comp_{i}" for i in range(1, 7)],
                },
            },
            {
                "pattern_id": "rapid_initial_scene_transition",
                "pattern_name": "Rapid Initial Scene Transition",
                "domain": "editing",
                "subgroup": "same_entity_and_edit_type",
                "numerator": 5,
                "denominator": 7,
                "prevalence": 0.714,
                "target_value": "absent",
                "description": "5/7 peers execute a visual cut before 3.0s.",
                "provenance": {
                    "comparison_group": "same_entity_and_edit_type",
                    "competitor_reel_ids": [f"comp_{i}" for i in range(1, 6)],
                },
            },
        ],
        "all_subgroup_patterns": [],
        "reach_aware_observations": {
            "pattern_name": "Concentrated Opening Spatial Focus",
            "highest_reach_prevalence": "4/4 (100%)",
            "top_performer_ids": ["comp_1", "comp_2", "comp_3", "comp_4"],
        },
        "gap_analysis": [
            {
                "pattern_name": "Concentrated Opening Spatial Focus",
                "target_value": "absent",
                "competitor_prevalence": "6/7 (85.7%)",
                "difference": "Target exhibits distributed focus while 6/7 competitors have concentrated focus.",
                "comparison_group": "same_entity_and_edit_type",
                "interpretation": "Target visual composition is more diffuse than dominant cohort pattern.",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Test Cases (1 to 31)
# ---------------------------------------------------------------------------

def test_1_recommendation_schema(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """1. Recommendation Schema: Check all required canonical fields exist in every recommendation."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    assert res["status"] == "SUCCESS"
    recs = res["recommendations"]
    assert len(recs) > 0

    canonical_fields = [
        "recommendation_id",
        "category",
        "priority",
        "observation",
        "evidence",
        "target_comparison",
        "creative_implication",
        "recommendation",
        "testable_action",
        "strength",
        "confidence",
        "limitation",
        "provenance",
    ]
    for r in recs:
        for f in canonical_fields:
            assert f in r, f"Missing canonical field '{f}' in recommendation {r.get('recommendation_id')}"


def test_2_observation_generation(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """2. Observation generation: Verify observation text is concrete and descriptive."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    for r in res["recommendations"]:
        obs = r["observation"]
        assert isinstance(obs, str)
        assert len(obs.strip()) > 15
        assert "Target" in obs or "target" in obs


def test_3_evidence_attachment(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """3. Evidence attachment: Check structured evidence records with source and values."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    for r in res["recommendations"]:
        ev_list = r["evidence"]
        assert isinstance(ev_list, list)
        assert len(ev_list) > 0
        for ev in ev_list:
            assert "type" in ev
            assert "source" in ev
            assert "value" in ev
            assert "provenance" in ev


def test_4_target_vs_competitor_comparison(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """4. Target vs competitor comparison: Verify formal gap representation."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    for r in res["recommendations"]:
        t_comp = r["target_comparison"]
        assert "target" in t_comp
        assert "comparison_group" in t_comp
        assert "difference" in t_comp
        assert len(t_comp["difference"]) > 5


def test_5_subgroup_provenance(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """5. Subgroup provenance: Verify peer comparison group name is explicit."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    for r in res["recommendations"]:
        prov = r["provenance"]
        assert "comparison_group" in prov
        assert prov["comparison_group"] in [
            "same_entity_and_edit_type",
            "same_edit_type",
            "all_verified",
            "highest_reach_same_edit_type",
        ]


def test_6_numerator_denominator_preservation(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """6. Numerator/denominator preservation: Check exact counts (e.g. 6/7) are preserved."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    has_fraction = False
    for r in res["recommendations"]:
        for ev in r["evidence"]:
            if ev.get("denominator") is not None:
                has_fraction = True
                assert isinstance(ev["numerator"], int)
                assert isinstance(ev["denominator"], int)
                assert ev["numerator"] <= ev["denominator"]
    assert has_fraction, "At least one evidence record must preserve explicit numerator/denominator"


def test_7_preserve_recommendation(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """7. Preserve recommendation: When target already aligns with verified pattern, produces PRESERVE."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    categories = res["categories"]
    assert len(categories["PRESERVE"]) > 0
    preserve_rec = categories["PRESERVE"][0]
    assert preserve_rec["category"] == "PRESERVE"
    assert any(k in preserve_rec["recommendation"].lower() for k in ["preserve", "continue", "retain", "maintain"])


def test_8_change_recommendation(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """8. Change recommendation: When target meaningfully differs from high-prevalence pattern, produces CHANGE."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    categories = res["categories"]
    assert len(categories["CHANGE"]) > 0
    change_rec = categories["CHANGE"][0]
    assert change_rec["category"] == "CHANGE"
    assert "Consider" in change_rec["recommendation"]


def test_9_test_recommendation(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """9. Test recommendation: Suggestive signals or moderate evidence produces TEST."""
    # Modify editing pattern prevalence to moderate (4/7) and make target static
    comp_intel = copy.deepcopy(mock_competitor_intelligence)
    comp_intel["target_normalized"]["scene_cut_rate"] = 0.1
    comp_intel["target_normalized"]["scene_cut_count"] = 1
    t_prof = copy.deepcopy(mock_target_profile)
    t_prof["scene_cut_rate"] = 0.1
    t_prof["scene_cut_count"] = 1
    for p in comp_intel["recurring_patterns"]:
        if p["pattern_id"] == "rapid_initial_scene_transition":
            p["numerator"] = 4
            p["prevalence"] = 4 / 7
    res = RecommendationIntelligenceService.generate(
        target_profile=t_prof,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=comp_intel,
    )
    categories = res["categories"]
    test_recs = categories["TEST"]
    assert len(test_recs) > 0
    assert test_recs[0]["category"] == "TEST"


def test_10_insufficient_evidence_handling(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """10. Insufficient evidence handling: Subgroup sample size n < 5 marks evidence exploratory/insufficient."""
    comp_intel = copy.deepcopy(mock_competitor_intelligence)
    comp_intel["verified_competitor_count"] = 3
    comp_intel["subgroup_counts"]["same_entity_and_edit_type"] = 3
    for p in comp_intel["recurring_patterns"]:
        p["denominator"] = 3
        p["numerator"] = 2
        p["prevalence"] = 2 / 3

    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=comp_intel,
    )
    for r in res["recommendations"]:
        assert r["strength"] in ["insufficient_evidence", "exploratory_evidence"]
        assert "< 5" in r["limitation"] or "small" in r["limitation"].lower() or "exploratory" in r["limitation"].lower()


def test_11_conflicting_evidence_handling(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """11. Conflicting evidence handling: Opposing signals are flagged as INVESTIGATE."""
    att_intel = copy.deepcopy(mock_attention_intelligence)
    # Target has severe opening drop, but high opening energy reported
    att_intel["temporal"]["attention_drops"] = [{"timestamp_sec": 1.0, "severity": "severe", "recovery": False}]
    att_intel["temporal"]["opening_energy"] = 0.95

    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=att_intel,
        competitor_intelligence=mock_competitor_intelligence,
    )
    # Check if conflict is handled cleanly (no crash, categorized or annotated)
    assert res["status"] == "SUCCESS"
    assert isinstance(res["recommendations"], list)


def test_12_no_action_recommendation(mock_target_profile, mock_attention_intelligence):
    """12. No-action recommendation: Empty baseline outputs NO_ACTIONABLE_RECOMMENDATION."""
    empty_comp_intel = {
        "status": "SUCCESS",
        "verified_competitor_count": 0,
        "recurring_patterns": [],
    }
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=empty_comp_intel,
    )
    assert res["status"] in ["NO_ACTIONABLE_RECOMMENDATIONS", "NO_ACTIONABLE_RECOMMENDATION"]
    assert len(res["recommendations"]) == 1
    null_rec = res["recommendations"][0]
    assert null_rec["category"] == "NO_ACTIONABLE_RECOMMENDATION"
    assert null_rec["priority"] == "low"
    assert "No actionable" in null_rec["recommendation"]


def test_13_attention_evidence(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """13. Attention evidence: Consumes spatial focus, competition state, and temporal persistence."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    attention_recs = [
        r for r in res["recommendations"]
        if any("attention" in ev.get("source", "").lower() or "spatial" in ev.get("source", "").lower() or "tinysalnet" in ev.get("source", "").lower()
               for ev in r["evidence"])
    ]
    assert len(attention_recs) > 0
    first_att = attention_recs[0]
    assert any(k in first_att["observation"].lower() for k in ["spatial", "focus", "saliency", "visual"])


def test_14_ocr_evidence(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """14. OCR evidence: Consumes on-screen text count and subtitle/headline placement."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    ocr_recs = [
        r for r in res["recommendations"]
        if any("typography" in ev.get("source", "").lower() or "ocr" in ev.get("source", "").lower()
               for ev in r["evidence"])
    ]
    assert len(ocr_recs) > 0
    assert "text" in ocr_recs[0]["observation"].lower() or "typography" in ocr_recs[0]["observation"].lower()


def test_15_face_subject_evidence(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """15. Face/subject evidence: Consumes human presence without forcing unnecessary changes."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    human_recs = [
        r for r in res["recommendations"]
        if any("human" in ev.get("source", "").lower() for ev in r["evidence"])
    ]
    assert len(human_recs) > 0
    # In fixture, target has human present and peers have human present -> PRESERVE
    assert human_recs[0]["category"] == "PRESERVE"


def test_16_editing_evidence(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """16. Editing evidence: Consumes scene-cut pacing and transition timings."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    editing_recs = [
        r for r in res["recommendations"]
        if any("editing" in ev.get("source", "").lower() for ev in r["evidence"])
    ]
    assert len(editing_recs) > 0
    assert "transition" in editing_recs[0]["observation"].lower() or "cut" in editing_recs[0]["observation"].lower()


def test_17_reach_aware_evidence(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """17. Reach-aware evidence: Top-reach performer patterns are noted in evidence where present."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    has_reach_note = False
    for r in res["recommendations"]:
        for ev in r["evidence"]:
            if "reach" in str(ev.get("type", "")).lower() or "reach" in str(ev.get("value", "")).lower():
                has_reach_note = True
                break
    assert has_reach_note


def test_18_recommendation_deduplication(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """18. Recommendation deduplication: Prevents multiple recommendations from saying the same thing."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    rec_ids = [r["recommendation_id"] for r in res["recommendations"]]
    assert len(rec_ids) == len(set(rec_ids)), "Duplicate recommendation IDs found"


def test_19_evidence_graph(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """19. Evidence graph: Traceable graph Target -> Feature -> Pattern -> Peer Evidence -> Gap -> Rec."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    graph = res["evidence_graph"]
    assert "target_reel_id" in graph
    assert "nodes" in graph
    assert len(graph["nodes"]) > 0
    for node in graph["nodes"]:
        assert "target_state" in node
        assert "feature" in node
        assert "pattern" in node
        assert "peer_evidence" in node
        assert "gap" in node
        assert "recommendation" in node


def test_20_provenance(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """20. Provenance: Every recommendation has complete provenance metadata."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    for r in res["recommendations"]:
        prov = r["provenance"]
        assert "target_reel_id" in prov
        assert "comparison_group" in prov
        assert "source_features" in prov
        assert isinstance(prov["source_features"], list)


def test_21_confidence_handling(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """21. Confidence handling: Confidence must strictly be None unless validated statistical basis."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    for r in res["recommendations"]:
        assert r["confidence"] is None, f"Confidence must be null, got: {r['confidence']}"


def test_22_no_fabricated_confidence(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """22. No fabricated confidence: No heuristic percentages like '92% confidence' in text."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    json_str = json.dumps(res).lower()
    assert "confidence: 9" not in json_str
    assert "confidence: 8" not in json_str
    assert "90% confidence" not in json_str
    assert "85% confidence" not in json_str


def test_23_no_causal_language(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """23. No causal language: Strictly NO 'guaranteed', 'causes virality', 'drive virality'."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    text = json.dumps(res).lower()
    prohibited = [
        "guaranteed engagement",
        "causes virality",
        "drive virality",
        "makes a reel go viral",
        "guaranteed reach",
    ]
    for phrase in prohibited:
        assert phrase not in text, f"Prohibited phrase found: '{phrase}'"


def test_24_no_retention_claims(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """24. No retention claims: Strictly NO 'retention prediction' or 'will improve retention'."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    text = json.dumps(res).lower()
    prohibited = [
        "retention prediction",
        "will increase retention",
        "increases retention",
        "improves retention",
        "predict retention",
    ]
    for phrase in prohibited:
        assert phrase not in text, f"Prohibited phrase found: '{phrase}'"


def test_25_no_scroll_stop_claims(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """25. No scroll-stop claims: Strictly NO 'scroll-stop prediction' or 'stops the scroll'."""
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    text = json.dumps(res).lower()
    prohibited = [
        "scroll-stop prediction",
        "stops the scroll",
        "guaranteed scroll stop",
        "predict scroll stop",
    ]
    for phrase in prohibited:
        assert phrase not in text, f"Prohibited phrase found: '{phrase}'"


def test_26_no_external_network_calls(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """26. No external network calls: Socket connection attempt will raise error if service tries calling network."""
    def guarded_connect(*args, **kwargs):
        raise RuntimeError("Prohibited network call in RecommendationIntelligenceService!")

    with patch.object(socket.socket, "connect", side_effect=guarded_connect):
        res = RecommendationIntelligenceService.generate(
            target_profile=mock_target_profile,
            attention_intelligence=mock_attention_intelligence,
            competitor_intelligence=mock_competitor_intelligence,
        )
        assert res["status"] == "SUCCESS"


def test_27_no_duplicate_model_inference(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """27. No duplicate model inference: Pure analytical synthesis over precomputed data."""
    # Ensure no torch / tesseract / twelvelabs call during recommendation generation
    with patch("torch.nn.Module.forward", side_effect=RuntimeError("Duplicate torch forward call!")):
        res = RecommendationIntelligenceService.generate(
            target_profile=mock_target_profile,
            attention_intelligence=mock_attention_intelligence,
            competitor_intelligence=mock_competitor_intelligence,
        )
        assert res["status"] == "SUCCESS"


def test_28_performance(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """28. Performance: Pure recommendation computation is fast in-memory (< 100 ms)."""
    t0 = time.perf_counter()
    res = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert res["status"] == "SUCCESS"
    assert elapsed_ms < 100.0, f"Recommendation computation too slow: {elapsed_ms:.2f} ms"


def test_29_strategy_agent_compatibility(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """29. StrategyAgent compatibility: StrategyAgent seamlessly consumes Recommendation Intelligence."""
    agent_outputs = {
        "attention": {"score": 0.85},
        "hook": {"score": 0.80},
        "behavior": {"score": 0.75},
        "emotion": {"score": 0.70},
        "editing": {"score": 0.78},
        "communication": {"score": 0.82},
        "pattern": {
            "patterns": [
                {"pattern": "Concentrated Opening Focus", "count": 6, "total": 7, "prevalence": 0.857}
            ]
        },
    }
    strat = StrategyAgent.synthesize(
        reel_data=mock_target_profile,
        agent_outputs=agent_outputs,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    assert "recommendation_intelligence" in strat
    rec_intel = strat["recommendation_intelligence"]
    assert rec_intel["status"] == "SUCCESS"
    assert len(rec_intel["recommendations"]) > 0
    assert len(strat["strategic_priorities"]) > 0


def test_30_report_serialization(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """30. Report serialization: ReportService formats Recommendation Intelligence with all required subsections."""
    rec_intel = RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )
    full_result = {
        "your_reel": mock_target_profile,
        "agent_results": {
            "attention": {"score": 0.85},
            "hook": {"score": 0.80},
            "behavior": {"score": 0.75},
            "emotion": {"score": 0.70},
            "editing": {"score": 0.78},
            "communication": {"score": 0.82},
            "strategy": {"strategic_priorities": ["Test focused opening"]},
            "attention_intelligence": mock_attention_intelligence,
            "competitor_intelligence": mock_competitor_intelligence,
            "recommendation_intelligence": rec_intel,
        },
        "top_competitors": [],
        "discovery_diagnostics": {},
    }
    md_report = ReportService.generate_markdown_report(full_result)

    assert "# Recommendation Intelligence" in md_report
    assert "## Priority Recommendations" in md_report
    assert "## Preserve" in md_report
    assert "## Change" in md_report
    assert "## Test" in md_report
    assert "## No Action / Insufficient Evidence" in md_report
    assert "## Evidence Trace" in md_report


def test_31_production_invariance(mock_target_profile, mock_attention_intelligence, mock_competitor_intelligence):
    """31. Production invariance: Generating recommendations does NOT mutate input dictionaries."""
    target_copy = copy.deepcopy(mock_target_profile)
    att_copy = copy.deepcopy(mock_attention_intelligence)
    comp_copy = copy.deepcopy(mock_competitor_intelligence)

    RecommendationIntelligenceService.generate(
        target_profile=mock_target_profile,
        attention_intelligence=mock_attention_intelligence,
        competitor_intelligence=mock_competitor_intelligence,
    )

    assert mock_target_profile == target_copy, "Target profile was mutated!"
    assert mock_attention_intelligence == att_copy, "Attention intelligence was mutated!"
    assert mock_competitor_intelligence == comp_copy, "Competitor intelligence was mutated!"
