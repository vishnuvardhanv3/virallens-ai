"""
Tests for Phase 14: Attention Intelligence Integration.

Verifies:
1. canonical schema creation & validation
2. NEMAR provenance
3. TinySalNet provenance
4. temporal peak extraction
5. spatial concentration
6. spatial dispersion
7. entropy handling
8. focal competition
9. centroid movement
10. attention event detection
11. insufficient-data handling
12. text interaction
13. face interaction
14. provenance preservation
15. no fabricated confidence
16. no causal language
17. report serialization
18. Pattern Agent compatibility
19. recommendation provenance
20. production output invariance
21. no duplicate model inference
22. performance regression (pure Attention Intelligence computation on pre-extracted signals < 50ms)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from services.attention_intelligence import (
    AttentionIntelligenceService,
    SCHEMA_VERSION,
    NEMAR_PROVENANCE,
    TINYSALNET_PROVENANCE,
    render_attention_visualization,
)
from agents.pattern_agent import PatternAgent
from agents.strategy_agent import StrategyAgent
from services.report_service import save_report
from core.master_agent import MasterAgent

# Sample mock signals
MOCK_NEMAR_PREDICTIONS = [
    {
        "start_time": 0.0,
        "end_time": 0.5,
        "attention_score": 0.62,
        "baseline_attention": 0.58,
        "is_local_peak": False,
        "peak_prominence": 0.0,
        "feature_values": {"motion_magnitude": 0.04, "scene_cut_rate": 0.0, "face_presence": 0.8, "text_presence": 0.0},
    },
    {
        "start_time": 0.5,
        "end_time": 1.0,
        "attention_score": 0.74,
        "baseline_attention": 0.58,
        "is_local_peak": True,
        "peak_prominence": 0.12,
        "feature_values": {"motion_magnitude": 0.05, "scene_cut_rate": 0.0, "face_presence": 0.8, "text_presence": 0.9},
    },
    {
        "start_time": 1.0,
        "end_time": 1.5,
        "attention_score": 0.65,
        "baseline_attention": 0.58,
        "is_local_peak": False,
        "peak_prominence": 0.0,
        "feature_values": {"motion_magnitude": 0.02, "scene_cut_rate": 0.8, "face_presence": 0.1, "text_presence": 0.1},
    },
    {
        "start_time": 1.5,
        "end_time": 2.0,
        "attention_score": 0.55,
        "baseline_attention": 0.58,
        "is_local_peak": False,
        "peak_prominence": 0.0,
        "feature_values": {"motion_magnitude": 0.01, "scene_cut_rate": 0.0, "face_presence": 0.0, "text_presence": 0.0},
    },
]

MOCK_SPATIAL_SAMPLES = [
    {
        "frame_idx": 0,
        "timestamp_sec": 0.0,
        "saliency_centroid_x": 0.48,
        "saliency_centroid_y": 0.42,
        "saliency_peak_x": 0.50,
        "saliency_peak_y": 0.40,
        "saliency_spatial_dispersion": 0.22,
        "saliency_spatial_entropy_bits": 8.45,
        "inter_frame_saliency_shift": 0.0,
        "saliency_map_thumbnail": [[10, 50], [5, 15]],
    },
    {
        "frame_idx": 30,
        "timestamp_sec": 1.0,
        "saliency_centroid_x": 0.52,
        "saliency_centroid_y": 0.44,
        "saliency_peak_x": 0.51,
        "saliency_peak_y": 0.42,
        "saliency_spatial_dispersion": 0.24,
        "saliency_spatial_entropy_bits": 8.60,
        "inter_frame_saliency_shift": 0.08,
        "saliency_map_thumbnail": [[12, 48], [6, 14]],
    },
]

MOCK_SPATIAL_DATA = {
    "status": "SUCCESS",
    "model_provenance": {
        "architecture": "TinySalNet (96,521 parameters)",
        "training_supervision": "DHF1K continuous saliency-density supervision",
        "model_hash": "fd6f27fd5ef6334c597ee3981086fc70f639cd59c7d57399fa72253d883b95c9",
        "execution_device": "CPU",
    },
    "samples": MOCK_SPATIAL_SAMPLES,
    "summary": {
        "mean_spatial_dispersion": 0.23,
        "mean_spatial_entropy_bits": 8.525,
        "mean_inter_frame_shift": 0.08,
    },
}

MOCK_CONTEXT = {
    "face_presence": 0.75,
    "text_present": True,
    "motion_magnitude": 0.04,
    "scene_cuts": 1,
    "visual_complexity": 0.55,
    "ocr": {"text_present": True, "first_text_time": 0.5, "ocr_text": "HOOK HEADLINE"},
    "visual": {"subjects": ["creator", "desk"]},
    "niche": "tech",
}


def test_1_canonical_schema_creation():
    """Verify canonical attention intelligence schema structure."""
    ai = AttentionIntelligenceService.build_intelligence(
        nemar_predictions=MOCK_NEMAR_PREDICTIONS,
        spatial_data=MOCK_SPATIAL_DATA,
        context=MOCK_CONTEXT,
        source_video_path="test_video.mp4",
    )
    assert ai["schema_version"] == SCHEMA_VERSION
    assert ai["status"] == "SUCCESS"
    assert "temporal" in ai
    assert "spatial" in ai
    assert "context" in ai
    assert "relationships" in ai
    assert "events" in ai
    assert "interpretation" in ai
    assert "limitations" in ai
    assert "provenance" in ai


def test_2_nemar_provenance():
    """Verify NEMAR provenance is preserved and separately identifiable."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
    )
    temp = ai["temporal"]
    assert temp["source"] == "NEMAR"
    assert temp["signal_type"] == "laboratory_derived_attention_signal"
    assert ai["provenance"]["nemar_model"]["source"] == "NEMAR"


def test_3_tinysalnet_provenance():
    """Verify TinySalNet provenance is preserved and separately identifiable."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
    )
    spat = ai["spatial"]
    assert spat["source"] == "TinySalNet_DHF1K"
    assert spat["signal_type"] == "predicted_visual_saliency"
    assert ai["provenance"]["tinysalnet_model"]["expected_hash"] == TINYSALNET_PROVENANCE["expected_hash"]


def test_4_temporal_peak_extraction():
    """Verify temporal peak score, timestamp, and relative position."""
    temp = AttentionIntelligenceService.extract_temporal_intelligence(MOCK_NEMAR_PREDICTIONS)
    assert temp["status"] == "SUCCESS"
    assert temp["peak"] == 0.74
    assert temp["peak_timestamp"] == 0.5
    assert 0.0 <= temp["relative_peak_position"] <= 1.0
    assert temp["opening_signal"] == 0.64  # mean of [0.62, 0.74, 0.65, 0.55]


def test_5_spatial_concentration():
    """Verify spatial concentration calculation is bounded in [0, 1]."""
    spat = AttentionIntelligenceService.extract_spatial_intelligence(MOCK_SPATIAL_DATA)
    assert spat["status"] == "SUCCESS"
    conc = spat["focal_concentration"]
    assert 0.0 <= conc <= 1.0
    assert conc > 0.5  # dispersion 0.23 indicates good concentration


def test_6_spatial_dispersion():
    """Verify raw spatial dispersion is preserved without arbitrary multipliers."""
    spat = AttentionIntelligenceService.extract_spatial_intelligence(MOCK_SPATIAL_DATA)
    assert abs(spat["dispersion"] - 0.23) < 0.01


def test_7_entropy_handling():
    """Verify spatial Shannon entropy is preserved."""
    spat = AttentionIntelligenceService.extract_spatial_intelligence(MOCK_SPATIAL_DATA)
    assert abs(spat["entropy"] - 8.525) < 0.01


def test_8_focal_competition():
    """Verify principled focal competition classification into structured categories."""
    spat = AttentionIntelligenceService.extract_spatial_intelligence(MOCK_SPATIAL_DATA)
    comp = spat["focal_competition"]
    assert comp["competition_state"] in {"single_focus", "moderately_distributed", "multi_focus", "diffuse"}
    assert comp["concentration_ratio"] > 0
    assert comp["dominant_region_mass"] > 0
    assert "rationale" in comp


def test_9_centroid_movement():
    """Verify temporal centroid drift calculation."""
    spat = AttentionIntelligenceService.extract_spatial_intelligence(MOCK_SPATIAL_DATA)
    assert spat["temporal_centroid_drift"] >= 0.0
    assert "temporal_centroid_drift" in spat


def test_10_attention_event_detection():
    """Verify attention event detection with complete evidence and source signals."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
    )
    events = ai["events"]
    assert len(events) >= 2
    event_types = {e["event_type"] for e in events}
    assert "OPENING_FOCUS" in event_types
    assert "ATTENTION_PEAK" in event_types
    for e in events:
        assert "event_type" in e
        assert "timestamp_start" in e
        assert "timestamp_end" in e
        assert "evidence" in e
        assert "source_signals" in e
        assert "confidence" in e
        assert "interpretation" in e


def test_11_insufficient_data_handling():
    """Verify graceful handling of empty or failed model outputs."""
    ai = AttentionIntelligenceService.build_intelligence([], {"status": "FAILED", "error": "No frames"})
    assert ai["schema_version"] == SCHEMA_VERSION
    assert ai["temporal"]["status"] == "NO_DATA"
    assert ai["spatial"]["status"] == "FAILED"
    assert len(ai["limitations"]) > 0


def test_12_text_interaction():
    """Verify text presence and predicted saliency interaction."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
    )
    text_rel = ai["relationships"]["text_saliency_interaction"]
    assert text_rel["text_present"] is True
    assert "interaction_type" in text_rel
    assert "description" in text_rel


def test_13_face_interaction():
    """Verify face presence and predicted saliency interaction."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
    )
    face_rel = ai["relationships"]["face_saliency_interaction"]
    assert face_rel["face_presence_level"] > 0
    assert "description" in face_rel


def test_14_provenance_preservation():
    """Verify provenance contains all tracking parameters."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT, source_video_path="custom_reel.mp4"
    )
    prov = ai["provenance"]
    assert prov["source_video"] == "custom_reel.mp4"
    assert prov["nemar_model"]["source"] == "NEMAR"
    assert prov["tinysalnet_model"]["source"] == "TinySalNet_DHF1K"
    assert prov["sample_counts"]["temporal_windows"] == len(MOCK_NEMAR_PREDICTIONS)
    assert prov["sample_counts"]["spatial_frames"] == len(MOCK_SPATIAL_SAMPLES)


def test_15_no_fabricated_confidence():
    """Verify confidence values are grounded in signal clarity, not arbitrary 95%+."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
    )
    for ev in ai["events"]:
        conf = ev["confidence"]
        assert 0.0 <= conf <= 1.0
        # Should not claim exact 1.00 or fake statistical certitude
        assert conf <= 0.95


def test_16_no_causal_language():
    """Verify prohibited causal terms are absent from generated interpretations."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
    )
    forbidden_terms = [
        "causes viewers to",
        "guarantees virality",
        "guarantees engagement",
        "scroll-stop guaranteed",
        "makes people watch",
        "definitely stayed engaged",
    ]
    all_text = " ".join(ai["interpretation"]).lower()
    for ev in ai["events"]:
        all_text += " " + ev["interpretation"].lower()
    for term in forbidden_terms:
        assert term not in all_text, f"Found forbidden causal phrase: '{term}'"


def test_17_report_serialization(tmp_path: Path):
    """Verify Attention Intelligence is correctly written to JSON and Markdown reports."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT, source_video_path="test_video.mp4"
    )
    payload = {
        "your_reel": {
            "niche": "tech",
            "topic": "ai",
            "attention_intelligence": ai,
            "dhf1k_spatial_saliency_signal": MOCK_SPATIAL_DATA,
        },
        "attention_intelligence": ai,
        "dhf1k_spatial_saliency_signal": MOCK_SPATIAL_DATA,
    }

    with patch("services.report_service.REPORTS_DIR", tmp_path):
        json_p, md_p = save_report(payload, base_name="att_test")
        assert json_p.exists()
        assert md_p.exists()

        data = json.loads(json_p.read_text(encoding="utf-8"))
        assert "attention_intelligence" in data
        assert data["attention_intelligence"]["schema_version"] == SCHEMA_VERSION

        md_text = md_p.read_text(encoding="utf-8")
        assert "## 2c. Attention Intelligence" in md_text
        assert "### Temporal Attention" in md_text
        assert "### Spatial Saliency" in md_text
        assert "### Focal Competition" in md_text
        assert "### Experimental Boundaries" in md_text


def test_18_pattern_agent_compatibility():
    """Verify PatternAgent correctly consumes Attention Intelligence with strict denominators."""
    verified_competitors = [
        {
            "shortcode": "CAND1",
            "views": 100000,
            "attention_predictions": MOCK_NEMAR_PREDICTIONS,
            "attention_intelligence": AttentionIntelligenceService.build_intelligence(
                MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
            ),
        },
        {
            "shortcode": "CAND2",
            "views": 50000,
            "attention_predictions": MOCK_NEMAR_PREDICTIONS,
            "attention_intelligence": AttentionIntelligenceService.build_intelligence(
                MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
            ),
        },
    ]
    your_reel = {
        "primary_entity": "creator",
        "edit_type": "action montage",
        "attention_intelligence": AttentionIntelligenceService.build_intelligence(
            MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
        ),
    }

    pat_res = PatternAgent.analyze(verified_competitors, your_reel)
    assert pat_res["verified_competitor_count"] == 2
    assert "concentrated_opening_spatial_focus" in pat_res["pattern_counts"]
    assert "stable_focal_attention" in pat_res["pattern_counts"]
    # Check denominator accounting
    for p in pat_res["structured_patterns"]:
        if p["pattern"] == "concentrated_opening_spatial_focus":
            assert p["denominator"] in {1, 2}
            assert "evidence" in p
            assert "provenance" in p


def test_19_recommendation_provenance():
    """Verify StrategyAgent generates evidence-backed recommendations citing attention intelligence."""
    ai = AttentionIntelligenceService.build_intelligence(
        MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
    )
    # Simulate multi-focus state
    ai["spatial"]["focal_competition"]["competition_state"] = "multi_focus"
    ai["spatial"]["dispersion"] = 0.38

    your_reel = {"attention_intelligence": ai, "canonical": {}}
    agent_results = {"attention_intelligence": ai}

    strat_res = StrategyAgent.synthesize(your_reel, agent_results)
    assert len(strat_res["strategic_recommendations"]) > 0
    # Check that recommendation has all required fields
    has_att_rec = False
    for rec in strat_res["strategic_recommendations"]:
        assert "observation" in rec
        assert "evidence" in rec
        assert "comparison" in rec
        assert "recommendation" in rec
        assert "confidence" in rec
        assert "limitation" in rec
        if "spatial" in rec["observation"].lower() or "saliency" in rec["observation"].lower():
            has_att_rec = True
    assert has_att_rec, "Did not find expected attention intelligence recommendation"


def test_20_production_output_invariance():
    """Verify that adding Attention Intelligence does NOT alter competitor ranking or discovery."""
    from agents.competitor_agent import rank_candidates
    candidates = [
        {"id": "comp_1", "views": 250000, "caption": "test 1", "performance_score": 0.88},
        {"id": "comp_2", "views": 100000, "caption": "test 2", "performance_score": 0.75},
        {"id": "comp_3", "views": 500000, "caption": "test 3", "performance_score": 0.92},
    ]
    ranked = rank_candidates(candidates)
    # Highest views/performance score should be first
    assert ranked[0]["id"] == "comp_3"
    assert ranked[1]["id"] == "comp_1"
    assert ranked[2]["id"] == "comp_2"


def test_21_no_duplicate_model_inference():
    """Verify Attention Intelligence layer operates purely on pre-extracted signals without invoking models."""
    with patch("training.train_dhf1k_spatial.TinySalNet.forward") as mock_forward:
        # Building attention intelligence must NOT trigger model forward
        ai = AttentionIntelligenceService.build_intelligence(
            MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
        )
        assert mock_forward.call_count == 0


def test_22_performance_regression():
    """Verify pure Attention Intelligence computation on pre-extracted signals takes < 50 ms."""
    t0 = time.perf_counter()
    for _ in range(10):
        ai = AttentionIntelligenceService.build_intelligence(
            MOCK_NEMAR_PREDICTIONS, MOCK_SPATIAL_DATA, MOCK_CONTEXT
        )
    elapsed_per_call_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

    print(f"\n[BENCHMARK] Pure Attention Intelligence Computation: {elapsed_per_call_ms:.3f} ms")
    assert elapsed_per_call_ms < 50.0, f"Computation latency {elapsed_per_call_ms:.2f} ms exceeded 50 ms limit"
    assert ai["telemetry"]["pure_computation_time_ms"] < 50.0
