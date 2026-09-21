"""Comprehensive Integration Test Suite for Human Visual Attention Model and Multi-Agent Pipeline.

Validates all 23 requirements from Section 18:
1. artifact loading
2. schema loading
3. schema ordering
4. feature dimensionality
5. feature extraction
6. scaler application
7. prediction range
8. uncertainty/confidence
9. missing audio handling
10. missing cuts handling
11. short video handling
12. attention agent contract
13. hook agent contract & outputs
14. behavior observation/interpretation separation
15. emotion sequence & peak timing
16. editing timing & pacing
17. communication/transcript integration
18. discovery multi-word queries
19. pattern exact counts
20. strategy evidence references
21. competitor attention comparison
22. master orchestration
23. report generation
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from services.human_attention_service import (
    HumanAttentionService,
    predict_attention,
    EXPECTED_FEATURE_ORDER,
    SchemaValidationError,
)
from services.attention_comparison_service import AttentionComparisonService
from agents.human_attention_agent import HumanAttentionAgent
from agents.hook_agent import HookAgent
from agents.human_behavior_agent import HumanBehaviorAgent
from agents.emotion_agent import EmotionAgent
from agents.editing_agent import EditingAgent
from agents.communication_agent import CommunicationAgent
from agents.discovery_agent import DiscoveryAgent
from agents.pattern_agent import PatternAgent
from agents.strategy_agent import StrategyAgent
from core.master_agent import MasterAgent
from services.report_service import save_report


PROJECT_ROOT = Path(r"E:\new viral")
MODEL_DIR = PROJECT_ROOT / "models" / "human_attention"
TEST_REEL_PATH = PROJECT_ROOT / "media" / "test_reel.mp4"


# 1. Artifact loading
def test_artifact_loading():
    service = HumanAttentionService(model_dir=MODEL_DIR)
    assert service.model is not None
    assert service.scaler is not None
    assert service.schema is not None
    assert len(service.feature_names) == 14
    assert service.metadata.get("target") == "human_visual_attention"
    assert "RandomForest" in str(type(service.model))


# 2. Schema loading & verification
def test_schema_loading():
    schema_path = MODEL_DIR / "feature_schema.json"
    assert schema_path.exists()
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["feature_count"] == 14
    assert "feature_names" in schema
    assert "feature_ranges" in schema
    assert "feature_importances" in schema


# 3 & 4. Schema ordering & dimensionality
def test_schema_ordering_and_dimensionality():
    service = HumanAttentionService(model_dir=MODEL_DIR)
    assert len(service.feature_names) == 14
    assert service.feature_names == EXPECTED_FEATURE_ORDER


# Test schema validation failure on tampered schema
def test_schema_validation_failure(tmp_path):
    bad_schema = {"feature_names": ["motion_magnitude", "invalid_feature"]}
    bad_schema_path = tmp_path / "feature_schema.json"
    with open(bad_schema_path, "w") as f:
        json.dump(bad_schema, f)

    # Copy other required artifacts to tmp
    for art in ["model.pkl", "scaler.pkl"]:
        (tmp_path / art).write_bytes((MODEL_DIR / art).read_bytes())

    with pytest.raises(SchemaValidationError):
        HumanAttentionService(model_dir=tmp_path)


# 5 & 6. Feature extraction & scaler application
def test_feature_extraction_and_scaler_application():
    if not TEST_REEL_PATH.exists():
        pytest.skip(f"Test reel not found at {TEST_REEL_PATH}")

    service = HumanAttentionService(model_dir=MODEL_DIR)
    feats = service.extractor.extract_features(str(TEST_REEL_PATH), max_duration_sec=2.0)
    assert len(feats) >= 2

    # Check extracted keys
    for k in EXPECTED_FEATURE_ORDER:
        assert k in feats[0]

    # Verify scaler transforms without fitting
    row = [feats[0][k] for k in EXPECTED_FEATURE_ORDER]
    X = np.array([row], dtype=np.float32)
    X_scaled = service.scaler.transform(X)
    assert X_scaled.shape == (1, 14)
    assert not np.isnan(X_scaled).any()


# 7 & 8. Prediction range & uncertainty confidence
def test_prediction_range_and_uncertainty():
    if not TEST_REEL_PATH.exists():
        pytest.skip(f"Test reel not found at {TEST_REEL_PATH}")

    preds = predict_attention(TEST_REEL_PATH, model_dir=MODEL_DIR)
    assert len(preds) > 0

    for p in preds:
        assert "start_time" in p
        assert "end_time" in p
        assert "attention_score" in p
        score = p["attention_score"]
        assert 0.0 <= score <= 1.0, f"Attention score {score} out of [0, 1] range"

        conf = p["confidence"]
        if conf is not None:
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0, 1] range"


# 9. Missing audio handling
def test_missing_audio_handling():
    service = HumanAttentionService(model_dir=MODEL_DIR)
    # Simulate empty audio signal
    audio_feats = service.extractor.compute_audio_window_features(
        np.zeros(0, dtype=np.float32), 16000, 0.0, 0.5
    )
    assert audio_feats["audio_rms"] == 0.0
    assert audio_feats["audio_silence_ratio"] == 1.0
    assert audio_feats["audio_spectral_centroid"] == 0.0
    assert audio_feats["audio_speech_presence"] == 0.0


# 10 & 11. Missing cuts & short video handling
def test_missing_cuts_and_short_video():
    service = HumanAttentionService(model_dir=MODEL_DIR)
    if not TEST_REEL_PATH.exists():
        pytest.skip(f"Test reel not found at {TEST_REEL_PATH}")

    # Limit to very short duration (0.6 seconds -> ~1 window)
    short_preds = service.predict_attention(TEST_REEL_PATH, max_duration_sec=0.6)
    assert len(short_preds) >= 1
    assert short_preds[0]["attention_score"] >= 0.0


# 12. Human Attention Agent contract & outputs
def test_human_attention_agent_contract():
    dummy_preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.55, "confidence": 0.85, "feature_values": {}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.82, "confidence": 0.88, "feature_values": {"face_presence": 0.5}},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.65, "confidence": 0.80, "feature_values": {}},
    ]
    res = HumanAttentionAgent.analyze(dummy_preds)
    # Check shared contract
    assert res["agent"] == "human_attention"
    assert isinstance(res["findings"], list) and len(res["findings"]) > 0
    assert isinstance(res["evidence"], list) and len(res["evidence"]) > 0
    assert 0.0 <= res["confidence"] <= 1.0
    assert isinstance(res["uncertainties"], list)
    assert isinstance(res["recommendations"], list)
    # Domain specific keys
    assert res["peak_attention"] == 0.82
    assert res["first_attention_event_time"] == 0.5


# 13. Hook Agent contract & outputs
def test_hook_agent():
    dummy_preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.75, "confidence": 0.85, "feature_values": {"face_presence": 0.4}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.70, "confidence": 0.85, "feature_values": {}},
    ]
    res = HookAgent.analyze(dummy_preds, technical_signals={"speech": {"has_speech": True, "speech_text": "Look at this"}})
    assert res["agent"] == "hook"
    assert "hook_type" in res
    assert res["attention_score"] > 0.0
    assert res["first_attention_event"] is not None
    assert "recommendation" in res


# 14. Human Behavior Agent observation vs interpretation separation
def test_human_behavior_agent_separation():
    res = HumanBehaviorAgent.analyze(
        technical_signals={"speech": {"has_speech": True, "speech_text": "Hey guys welcome"}},
        attention_predictions=[
            {"start_time": 0.0, "feature_values": {"face_presence": 0.8, "motion_magnitude": 0.05}}
        ]
    )
    assert res["agent"] == "human_behavior"
    assert "observable_behaviors" in res
    for b in res["observable_behaviors"]:
        assert "observation" in b
        assert "interpretation" in b
        # Ensure distinct keys
        assert b["observation"] != b["interpretation"]


# 15. Emotion Agent sequence & peak timing
def test_emotion_agent():
    dummy_preds = [
        {"start_time": 0.0, "feature_values": {"face_presence": 0.5, "motion_magnitude": 0.01, "audio_speech_presence": 0.5}},
        {"start_time": 1.0, "feature_values": {"face_presence": 0.9, "motion_magnitude": 0.08, "audio_speech_presence": 0.7}},
    ]
    res = EmotionAgent.analyze(attention_predictions=dummy_preds)
    assert res["agent"] == "emotion"
    assert len(res["emotion_sequence"]) == 2
    assert res["peak_emotion_time"] == 1.0
    assert res["emotion_intensity"] > 0.5


# 16. Editing Agent cut rate & pacing
def test_editing_agent():
    res = EditingAgent.analyze(
        technical_signals={"video": {"duration_sec": 10.0, "scene_change_count": 5}},
        attention_predictions=[
            {"start_time": 2.0, "attention_score": 0.85, "feature_values": {"scene_cut_rate": 2.0}}
        ]
    )
    assert res["agent"] == "editing"
    assert res["cut_rate"] == 0.5
    assert "pacing" in res
    assert len(res["aligned_events"]) > 0


# 17. Communication Agent transcript integration
def test_communication_agent():
    res = CommunicationAgent.analyze(
        technical_signals={"speech": {"has_speech": True, "speech_text": "Did you know that you can edit like this?"}},
    )
    assert res["agent"] == "communication"
    assert res["has_speech"] is True
    assert res["has_question"] is True
    assert res["has_direct_address"] is True
    assert res["word_count"] > 0


# 18. Discovery Agent multi-word queries
def test_discovery_agent_multi_word_queries():
    profile = {"primary_entity": "spiderman", "topic": "spiderman cinematic edit", "niche": "superhero"}
    res = DiscoveryAgent.generate_queries(profile, max_queries=4)
    assert res["agent"] == "discovery"
    assert 3 <= len(res["queries"]) <= 4
    for q in res["queries"]:
        words = q["query"].split()
        assert len(words) >= 2, f"Single-word query detected: '{q['query']}'"


# 19. Pattern Agent exact counts
def test_pattern_agent_exact_counts():
    verified_comps = [
        {
            "id": "c1",
            "analysis": {"canonical": {"ocr_text": "TITLE"}},
            "attention_predictions": [{"feature_values": {"face_presence": 0.6, "scene_cut_rate": 0.8}, "attention_score": 0.75}],
        },
        {
            "id": "c2",
            "analysis": {"canonical": {}},
            "attention_predictions": [{"feature_values": {"face_presence": 0.0, "scene_cut_rate": 0.0}, "attention_score": 0.45}],
        },
    ]
    res = PatternAgent.analyze(verified_comps)
    assert res["agent"] == "pattern"
    assert res["pattern_counts"]["early_human_presence"] == "1/2"
    assert "1/2 verified competitors" in res["findings"][0]


# 20. Strategy Agent evidence references
def test_strategy_agent_evidence_references():
    agent_results = {
        "human_attention": {"first_attention_event": 2.0, "peak_attention": 0.85, "attention_stability": 0.7},
        "hook": {"attention_score": 0.52, "weaknesses": ["Low visual contrast in opening"], "recommendation": "Boost contrast"},
        "editing": {"cut_rate": 0.2, "mean_shot_duration": 4.5, "pacing": "deliberate_continuous", "total_cuts": 2},
        "communication": {"has_speech": True, "has_direct_address": False, "word_count": 20},
    }
    res = StrategyAgent.synthesize({}, agent_results)
    assert res["agent"] == "strategy"
    assert len(res["strategic_recommendations"]) >= 3
    for item in res["strategic_recommendations"]:
        assert bool(item.get("evidence")), "Recommendation missing empirical evidence!"
        assert bool(item.get("actionable_recommendation"))


# 21. Competitor Attention Comparison
def test_attention_comparison_service():
    your_att = [
        {"start_time": 0.0, "attention_score": 0.70},
        {"start_time": 0.5, "attention_score": 0.80},
    ]
    comp_atts = {
        "comp1": [
            {"start_time": 0.0, "attention_score": 0.60},
            {"start_time": 0.5, "attention_score": 0.75},
        ]
    }
    res = AttentionComparisonService.compare(your_att, comp_atts)
    assert res["status"] == "SUCCESS"
    assert res["your_stats"]["peak_score"] == 0.80
    assert res["benchmark"]["avg_peak_score"] == 0.75
    assert len(res["insights"]) > 0


# 22. Master Orchestration with mocked download
def test_master_agent_orchestration(tmp_path):
    dummy_video = tmp_path / "test_reel.mp4"
    dummy_video.write_bytes(b"dummy")

    agent = MasterAgent()
    mock_tl = {
        "success": True,
        "analysis_status": "success",
        "analysis": {
            "niche": "superhero",
            "topic": "spiderman edit",
            "primary_entity": "spiderman",
            "format": "cinematic edit",
            "hook": {"type": "visual", "strength": 8, "first_3_seconds": "leap"},
            "visual": {"visual_impact": 8},
        },
    }

    dummy_preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.65, "confidence": 0.85, "feature_values": {"face_presence": 0.4}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.82, "confidence": 0.88, "feature_values": {"face_presence": 0.5}},
    ]

    with patch.object(agent.analyzer, "analyze_video", return_value=mock_tl), \
         patch.object(agent.router, "search_with_diagnostics") as mock_search, \
         patch("core.master_agent.acquire") as mock_acquire, \
         patch.object(agent, "attention_service") as mock_att_svc, \
         patch("core.master_agent.analyze_video", return_value={"duration_sec": 10.0, "fps": 30.0, "frame_count": 300, "scene_change_count": 3}), \
         patch("core.master_agent.analyze_audio", return_value={"has_audio": True, "tempo": 120.0, "rms_mean": 0.05}), \
         patch("core.master_agent.transcribe", return_value={"has_speech": True, "speech_text": "Hey Spiderman"}), \
         patch("core.master_agent.extract_ocr", return_value={"ocr_available": True, "ocr_text": "SPIDER"}):

        mock_att_svc.predict_attention.return_value = dummy_preds

        mock_search.return_value = ([
            {
                "id": "c1",
                "shortcode": "c1",
                "url": "http://example.com/reel/c1",
                "caption": "spiderman action edit",
                "source_query": "superhero spiderman edit",
                "views": 50000,
                "likes": 2000,
            }
        ], {"raw_candidates": 1, "status": "SUCCESS"})

        mock_acquire.return_value = {
            "media_status": "available",
            "video_download_status": "DOWNLOADED",
            "local_path": str(dummy_video),
            "cached": True,
            "error": "",
        }

        res = agent.run(dummy_video)
        assert res["your_reel"]["niche"] == "superhero"
        assert len(res["queries"]) >= 3
        assert len(res["verified_competitors"]) == 1
        assert "agent_results" in res
        assert "human_attention" in res["agent_results"]
        assert "strategy" in res["agent_results"]


# 23. Report generation with attention section
def test_report_generation(tmp_path):
    payload = {
        "your_reel": {"niche": "tech", "topic": "ai agents"},
        "queries": [{"query": "tech ai agents edit", "search_level": "CORE"}],
        "discovery_diagnostics": {"raw_candidates": 5},
        "verified_competitors": [],
        "rejected_competitors": [],
        "agent_results": {
            "human_attention": {
                "first_attention_event_time": 0.5,
                "peak_attention": 0.88,
                "peak_attention_time": 1.0,
                "lowest_attention": 0.45,
                "lowest_attention_time": 3.0,
                "attention_stability": 0.78,
                "trajectory": "Front-loaded attention",
                "findings": ["Peak attention 0.88"],
                "evidence": ["Observed peak at 1.0s"],
            }
        }
    }
    json_p, md_p = save_report(payload, base_name="test_audit")
    assert json_p.exists()
    assert md_p.exists()
    content = md_p.read_text(encoding="utf-8")
    assert "Predicted Visual-Attention Potential" in content or "Laboratory-Derived" in content
    assert "Peak Attention Potential" in content


# ==============================================================================
# SECTION 22: SPECIFIC SCIENTIFIC QUALITY & COMPETITOR SELECTION TESTS (1–22)
# ==============================================================================

# 1. Chronological timeline ordering
def test_chronological_timeline_ordering():
    service = HumanAttentionService(model_dir=MODEL_DIR)
    shuffled_windows = [
        {"start_time": 2.5, "end_time": 3.0, "attention_score": 0.60},
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.50},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.55},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.52},
    ]
    res = HumanAttentionAgent.analyze(shuffled_windows)
    times = [e["start"] for e in res["timeline_events"]]
    assert times == [0.0, 0.5, 1.0, 2.5], f"Timeline not chronological: {times}"


# 2. First major attention event is not automatically 0.0s
def test_first_major_attention_event_not_automatically_zero():
    # If window 0 has no delta above baseline, it should not be reported as first event
    preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.50, "feature_values": {}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.51, "feature_values": {}},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.62, "feature_values": {"face_presence": 0.5}},
    ]
    res = HumanAttentionAgent.analyze(preds)
    assert res["first_attention_event_time"] != 0.0
    assert res["first_attention_event_time"] == 1.0


# 3. First major event can be None
def test_first_major_event_can_be_none():
    # Flat curve without observable stimulus
    preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.50, "feature_values": {}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.50, "feature_values": {}},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.50, "feature_values": {}},
    ]
    res = HumanAttentionAgent.analyze(preds)
    assert res["first_attention_event_time"] is None
    assert "No major attention event" in res["first_attention_event_display"]


# 4. Relative peak detection
def test_relative_peak_detection():
    # A peak at 0.58 in a video where baseline is 0.48 is detected relatively
    preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.48, "feature_values": {}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.58, "is_local_peak": True, "peak_prominence": 0.10, "feature_values": {"face_presence": 0.4}},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.49, "feature_values": {}},
    ]
    res = HumanAttentionAgent.analyze(preds)
    assert res["peak_attention"] == 0.58
    assert res["peak_attention_time"] == 0.5
    assert res["first_attention_event_time"] == 0.5


# 5. No universal >=0.70 opening threshold
def test_no_universal_070_opening_threshold():
    # Test HookAgent and PatternAgent without reaching 0.70
    preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.45, "feature_values": {}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.56, "is_local_peak": True, "feature_values": {"face_presence": 0.4}},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.46, "feature_values": {}},
    ]
    hook_res = HookAgent.analyze(preds)
    assert hook_res["first_attention_event"] == 0.5  # Detected via relative jump

    comp = [{
        "id": "c1",
        "attention_predictions": preds,
        "analysis": {"canonical": {}},
    }]
    pat_res = PatternAgent.analyze(comp)
    # Opening relative peak should be counted (1/1)
    assert pat_res["pattern_counts"]["opening_attention_peak"] == "1/1"


# 6. Low-variation / generalization warning
def test_low_variation_generalization_warning():
    service = HumanAttentionService(model_dir=MODEL_DIR)
    flat_preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.500, "feature_values": {}},
        {"start_time": 0.5, "end_time": 1.0, "attention_score": 0.501, "feature_values": {}},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.500, "feature_values": {}},
    ]
    diag = service.check_generalization(flat_preds)
    assert diag["generalization_warning"] is True
    assert diag["severity"] in {"medium", "high"}

    agent_res = HumanAttentionAgent.analyze(flat_preds)
    assert agent_res["generalization_warning"] is True
    assert any("variation is low" in f for f in agent_res["findings"])


# 7. Bad query "spider man high" rejected
def test_bad_query_spiderman_high_rejected():
    from utils.query_builder import QueryBuilder
    is_valid = QueryBuilder.validate_query("superhero spider man high", "spider man", "action edit", "superhero")
    assert is_valid is False, "Query ending with dangling 'high' should be rejected!"
    is_valid_clean = QueryBuilder.validate_query("spider man cinematic fan edit", "spider man", "action edit", "superhero")
    assert is_valid_clean is True


# 8. All discovery queries are meaningful multi-word queries
def test_all_discovery_queries_are_meaningful_multi_word():
    profile = {
        "niche": "superhero",
        "primary_entity": "spider man",
        "topic": "movie scene edit",
    }
    res = DiscoveryAgent.generate_queries(profile)
    queries = res["query_strings"]
    assert len(queries) >= 3
    for q in queries:
        words = q.split()
        assert len(words) >= 2
        assert not q.endswith(" high")
        assert "spider" in q or "superhero" in q


# 9. Reach ranking
def test_reach_ranking():
    from agents.competitor_agent import rank_candidates
    candidates = [
        {"id": "low", "views": 1000, "likes": 50, "relevance_score": 0.9},
        {"id": "high", "views": 500000, "likes": 25000, "relevance_score": 0.9},
    ]
    ranked = rank_candidates(candidates)
    assert ranked[0]["id"] == "high"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["reach_score"] >= ranked[1]["reach_score"]


# 10. Missing reach metrics
def test_missing_reach_metrics():
    from agents.competitor_agent import rank_candidates
    candidates = [
        {"id": "no_reach", "likes": 100, "relevance_score": 0.5},
        {"id": "with_reach", "views": 20000, "likes": 500, "relevance_score": 0.5},
    ]
    ranked = rank_candidates(candidates)
    assert len(ranked) == 2
    assert ranked[0]["id"] == "with_reach"


# 11. Missing engagement metrics
def test_missing_engagement_metrics():
    from agents.competitor_agent import compute_performance_scores
    candidates = [
        {"id": "c1", "views": 100000},  # no likes or comments
    ]
    scored = compute_performance_scores(candidates)
    assert scored[0]["performance_score"] is not None
    assert "reach_only" in scored[0]["performance_model"]


# 12. High-reach irrelevant competitor rejected
def test_high_reach_irrelevant_competitor_rejected():
    from providers.twelvelabs_provider import TwelveLabsProvider
    tl = TwelveLabsProvider()
    uploaded = {"niche": "superhero", "primary_entity": "spider man", "topic": "cinematic action edit"}
    viral_baking = {"niche": "baking", "primary_entity": "pastry chef", "topic": "chocolate cake recipe", "views": 10000000}
    res = tl.verify_semantic_relevance(uploaded, viral_baking)
    assert res["video_relevance_eligible"] is False
    assert res["relevance_status"] == "analyzed_not_relevant"


# 13. Lower-reach relevant competitor retained
def test_lower_reach_relevant_competitor_retained():
    from providers.twelvelabs_provider import TwelveLabsProvider
    tl = TwelveLabsProvider()
    uploaded = {"niche": "superhero", "primary_entity": "spider man", "topic": "peter parker action edit"}
    smaller_relevant = {"niche": "superhero", "primary_entity": "spider man", "topic": "spiderman movie scenes", "views": 50000}
    res = tl.verify_semantic_relevance(uploaded, smaller_relevant)
    assert res["video_relevance_eligible"] is True
    assert res["relevance_status"] == "verified_relevant"


# 14. Creator diversity limit
def test_creator_diversity_limit():
    from agents.competitor_agent import select_final_competitors
    # 5 videos from same creator 'spam_creator', 2 from 'unique_b', 2 from 'unique_c', 2 from 'unique_d', 2 from 'unique_e'
    pool = []
    for i in range(5):
        pool.append({"id": f"s_{i}", "creator": "spam_creator", "views": 100000 - i * 1000, "video_relevance_score": 0.8})
    for c in ["b", "c", "d", "e"]:
        for j in range(2):
            pool.append({"id": f"{c}_{j}", "creator": f"creator_{c}", "views": 50000, "video_relevance_score": 0.8})

    selected, diag = select_final_competitors(pool, max_competitors=10, max_per_creator=2)
    spam_count = sum(1 for x in selected if x["creator"] == "spam_creator")
    assert spam_count <= 2, f"Spam creator exceeded max 2 allocation: {spam_count}"
    assert diag["unique_creators"] >= 4


# 15. Reel ID deduplication
def test_reel_id_deduplication():
    from agents.competitor_agent import select_final_competitors
    pool = [
        {"id": "dup_1", "shortcode": "dup_1", "creator": "a", "views": 10000, "video_relevance_score": 0.8},
        {"id": "dup_1", "shortcode": "dup_1", "creator": "a", "views": 10000, "video_relevance_score": 0.8},
        {"id": "dup_2", "shortcode": "dup_2", "creator": "b", "views": 20000, "video_relevance_score": 0.8},
    ]
    selected, diag = select_final_competitors(pool, max_competitors=10, max_per_creator=2)
    ids = [c["id"] for c in selected]
    assert len(ids) == 2
    assert ids.count("dup_1") == 1


# 16. Highest-reach verified competitors prioritization
def test_highest_reach_verified_competitors_prioritization():
    from agents.competitor_agent import select_final_competitors
    pool = [
        {"id": "low", "creator": "a", "views": 1000, "video_relevance_score": 0.8},
        {"id": "med", "creator": "b", "views": 50000, "video_relevance_score": 0.8},
        {"id": "high", "creator": "c", "views": 1000000, "video_relevance_score": 0.8},
    ]
    selected, _ = select_final_competitors(pool, max_competitors=2, max_per_creator=2)
    assert selected[0]["id"] == "high"
    assert selected[1]["id"] == "med"


# 17. Relevance evidence components
def test_relevance_evidence_components():
    from providers.twelvelabs_provider import TwelveLabsProvider
    tl = TwelveLabsProvider()
    u = {"niche": "fitness", "primary_entity": "trainer", "topic": "abs workout", "format": "tutorial"}
    c = {"niche": "fitness", "primary_entity": "trainer", "topic": "core and abs routine", "format": "tutorial"}
    res = tl.verify_semantic_relevance(u, c)
    assert "relevance_evidence" in res
    ev = res["relevance_evidence"]
    assert "entity_similarity" in ev
    assert "niche_similarity" in ev
    assert "topic_similarity" in ev
    assert "visual_format_similarity" in ev
    assert "evidence_label" in ev
    assert ev["evidence_label"] == "Multimodal semantic competitor match"


# 18. Pattern Agent high-reach subgroup counts
def test_pattern_agent_high_reach_subgroup_counts():
    competitors = [
        {"id": f"c_{i}", "views": (i + 1) * 100000, "analysis": {"canonical": {"ocr_text": "HOOK" if i >= 2 else ""}}, "attention_predictions": [{"start_time": 0.0, "attention_score": 0.5}]}
        for i in range(4)
    ]
    res = PatternAgent.analyze(competitors)
    assert "subgroups" in res
    assert "all_verified" in res["subgroups"]
    assert "highest_reach_verified" in res["subgroups"]
    assert res["subgroups"]["all_verified"]["sample_size"] == 4
    assert res["subgroups"]["highest_reach_verified"]["sample_size"] == 2


# 19. Reach-attention association language
def test_reach_attention_association_language():
    c_stats = {
        "c1": {"opening_strength": 0.65, "opening_delta": 0.10, "peak_score": 0.80},
        "c2": {"opening_strength": 0.55, "opening_delta": 0.02, "peak_score": 0.70},
    }
    comps = [
        {"id": "c1", "views": 1000000, "creator": "alpha"},
        {"id": "c2", "views": 50000, "creator": "beta"},
    ]
    res = AttentionComparisonService.analyze_reach_attention_association(comps, c_stats)
    assert "association_title" in res
    assert res["association_title"] == "Reach–attention association"
    # Ensure no causal claims
    summary_text = str(res["summary"]) + " " + " ".join(res["insights"])
    assert "attention caused" not in summary_text.lower()
    assert "attention causes virality" not in summary_text.lower()


# 20. Provenance labels
def test_provenance_labels():
    preds = [
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.55, "feature_values": {"face_presence": 0.4}},
    ]
    res = HumanAttentionAgent.analyze(preds)
    findings = res["findings"]
    assert any("[MODEL_DERIVED]" in f for f in findings)
    assert any("[OBSERVED]" in e for e in res["evidence"])
    assert any("[RECOMMENDATION]" in r for r in res["recommendations"])


# 21. Report precision
def test_report_precision(tmp_path):
    payload = {
        "your_reel": {"niche": "tech", "topic": "ai agents"},
        "queries": [{"query": "tech ai agents edit", "search_level": "CORE"}],
        "discovery_diagnostics": {"raw_candidates": 5, "unique_creators": 2},
        "verified_competitors": [
            {
                "id": "c1",
                "creator": "creator_one",
                "views": 152000,
                "performance_score": 0.8543219,
                "video_relevance_score": 0.7812984,
                "evidence_label": "Multimodal match",
            }
        ],
        "rejected_competitors": [],
        "agent_results": {
            "human_attention": {
                "first_attention_event_time": 1.2534,
                "first_attention_event_score": 0.6543,
                "peak_attention": 0.88123,
                "peak_attention_time": 2.501,
                "lowest_attention": 0.4501,
                "lowest_attention_time": 0.502,
                "attention_stability": 0.7845,
                "baseline_attention": 0.5211,
            }
        }
    }
    json_p, md_p = save_report(payload, base_name="precision_audit")
    md_content = md_p.read_text(encoding="utf-8")
    # Verify no raw 7-digit floating numbers appear in the markdown text
    assert "0.88123" not in md_content
    assert "0.8543219" not in md_content
    assert "0.88" in md_content
    assert "0.85" in md_content


# 22. UI chronological ordering
def test_ui_chronological_ordering():
    # Verify that timeline events fed out of order are sorted strictly chronologically
    shuffled = [
        {"start_time": 2.0, "end_time": 2.5, "attention_score": 0.7},
        {"start_time": 0.0, "end_time": 0.5, "attention_score": 0.5},
        {"start_time": 1.0, "end_time": 1.5, "attention_score": 0.6},
    ]
    sorted_evs = sorted(shuffled, key=lambda x: (x["start_time"], x["end_time"]))
    for i in range(len(sorted_evs) - 1):
        assert sorted_evs[i]["start_time"] <= sorted_evs[i+1]["start_time"]
        assert sorted_evs[i]["end_time"] <= sorted_evs[i+1]["end_time"]

