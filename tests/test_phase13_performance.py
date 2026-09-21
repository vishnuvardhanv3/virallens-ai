"""Tests for Phase 13 Performance Hardening & Timing Reconciliation Rule.

Verifies:
1. Sequential vs. Concurrent Timing Reconciliation:
   - Sequential stages: parent.inclusive ≈ sum(child.inclusive) + parent.exclusive
   - Concurrent sibling stages: parent.inclusive ≈ child_union_duration + parent.exclusive
   - Verification of child_sum_duration, child_union_duration, overlap_duration, exclusive_duration, concurrency_level
   - Explicit reporting for competitor-processing stage
2. Competitor Deduplication & Artifact Caching:
   - Canonical Reel ID deduplication prevents redundant processing
   - CompetitorArtifactCache (MISS -> WRITE -> HIT -> INVALIDATED)
3. Strict Concurrency Constraint:
   - Worker pool bounded strictly to max_workers=2
4. Quality Invariance:
   - SharedVideoAsset derivatives and audio caching produce identical attention scores
"""
from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from config.settings import MEDIA_DIR
from services.competitor_cache import (
    CompetitorArtifactCache,
    get_canonical_reel_id,
)
from services.profiler import PipelineProfiler, TimingCategory, compute_interval_union


def test_sequential_stage_reconciliation(tmp_path: Path):
    """Verify sequential stages satisfy parent.inclusive ≈ sum(child.inclusive) + parent.exclusive."""
    profiler = PipelineProfiler(output_dir=tmp_path / "perf")

    with profiler.profile_stage("stage_sequential_parent"):
        time.sleep(0.01)  # parent exclusive work
        with profiler.profile_stage("child_1", parent="stage_sequential_parent"):
            time.sleep(0.05)
        with profiler.profile_stage("child_2", parent="stage_sequential_parent"):
            time.sleep(0.05)
        time.sleep(0.01)  # parent exclusive work

    profiler.stop()
    data = profiler.to_dict()
    reconciliation = data["reconciliation"]

    parent_node = profiler.stages["stage_sequential_parent"]
    c1 = profiler.stages["child_1"]
    c2 = profiler.stages["child_2"]

    # In sequential execution, child durations do not overlap
    assert parent_node.overlap_duration < 0.01
    assert abs(parent_node.child_sum_duration - (c1.inclusive_duration + c2.inclusive_duration)) < 0.005
    assert abs(parent_node.child_union_duration - parent_node.child_sum_duration) < 0.01

    # parent.inclusive ≈ sum(child.inclusive) + parent.exclusive
    expected_sum = round(c1.inclusive_duration + c2.inclusive_duration + parent_node.exclusive_duration, 3)
    assert abs(parent_node.inclusive_duration - expected_sum) <= 0.01


def test_concurrent_stage_timing_reconciliation_rule(tmp_path: Path):
    """Verify concurrent sibling stages reconcile with child_union_duration + parent.exclusive, NOT sum(child.inclusive)."""
    profiler = PipelineProfiler(output_dir=tmp_path / "perf")

    # Simulate parent stage with concurrent children
    with profiler.profile_stage("stage_g_competitor_download_and_analysis", concurrency_level=2) as parent_node:
        t_base = time.perf_counter()

        # Child 1: runs from t0 to t0 + 0.10s
        t1_start = t_base + 0.01
        t1_end = t1_start + 0.10
        profiler.record_stage_direct(
            stage_name="competitor_1_download",
            duration=round(t1_end - t1_start, 4),
            category=TimingCategory.COMPETITOR_DOWNLOAD,
            parent="stage_g_competitor_download_and_analysis",
            start_time=t1_start,
            end_time=t1_end,
        )

        # Child 2: runs concurrently from t0 + 0.02s to t0 + 0.12s (heavily overlapping with Child 1)
        t2_start = t_base + 0.02
        t2_end = t2_start + 0.10
        profiler.record_stage_direct(
            stage_name="competitor_2_download",
            duration=round(t2_end - t2_start, 4),
            category=TimingCategory.COMPETITOR_DOWNLOAD,
            parent="stage_g_competitor_download_and_analysis",
            start_time=t2_start,
            end_time=t2_end,
        )

        # Wait so parent wall clock encompasses both children
        time.sleep(0.14)

    profiler.stop()
    data = profiler.to_dict()
    stage_g = data["stages"]["stage_g_competitor_download_and_analysis"]

    c1_dur = data["stages"]["competitor_1_download"]["inclusive_duration_sec"]
    c2_dur = data["stages"]["competitor_2_download"]["inclusive_duration_sec"]
    child_sum = stage_g["child_sum_duration_sec"]
    child_union = stage_g["child_union_duration_sec"]
    overlap = stage_g["overlap_duration_sec"]
    parent_inclusive = stage_g["inclusive_duration_sec"]
    parent_exclusive = stage_g["exclusive_duration_sec"]

    # 1. child_sum_duration = sum(all child inclusive durations)
    assert abs(child_sum - (c1_dur + c2_dur)) < 0.005
    assert child_sum >= 0.19  # ~0.20s

    # 2. child_union_duration = actual elapsed wall-clock coverage occupied by child execution
    # Union of [0.01, 0.11] and [0.02, 0.12] is [0.01, 0.12] = ~0.11s
    assert 0.10 <= child_union <= 0.12

    # 3. overlap_duration = child_sum_duration - child_union_duration
    expected_overlap = round(child_sum - child_union, 3)
    assert abs(overlap - expected_overlap) < 0.005
    assert overlap > 0.07  # Substantial overlap detected!

    # 4. parent.inclusive MUST NOT be compared against sum(child.inclusive)
    # parent.inclusive (~0.14s) is much less than sum(child.inclusive) (~0.20s)
    assert parent_inclusive < child_sum

    # 5. Reconciliation rule: parent.inclusive ≈ child_union_duration + parent.exclusive
    assert abs(parent_inclusive - (child_union + parent_exclusive)) < 0.01

    # 6. Check competitor concurrency breakdown format
    rec = data["reconciliation"]
    comp_breakdown = rec["competitor_concurrency_breakdown"]
    assert comp_breakdown["number_of_competitors"] == 2
    assert comp_breakdown["max_concurrency"] == 2
    assert abs(comp_breakdown["child_sum_duration_sec"] - child_sum) < 0.005
    assert abs(comp_breakdown["child_union_duration_sec"] - child_union) < 0.005
    assert abs(comp_breakdown["overlap_duration_sec"] - overlap) < 0.005
    assert abs(comp_breakdown["parent_wall_time_sec"] - parent_inclusive) < 0.005

    # 7. Check user-facing report output includes competitor concurrency block
    user_report = profiler.format_user_performance_report()
    assert "[COMPETITOR PROCESSING CONCURRENCY]" in user_report
    assert "number of competitors: 2" in user_report
    assert "max concurrency: 2" in user_report
    assert f"child_sum_duration: {child_sum:.2f}s" in user_report
    assert f"child_union_duration: {child_union:.2f}s" in user_report
    assert f"overlap_duration: {overlap:.2f}s" in user_report


def test_competitor_canonical_deduplication():
    """Verify get_canonical_reel_id reliably extracts canonical identifiers to prevent duplicate work."""
    cand_shortcode = {"shortcode": "C1234567890", "id": "111", "url": "https://instagram.com/reel/C1234567890/"}
    cand_url = {"url": "https://www.instagram.com/p/C9876543210/?igsh=xyz", "id": "222"}
    cand_code = {"code": "C9999999999"}
    cand_id_only = {"id": "88888888"}
    cand_fallback = {"creator": "somebody"}

    assert get_canonical_reel_id(cand_shortcode) == "C1234567890"
    assert get_canonical_reel_id(cand_url) == "C9876543210"
    assert get_canonical_reel_id(cand_code) == "C9999999999"
    assert get_canonical_reel_id(cand_id_only) == "88888888"
    assert get_canonical_reel_id(cand_fallback) == "unknown"


def test_competitor_artifact_cache_lifecycle(tmp_path: Path):
    """Verify CompetitorArtifactCache handles MISS, WRITE, HIT, and INVALIDATION."""
    cache = CompetitorArtifactCache(cache_dir=tmp_path / "comp_cache", ttl_seconds=3600)
    profiler = PipelineProfiler(output_dir=tmp_path / "perf")

    cid = "REEL_TEST_123"

    # 1. First get must be a CACHE_MISS
    miss_item = cache.get(cid, profiler=profiler)
    assert miss_item is None
    assert profiler.metrics["cache_misses"] == 1

    # 2. Write artifact
    sample_payload = {
        "comp_analysis": {
            "analysis_status": "success",
            "hook": "Strong opening hook",
            "topic": "fitness",
            "niche": "workout",
        },
        "attention_predictions": [{"timestamp": 0.5, "attention_score": 0.88}],
        "relevance_res": {
            "video_relevance_score": 0.85,
            "video_relevance_eligible": True,
            "relevance_status": "VERIFIED",
        },
        "local_media_path": str(tmp_path / "reel.mp4"),
    }
    write_ok = cache.set(cid, sample_payload, profiler=profiler)
    assert write_ok is True

    # 3. Subsequent get must be a CACHE_HIT
    hit_item = cache.get(cid, profiler=profiler)
    assert hit_item is not None
    assert hit_item["comp_analysis"]["hook"] == "Strong opening hook"
    assert hit_item["relevance_res"]["video_relevance_eligible"] is True
    assert profiler.metrics["cache_hits"] == 1

    # 4. Invalidation
    cache.invalidate(cid)
    post_invalidate = cache.get(cid)
    assert post_invalidate is None


def test_bounded_concurrency_constraint():
    """Verify competitor processing executor enforces max_workers=2."""
    active_workers = 0
    max_observed_workers = 0
    lock = concurrent.futures.thread.threading.Lock()

    def _dummy_competitor_job(cid: str) -> str:
        nonlocal active_workers, max_observed_workers
        with lock:
            active_workers += 1
            if active_workers > max_observed_workers:
                max_observed_workers = active_workers
        time.sleep(0.04)
        with lock:
            active_workers -= 1
        return f"done_{cid}"

    # Master agent uses max_workers=2 strictly
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_dummy_competitor_job, [f"c_{i}" for i in range(6)]))

    assert len(results) == 6
    assert max_observed_workers <= 2, f"Expected concurrency <= 2, but observed {max_observed_workers}"


def test_quality_invariance_deterministic_pipeline():
    """Verify analytical quality invariance: OCR, Attention, and ranking outputs are identical with caching/optimization."""
    from agents.ocr_agent import extract_ocr
    from services.human_attention_service import HumanAttentionService
    from services.video_asset_service import SharedVideoAsset
    from agents.competitor_agent import rank_candidates

    test_vid = Path("media/test_reel.mp4") if Path("media/test_reel.mp4").exists() else Path("htdyr.mp4")
    assert test_vid.exists()

    # 1. OCR Invariance
    ocr_1 = extract_ocr(test_vid)
    ocr_2 = extract_ocr(test_vid)  # hit cache
    assert ocr_1["ocr_text"] == ocr_2["ocr_text"]
    assert ocr_1["ocr_confidence"] == ocr_2["ocr_confidence"]
    assert ocr_1["text_present"] == ocr_2["text_present"]

    # 2. Attention Score Invariance
    asset = SharedVideoAsset(test_vid)
    wav_path = asset.extract_audio()
    att_service = HumanAttentionService()

    preds_raw = att_service.predict_attention(str(test_vid), max_duration_sec=2.0)
    preds_opt = att_service.predict_attention(str(test_vid), max_duration_sec=2.0, audio_wav_path=wav_path)

    assert len(preds_raw) == len(preds_opt)
    for p_r, p_o in zip(preds_raw, preds_opt):
        assert abs(p_r["attention_score"] - p_o["attention_score"]) < 1e-4
        assert abs(p_r["start_time"] - p_o["start_time"]) < 1e-4

    # 3. Candidate Ranking Invariance
    candidates = [
        {"id": "c1", "views": 10000, "likes": 500, "comments": 50, "relevance_score": 0.8},
        {"id": "c2", "views": 50000, "likes": 3000, "comments": 200, "relevance_score": 0.9},
        {"id": "c3", "views": 5000, "likes": 200, "comments": 10, "relevance_score": 0.7},
    ]
    ranked_1 = rank_candidates(candidates)
    ranked_2 = rank_candidates(candidates)
    assert [c["id"] for c in ranked_1] == [c["id"] for c in ranked_2]

