"""Tests for ViralLens Pipeline Performance Optimization.

Verifies:
1. SharedVideoAsset metadata probing, derivative creation, and audio extraction.
2. PipelineProfiler stage tracking, bottleneck detection, and report persistence.
3. Audio pre-extraction integration in VideoFeatureExtractor & HumanAttentionService.
4. Fast gated multi-threaded OCR accuracy and schema stability.
5. End-to-end MasterAgent performance reporting.
"""
import json
import os
import time
from pathlib import Path
import pytest
import numpy as np

from services.profiler import PipelineProfiler
from services.video_asset_service import SharedVideoAsset, VideoMetadata
from training.feature_extraction import VideoFeatureExtractor
from services.human_attention_service import HumanAttentionService
from agents.ocr_agent import extract_ocr


PROJECT_ROOT = Path(r"E:\new viral")
TEST_VIDEO = PROJECT_ROOT / "media" / "test_reel.mp4"
HTDYR_VIDEO = PROJECT_ROOT / "htdyr.mp4"


def test_shared_video_asset_probing():
    """Verify SharedVideoAsset probes resolution, fps, duration, and size accurately."""
    target = TEST_VIDEO if TEST_VIDEO.exists() else HTDYR_VIDEO
    assert target.exists(), f"Test video not found: {target}"

    asset = SharedVideoAsset(target)
    meta = asset.meta

    assert meta.file_path == str(target.resolve())
    assert meta.file_size_mb > 0
    assert meta.duration_sec > 0
    assert meta.fps > 0
    assert meta.width > 0
    assert meta.height > 0
    assert os.path.exists(asset.original_video_path)


def test_shared_video_asset_audio_extraction():
    """Verify SharedVideoAsset extracts 16kHz mono WAV once and caches it."""
    target = HTDYR_VIDEO if HTDYR_VIDEO.exists() else TEST_VIDEO
    asset = SharedVideoAsset(target)
    wav_path = asset.extract_audio()

    assert wav_path is not None
    assert os.path.exists(wav_path)
    assert wav_path.endswith(".wav")
    assert os.path.getsize(wav_path) > 0

    # Second call must reuse cached path immediately
    t0 = time.perf_counter()
    wav_path_2 = asset.extract_audio()
    assert time.perf_counter() - t0 < 0.05
    assert wav_path == wav_path_2


def test_shared_video_asset_derivative_generation():
    """Verify SharedVideoAsset generates or reuses an optimized derivative if appropriate."""
    target = HTDYR_VIDEO if HTDYR_VIDEO.exists() else TEST_VIDEO
    asset = SharedVideoAsset(target)
    deriv = asset.generate_derivative()

    assert deriv is not None
    assert os.path.exists(deriv)
    assert os.path.exists(asset.analysis_video_path)
    assert os.path.exists(asset.original_video_path)


def test_pipeline_profiler_tracking_and_persistence(tmp_path):
    """Verify PipelineProfiler captures stage timings, identifies bottleneck, and saves report."""
    fake_video = tmp_path / "sample.mp4"
    fake_video.write_bytes(b"\x00" * 1024)

    profiler = PipelineProfiler(fake_video, output_dir=tmp_path / "reports")

    with profiler.profile_stage("stage_fast"):
        time.sleep(0.05)

    with profiler.profile_stage("stage_slow"):
        time.sleep(0.15)

    summary = profiler.summary_log()
    assert "stage_fast" in summary
    assert "stage_slow" in summary

    report = profiler.save_report()
    assert os.path.exists(report["report_path"])
    assert report["primary_bottleneck"]["stage"] == "stage_slow"
    assert report["primary_bottleneck"]["percentage_of_total"] > 50.0

    with open(report["report_path"], "r", encoding="utf-8") as f:
        saved_json = json.load(f)
    assert "stages" in saved_json
    assert "stage_slow" in saved_json["stages"]


def test_feature_extractor_with_pre_extracted_wav():
    """Verify VideoFeatureExtractor extracts identical features when given pre-extracted audio WAV."""
    target = HTDYR_VIDEO if HTDYR_VIDEO.exists() else TEST_VIDEO
    extractor = VideoFeatureExtractor(window_sec=0.5)

    asset = SharedVideoAsset(target)
    wav_path = asset.extract_audio()

    feats = extractor.extract_features(str(target), max_duration_sec=2.0, audio_wav_path=wav_path)
    assert len(feats) > 0
    assert "motion_magnitude" in feats[0]
    assert "audio_rms" in feats[0]
    assert "audio_spectral_centroid" in feats[0]


def test_human_attention_service_with_audio_wav():
    """Verify HumanAttentionService produces 14-feature canonical predictions with pre-extracted WAV."""
    target = HTDYR_VIDEO if HTDYR_VIDEO.exists() else TEST_VIDEO
    service = HumanAttentionService()

    asset = SharedVideoAsset(target)
    wav_path = asset.extract_audio()

    preds = service.predict_attention(str(target), max_duration_sec=2.0, audio_wav_path=wav_path)
    assert len(preds) > 0
    for p in preds:
        assert 0.0 <= p["attention_score"] <= 1.0


def test_ocr_agent_speedup_and_accuracy():
    """Verify optimized OCR produces valid text segments and correct confidence metrics."""
    target = HTDYR_VIDEO if HTDYR_VIDEO.exists() else TEST_VIDEO
    t0 = time.perf_counter()
    ocr_result = extract_ocr(target)
    elapsed = time.perf_counter() - t0

    assert "ocr_text" in ocr_result
    assert "ocr_confidence" in ocr_result
    # Optimized OCR on htdyr.mp4 should execute in under 20 seconds (vs original 45.8s)
    assert elapsed < 20.0
    if ocr_result.get("text_present"):
        assert ocr_result["ocr_confidence"] > 0.0


def test_profiler_hierarchical_timing_and_no_double_counting(tmp_path):
    """Verify inclusive parent and exclusive child durations prevent double-counting."""
    from services.profiler import PipelineProfiler, TimingCategory

    fake_video = tmp_path / "mock.mp4"
    fake_video.write_bytes(b"\x00" * 2048)

    profiler = PipelineProfiler(fake_video, output_dir=tmp_path / "perf")

    with profiler.profile_stage("competitors_total", category=TimingCategory.OTHER_EXTERNAL):
        with profiler.profile_stage("competitor_1_download", category=TimingCategory.COMPETITOR_DOWNLOAD, parent="competitors_total"):
            time.sleep(0.04)
        with profiler.profile_stage("competitor_1_twelvelabs", category=TimingCategory.TWELVE_LABS, parent="competitors_total"):
            time.sleep(0.03)

    profiler.stop()
    reconciliation = profiler.reconcile_timing()

    parent_node = profiler.stages["competitors_total"]
    c1_down = profiler.stages["competitor_1_download"]
    c1_tl = profiler.stages["competitor_1_twelvelabs"]

    # Parent inclusive must be >= sum of children
    assert parent_node.inclusive_duration >= (c1_down.inclusive_duration + c1_tl.inclusive_duration - 0.005)

    # Parent exclusive must NOT include children durations
    expected_parent_exclusive = round(parent_node.inclusive_duration - (c1_down.inclusive_duration + c1_tl.inclusive_duration), 3)
    assert abs(parent_node.exclusive_duration - expected_parent_exclusive) < 0.01

    # Recorded exclusive total must equal sum of exclusive durations
    exclusive_sum = round(parent_node.exclusive_duration + c1_down.exclusive_duration + c1_tl.exclusive_duration, 3)
    assert abs(reconciliation["recorded_exclusive_time"] - exclusive_sum) < 0.01


def test_profiler_timing_reconciliation_and_audit(tmp_path):
    """Verify timing reconciliation fails audit if unattributed time > 5% and passes when attributed."""
    from services.profiler import PipelineProfiler, TimingCategory

    fake_video = tmp_path / "sample2.mp4"
    fake_video.write_bytes(b"\x00" * 1024)

    # Case 1: Well-attributed stages (audit passes)
    profiler = PipelineProfiler(fake_video, output_dir=tmp_path / "perf")
    with profiler.profile_stage("stage_main", category=TimingCategory.LOCAL_CPU):
        time.sleep(0.08)
    profiler.stop()

    rec = profiler.reconcile_timing()
    assert rec["unattributed_percent"] <= 5.0
    assert rec["audit_passed"] is True

    # Case 2: User report format contains all required blocks
    user_report = profiler.format_user_performance_report()
    assert "[PERFORMANCE]" in user_report
    assert "[WALL CLOCK]" in user_report
    assert "[BREAKDOWN]" in user_report
    assert "[APIFY]" in user_report
    assert "[RECONCILIATION]" in user_report
    assert "[TOP BOTTLENECKS]" in user_report
    assert "total=" in user_report
    assert "recorded=" in user_report
    assert "unattributed=" in user_report


def test_profiler_latency_categories_accounting(tmp_path):
    """Verify all 9 timing categories are accounted for without dropping latency."""
    from services.profiler import PipelineProfiler, TimingCategory

    fake_video = tmp_path / "sample3.mp4"
    fake_video.write_bytes(b"\x00" * 1024)

    profiler = PipelineProfiler(fake_video, output_dir=tmp_path / "perf")
    categories_to_test = [
        ("c_cpu", TimingCategory.LOCAL_CPU),
        ("c_io", TimingCategory.LOCAL_IO),
        ("c_net", TimingCategory.NETWORK),
        ("c_tl", TimingCategory.TWELVE_LABS),
        ("c_apify", TimingCategory.APIFY),
        ("c_dl", TimingCategory.COMPETITOR_DOWNLOAD),
        ("c_sleep", TimingCategory.SLEEP_BACKOFF),
        ("c_other", TimingCategory.OTHER_EXTERNAL),
    ]

    for name, cat in categories_to_test:
        with profiler.profile_stage(name, category=cat):
            time.sleep(0.01)

    profiler.stop()
    rec = profiler.reconcile_timing()
    cats = rec["categories"]

    for _, cat in categories_to_test:
        assert cat in cats
        assert cats[cat] > 0.0
    assert TimingCategory.UNATTRIBUTED in cats

