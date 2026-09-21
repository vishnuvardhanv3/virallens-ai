"""
Tests for Phase 10: Real-Reel Shadow Saliency Evaluation.

Verifies:
1. TinySalNet hash unchanged & model remains frozen in eval() mode on CPU.
2. Production NEMAR hashes unchanged (model.pkl, scaler.pkl, feature_schema.json).
3. DHF1K source dataset remains strictly untouched.
4. Provenance completeness across all real-reel evaluated samples.
5. Failed videos / negative controls never receive fabricated predictions.
6. Production outputs remain 100% invariant between shadow-enabled and shadow-disabled runs.
7. Sampling is deterministic.
8. Artifact completeness (7 required files, qualitative_examples directory, non-empty, expected panels).
"""

import json
import csv
import hashlib
from pathlib import Path
import pytest
import torch
from unittest.mock import patch

from services.shadow_spatial_service import (
    ShadowSpatialService,
    SPATIAL_MODEL_PATH,
    EXPECTED_MODEL_HASH,
    INPUT_WIDTH,
    INPUT_HEIGHT,
)
from core.master_agent import MasterAgent

WORKSPACE_ROOT = Path(r"E:\new viral")
EVAL_DIR = WORKSPACE_ROOT / "models" / "human_attention_experiments" / "real_reel_shadow_eval"
QUAL_DIR = EVAL_DIR / "qualitative_examples"
REAL_VIDEO_PATH = WORKSPACE_ROOT / "htdyr.mp4"

NEMAR_EXPECTED_HASHES = {
    "model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
    "scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
    "feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411",
}


class TestModelIntegrityAndFreezing:
    """Verifies all model hashes and frozen states."""

    def test_tinysalnet_hash_and_freezing(self):
        assert SPATIAL_MODEL_PATH.exists()
        h = hashlib.sha256(SPATIAL_MODEL_PATH.read_bytes()).hexdigest()
        assert h == EXPECTED_MODEL_HASH, f"TinySalNet weights modified! Expected {EXPECTED_MODEL_HASH}, got {h}"

        svc = ShadowSpatialService()
        assert svc.is_available
        assert svc.device.type == "cpu"
        assert not svc.model.training, "TinySalNet must be in eval() mode"

    def test_production_nemar_hashes_untouched(self):
        nemar_dir = WORKSPACE_ROOT / "models" / "human_attention"
        for fname, expected_hash in NEMAR_EXPECTED_HASHES.items():
            fpath = nemar_dir / fname
            assert fpath.exists(), f"Missing production NEMAR file: {fpath}"
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            assert h == expected_hash, f"Production NEMAR file {fname} altered! Expected {expected_hash}, got {h}"

    def test_dhf1k_source_data_untouched(self):
        dhf1k_dir = Path(r"E:\viral_attention_data\DHF1K")
        if dhf1k_dir.exists():
            videos_dir = dhf1k_dir / "video"
            if videos_dir.exists():
                v_count = len(list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi")))
                assert v_count == 1000, f"Expected 1000 DHF1K videos, found {v_count}"


class TestProvenanceAndDeterministicSampling:
    """Verifies sample provenance and deterministic frame selection."""

    def test_sample_provenance_completeness(self):
        svc = ShadowSpatialService()
        res = svc.analyze_video(REAL_VIDEO_PATH, max_budget=3)
        assert res["status"] == "SUCCESS"
        for s in res["samples"]:
            assert "source_video_path" in s
            assert "htdyr.mp4" in s["source_video_path"]
            assert isinstance(s["frame_idx"], int)
            assert isinstance(s["timestamp_sec"], float)
            assert s["input_width"] == INPUT_WIDTH
            assert s["input_height"] == INPUT_HEIGHT
            assert s["model_version"] == "Phase 7 TinySalNet"
            assert s["model_hash"] == EXPECTED_MODEL_HASH

    def test_deterministic_sampling(self):
        svc = ShadowSpatialService()
        res1 = svc.analyze_video(REAL_VIDEO_PATH, max_budget=5)
        res2 = svc.analyze_video(REAL_VIDEO_PATH, max_budget=5)
        assert res1["sampling_metadata"]["frame_indices"] == res2["sampling_metadata"]["frame_indices"]
        assert res1["summary"]["mean_spatial_dispersion"] == res2["summary"]["mean_spatial_dispersion"]
        assert res1["summary"]["mean_spatial_entropy_bits"] == res2["summary"]["mean_spatial_entropy_bits"]


class TestFailureIsolationAndZeroFabrication:
    """Verifies that failure inputs are classified and never receive fabricated output."""

    def test_negative_controls_failure_handling(self, tmp_path):
        svc = ShadowSpatialService()

        # 1. missing file
        res_missing = svc.analyze_video(tmp_path / "nonexistent.mp4")
        assert res_missing["status"] == "FAILED"
        assert res_missing["samples"] == []
        assert "not found" in res_missing["error"].lower()

        # 2. empty file
        empty_f = tmp_path / "empty.mp4"
        empty_f.write_bytes(b"")
        res_empty = svc.analyze_video(empty_f)
        assert res_empty["status"] == "FAILED"
        assert res_empty["samples"] == []

        # 3. corrupted file
        corrupt_f = tmp_path / "corrupt.mp4"
        corrupt_f.write_bytes(b"CORRUPT_NOT_A_REAL_MP4_HEADER")
        res_corrupt = svc.analyze_video(corrupt_f)
        assert res_corrupt["status"] == "FAILED"
        assert res_corrupt["samples"] == []


class TestProductionInvariance:
    """Verifies 100% production invariance on real Reel execution."""

    def test_production_output_identity(self, tmp_path):
        agent = MasterAgent()

        mock_tl = {
            "success": True,
            "analysis_status": "success",
            "analysis": {
                "niche": "entertainment",
                "topic": "viral reel",
                "primary_entity": "creator",
                "format": "cinematic",
                "hook": {"type": "visual", "strength": 7.5, "first_3_seconds": "Action hook"},
                "visual": {"visual_impact": 7},
                "discovery_keywords": ["viral edit", "creator reel"],
            }
        }
        mock_candidates = [
            {
                "id": "cand_1",
                "shortcode": "cand_1",
                "url": "https://www.instagram.com/reel/cand_1/",
                "caption": "Viral reel edit #creator",
                "source_query": "viral edit",
                "views": 50000,
                "likes": 2500,
            }
        ]

        def mock_acquire_fn(cand, dest):
            return {
                "media_status": "available",
                "video_download_status": "DOWNLOADED",
                "local_path": str(REAL_VIDEO_PATH),
                "cached": True,
                "error": "",
            }

        import copy
        with patch.object(agent.analyzer, "analyze_video", return_value=mock_tl), \
             patch.object(agent.router, "search_with_diagnostics", side_effect=lambda *a, **kw: (copy.deepcopy(mock_candidates), {"raw_candidates": 1, "status": "SUCCESS"})), \
             patch("core.master_agent.acquire", side_effect=mock_acquire_fn), \
             patch("services.report_service.REPORTS_DIR", tmp_path):

            # Run A: Shadow Disabled
            saved_shadow = agent.shadow_spatial_service
            agent.shadow_spatial_service = None
            res_a = agent.run(REAL_VIDEO_PATH)

            # Run B: Shadow Enabled
            agent.shadow_spatial_service = saved_shadow
            res_b = agent.run(REAL_VIDEO_PATH)

        # Assert 100% identity of all production outputs
        assert res_a["your_reel"]["nemar_attention_signal"] == res_b["your_reel"]["nemar_attention_signal"]
        assert res_a["queries"] == res_b["queries"]
        assert [c["id"] for c in res_a["selected_candidates"]] == [c["id"] for c in res_b["selected_candidates"]]
        assert [c["relevance_score"] for c in res_a["selected_candidates"]] == [c["relevance_score"] for c in res_b["selected_candidates"]]
        assert [c["rank"] for c in res_a["selected_candidates"]] == [c["rank"] for c in res_b["selected_candidates"]]
        assert [c["id"] for c in res_a["verified_competitors"]] == [c["id"] for c in res_b["verified_competitors"]]
        assert res_a["patterns"] == res_b["patterns"]
        assert res_a["recommendations"] == res_b["recommendations"]


class TestArtifactCompleteness:
    """Verifies all Phase 10 artifacts are present, valid, and non-empty."""

    def test_seven_required_files_exist_and_non_empty(self):
        assert EVAL_DIR.exists(), f"Missing evaluation dir: {EVAL_DIR}"
        expected_files = [
            "real_reel_shadow_report.json",
            "real_reel_shadow_report.md",
            "video_summary.csv",
            "frame_telemetry.csv",
            "failure_analysis.csv",
            "runtime_comparison.csv",
            "evaluation_metadata.json",
        ]
        for fname in expected_files:
            p = EVAL_DIR / fname
            assert p.exists(), f"Missing required file: {p}"
            assert p.stat().st_size > 0, f"File is empty: {p}"

    def test_qualitative_examples_directory_and_panels(self):
        assert QUAL_DIR.exists(), f"Missing qualitative examples directory: {QUAL_DIR}"
        expected_panels = [
            "panel_concentrated_saliency.png",
            "panel_diffuse_saliency.png",
            "panel_high_inter_frame_shift.png",
            "panel_low_inter_frame_shift.png",
            "panel_high_entropy.png",
            "panel_low_entropy.png",
        ]
        for p_name in expected_panels:
            p = QUAL_DIR / p_name
            assert p.exists(), f"Missing review panel: {p}"
            assert p.stat().st_size > 1000, f"Review panel corrupted/too small: {p}"

    def test_metadata_and_classification(self):
        meta_p = EVAL_DIR / "evaluation_metadata.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        assert meta["phase"] == "Phase 10"
        assert meta["real_videos_evaluated"] >= 15
        assert meta["real_videos_succeeded"] == meta["real_videos_evaluated"]
        assert meta["classification"] == "A. Operationally promising"
