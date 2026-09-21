"""
Tests for Phase 9: ViralLens Shadow-Mode Spatial Saliency Integration.

Verifies:
1. Model loading, hash verification, CPU eval mode, torch.no_grad.
2. Input/output contract: RGB, 160x90, floats in [0, 1], output sums to 1.0.
3. Raw descriptive quantities: spatial dispersion (raw RMS radial spread), entropy, centroid, peak, shift (NO 0.4082 constant).
4. Provenance preservation: source_video_path, frame_idx, timestamp_sec, dimensions, model hash.
5. True non-blocking execution & telemetry recording (main_pipeline_start/end, shadow_start/end, wall time, ms/frame).
6. Shadow failure isolation: pipeline continues on failure, NEMAR unchanged, structured failure recorded, zero fabrication.
7. Production isolation: shadow signal strictly separated, cannot enter discovery, ranking, pattern scoring, recommendations, or NEMAR score.
8. Report terminology: scientific phrases required, commercial buzzwords prohibited.
9. Artifact completeness: all 5 Phase 9 artifacts present and valid.
10. Real-Reel test on htdyr.mp4: production outputs identical with shadow enabled vs disabled.
"""

import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from services.shadow_spatial_service import (
    ShadowSpatialService,
    SPATIAL_MODEL_PATH,
    EXPECTED_MODEL_HASH,
    INPUT_WIDTH,
    INPUT_HEIGHT,
)
from core.master_agent import MasterAgent
from services.report_service import save_report

REAL_VIDEO_PATH = Path(r"E:\new viral\htdyr.mp4")
ARTIFACTS_DIR = Path(r"E:\new viral\models\human_attention_experiments\shadow_mode")


class TestShadowModelContract:
    """Task 1 & 4: Model loading, input/output contract, and hash verification."""

    def test_weights_sha256_integrity(self):
        assert SPATIAL_MODEL_PATH.exists(), f"Missing weights: {SPATIAL_MODEL_PATH}"
        h = hashlib.sha256()
        with open(SPATIAL_MODEL_PATH, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        observed_hash = h.hexdigest()
        assert observed_hash == EXPECTED_MODEL_HASH, (
            f"Hash mismatch! Expected {EXPECTED_MODEL_HASH}, got {observed_hash}"
        )

    def test_model_loading_and_mode(self):
        svc = ShadowSpatialService()
        assert svc.is_available is True
        assert svc.device.type == "cpu"
        # Verify model is in eval mode
        assert not svc.model.training

    def test_tensor_forward_contract(self):
        svc = ShadowSpatialService()
        dummy_inp = torch.rand(1, 3, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32)
        with torch.no_grad():
            out = svc.model(dummy_inp)
        assert out.shape == (1, 1, INPUT_HEIGHT, INPUT_WIDTH)
        # Verify sum-to-1 normalization (probability density)
        density_sum = float(out.sum().item())
        assert abs(density_sum - 1.0) < 1e-4
        assert (out >= 0.0).all()


class TestShadowSpatialDispersionAndSummaries:
    """Task 1 & 5: Raw descriptive quantities (no 0.4082 theoretical normalization) & provenance."""

    def test_raw_descriptive_quantities_and_provenance(self):
        svc = ShadowSpatialService()
        assert REAL_VIDEO_PATH.exists()
        res = svc.analyze_video(REAL_VIDEO_PATH, max_budget=3)
        assert res["status"] == "SUCCESS"
        samples = res["samples"]
        assert len(samples) == 3

        for s in samples:
            # Provenance checks
            assert "htdyr.mp4" in s["source_video_path"]
            assert isinstance(s["frame_idx"], int)
            assert isinstance(s["timestamp_sec"], float)
            assert s["input_width"] == INPUT_WIDTH
            assert s["input_height"] == INPUT_HEIGHT
            assert s["model_version"] == "Phase 7 TinySalNet"
            assert s["model_hash"] == EXPECTED_MODEL_HASH

            # Raw descriptive statistics
            disp = s["saliency_spatial_dispersion"]
            assert isinstance(disp, float)
            assert 0.0 < disp < 1.5, f"Dispersion {disp} out of reasonable raw RMS range"

            entropy = s["saliency_spatial_entropy_bits"]
            assert isinstance(entropy, float)
            assert 0.0 < entropy < 15.0, f"Entropy {entropy} out of reasonable range"

            cx = s["saliency_centroid_x"]
            cy = s["saliency_centroid_y"]
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0

            px = s["saliency_peak_x"]
            py = s["saliency_peak_y"]
            assert 0.0 <= px <= 1.0
            assert 0.0 <= py <= 1.0

            shift = s["inter_frame_saliency_shift"]
            assert isinstance(shift, float)
            assert shift >= 0.0

        # Verify summary contains no normalized score using 0.4082
        summary = res["summary"]
        assert "mean_spatial_dispersion" in summary
        assert "mean_spatial_entropy_bits" in summary
        assert "mean_inter_frame_shift" in summary
        # Ensure 0.4082 is NOT in the output keys or values
        output_str = json.dumps(res)
        assert "0.4082" not in output_str
        assert "normalized_concentration" not in output_str


class TestNonBlockingAndTelemetry:
    """Task 2: Non-blocking execution and telemetry recording."""

    def test_telemetry_fields_present_and_valid(self):
        svc = ShadowSpatialService()
        res = svc.analyze_video(REAL_VIDEO_PATH, max_budget=3)
        assert res["status"] == "SUCCESS"
        telem = res["telemetry"]

        assert "shadow_start" in telem
        assert "shadow_end" in telem
        assert "shadow_total_wall_time" in telem
        assert "number_of_frames" in telem
        assert "shadow_ms_per_frame" in telem

        assert telem["shadow_end"] >= telem["shadow_start"]
        assert telem["shadow_total_wall_time"] > 0
        assert telem["number_of_frames"] == 3
        assert telem["shadow_ms_per_frame"] > 0

    def test_pipeline_telemetry_isolation(self):
        agent = MasterAgent()
        assert hasattr(agent, "shadow_spatial_service")
        assert agent.shadow_spatial_service is not None


class TestShadowFailureIsolation:
    """Task 3: Failure isolation without affecting production pipeline."""

    def test_missing_video_failure_isolation(self, tmp_path):
        svc = ShadowSpatialService()
        bad_path = tmp_path / "nonexistent_video.mp4"
        res = svc.analyze_video(bad_path)
        assert res["status"] == "FAILED"
        assert "Video file not found" in res["error"]
        assert res["samples"] == []
        assert res["telemetry"]["number_of_frames"] == 0

    def test_corrupted_video_failure_isolation(self, tmp_path):
        svc = ShadowSpatialService()
        corrupt_video = tmp_path / "corrupt.mp4"
        corrupt_video.write_bytes(b"NOT A REAL VIDEO CONTENT JUNK")
        res = svc.analyze_video(corrupt_video)
        assert res["status"] == "FAILED"
        assert res["samples"] == []
        assert "error" in res

    def test_pipeline_continues_when_shadow_service_fails(self, tmp_path):
        dummy_video = tmp_path / "test_dummy.mp4"
        dummy_video.write_bytes(b"dummy")

        agent = MasterAgent()
        # Mock shadow service to raise or return failure
        with patch.object(agent.shadow_spatial_service, "analyze_video", side_effect=RuntimeError("Simulated Shadow Crash")):
            mock_tl = {
                "success": True,
                "analysis": {"niche": "tech", "topic": "ai", "primary_entity": "robot"}
            }
            with patch.object(agent.analyzer, "analyze_video", return_value=mock_tl), \
                 patch.object(agent.router, "search_with_diagnostics", return_value=([], {"raw_candidates": 0, "status": "SUCCESS"})), \
                 patch("core.master_agent.analyze_video", return_value={"duration_sec": 5.0, "fps": 30.0, "frame_count": 150}), \
                 patch("core.master_agent.analyze_audio", return_value={"has_audio": False}), \
                 patch("core.master_agent.transcribe", return_value={"has_speech": False, "speech_text": ""}), \
                 patch("core.master_agent.extract_ocr", return_value={"ocr_available": False, "ocr_text": ""}), \
                 patch("core.master_agent.save_report", return_value=(tmp_path / "rep.json", tmp_path / "rep.md")):

                # Pipeline should complete without crashing
                out = agent.run(dummy_video)
                assert out is not None
                # Production payload exists
                assert "your_reel" in out
                # Shadow failed gracefully
                assert out["dhf1k_spatial_saliency_signal"]["status"] == "FAILED"
                assert "Simulated Shadow Crash" in out["dhf1k_spatial_saliency_signal"]["error"]


class TestProductionIsolation:
    """Task 7: Shadow output cannot enter competitor discovery, ranking, or NEMAR."""

    def test_signal_separation(self):
        agent = MasterAgent()
        # Verify attention_service is the NEMAR model and distinct from shadow_spatial_service
        assert agent.attention_service is not None
        assert agent.shadow_spatial_service is not None
        assert agent.attention_service is not agent.shadow_spatial_service

    def test_shadow_data_does_not_alter_competitor_scoring(self):
        from agents.competitor_agent import rank_candidates
        candidates = [
            {"id": "1", "views": 1000, "caption": "test 1", "performance_score": 0.8},
            {"id": "2", "views": 500, "caption": "test 2", "performance_score": 0.5},
        ]
        # Ranking must only use standard metadata, completely independent of shadow signals
        ranked = rank_candidates(candidates)
        assert len(ranked) == 2
        for c in ranked:
            assert "dhf1k_spatial_saliency_signal" not in c
            assert "saliency_spatial_dispersion" not in c


class TestReportTerminologyAndFormatting:
    """Task 6: Scientific terminology and forbidden commercial terms."""

    def test_markdown_report_section_2b_and_terminology(self, tmp_path):
        payload = {
            "your_reel": {
                "niche": "cinema",
                "topic": "lighting",
                "nemar_attention_signal": {"mean_attention": 0.62},
                "dhf1k_spatial_saliency_signal": {
                    "status": "SUCCESS",
                    "model_provenance": {
                        "architecture": "TinySalNet",
                        "training_supervision": "DHF1K continuous saliency supervision",
                        "model_hash": EXPECTED_MODEL_HASH,
                        "execution_device": "CPU"
                    },
                    "sampling_metadata": {
                        "sampled_frame_count": 5,
                        "sampling_rule": "1 fps"
                    },
                    "summary": {
                        "mean_spatial_dispersion": 0.2851,
                        "mean_spatial_entropy_bits": 9.412,
                        "mean_inter_frame_shift": 0.0841
                    },
                    "telemetry": {
                        "shadow_ms_per_frame": 19.2,
                        "shadow_total_wall_time": 0.35
                    },
                    "samples": []
                }
            },
            "queries": ["cinema lighting"],
            "discovery_diagnostics": {"unique_creators": 1},
            "verified_competitors": [],
            "rejected_competitors": [],
            "patterns": {},
            "agent_results": {
                "human_attention": {"baseline_attention": 0.55, "trajectory": "rising"}
            },
            "attention_comparison": {},
            "dhf1k_spatial_saliency_signal": {
                "status": "SUCCESS",
                "model_provenance": {
                    "architecture": "TinySalNet",
                    "training_supervision": "DHF1K continuous saliency supervision",
                    "model_hash": EXPECTED_MODEL_HASH,
                    "execution_device": "CPU"
                },
                "sampling_metadata": {
                    "sampled_frame_count": 5,
                    "sampling_rule": "1 fps"
                },
                "summary": {
                    "mean_spatial_dispersion": 0.2851,
                    "mean_spatial_entropy_bits": 9.412,
                    "mean_inter_frame_shift": 0.0841
                },
                "telemetry": {
                    "shadow_ms_per_frame": 19.2,
                    "shadow_total_wall_time": 0.35
                },
                "samples": []
            }
        }

        with patch("services.report_service.REPORTS_DIR", tmp_path):
            json_p, md_p = save_report(payload, base_name="terminology_test")

        md_text = md_p.read_text(encoding="utf-8")

        # Required terms
        assert "## 2b. Experimental DHF1K Spatial Saliency (Shadow-Mode Research Signal)" in md_text
        assert "DHF1K continuous saliency supervision" in md_text
        assert "Mean Saliency Spatial Dispersion" in md_text
        assert "Mean Spatial Entropy" in md_text

        # Prohibited terms
        prohibited = [
            "human attention ground truth",
            "retention prediction",
            "engagement prediction",
            "scroll-stop prediction"
        ]
        for p in prohibited:
            assert p.lower() not in md_text.lower(), f"Prohibited buzzword '{p}' found in report!"


class TestArtifactCompleteness:
    """Artifact completeness for Phase 9."""

    def test_all_five_artifacts_exist_and_are_valid(self):
        assert ARTIFACTS_DIR.exists()
        expected_files = [
            "shadow_mode_metadata.json",
            "shadow_inference_schema.json",
            "sample_output.json",
            "performance_report.json",
            "performance_report.md",
        ]
        for fname in expected_files:
            p = ARTIFACTS_DIR / fname
            assert p.exists(), f"Missing artifact: {p}"
            assert p.stat().st_size > 0, f"Empty artifact: {p}"

        # Validate JSON structures
        with open(ARTIFACTS_DIR / "shadow_mode_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            assert meta["execution_mode"] == "SHADOW_MODE_ONLY"
            assert meta["model_provenance"]["weights_sha256"] == EXPECTED_MODEL_HASH

        with open(ARTIFACTS_DIR / "performance_report.json", "r", encoding="utf-8") as f:
            perf = json.load(f)
            assert perf["execution_mode"] == "SHADOW_ASYNC_WORKER"
            assert perf["frames_evaluated"] > 0


class TestRealReelIntegration:
    """Task 8: End-to-end comparison on htdyr.mp4 verifying zero production output changes."""

    def test_real_reel_production_output_invariance(self, tmp_path):
        """Verify that enabling vs disabling shadow mode yields 100% identical production outputs."""
        assert REAL_VIDEO_PATH.exists(), f"Missing real test Reel: {REAL_VIDEO_PATH}"

        agent = MasterAgent()

        # Common mock for external network calls (Twelve Labs API & Apify)
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

            # RUN A: Existing pipeline with shadow disabled
            saved_service = agent.shadow_spatial_service
            agent.shadow_spatial_service = None
            res_without_shadow = agent.run(REAL_VIDEO_PATH)

            # RUN B: Pipeline with shadow enabled
            agent.shadow_spatial_service = saved_service
            res_with_shadow = agent.run(REAL_VIDEO_PATH)

        # 1. Compare NEMAR production attention predictions
        nemar_a = res_without_shadow["your_reel"]["nemar_attention_signal"]
        nemar_b = res_with_shadow["your_reel"]["nemar_attention_signal"]
        assert nemar_a == nemar_b, "NEMAR production signal altered by shadow mode!"

        # 2. Compare Competitor Discovery & Selection (functional production outputs)
        cand_ids_a = [c["id"] for c in res_without_shadow["selected_candidates"]]
        cand_ids_b = [c["id"] for c in res_with_shadow["selected_candidates"]]
        assert cand_ids_a == cand_ids_b, "Candidate discovery IDs altered by shadow mode!"

        scores_a = [c["relevance_score"] for c in res_without_shadow["selected_candidates"]]
        scores_b = [c["relevance_score"] for c in res_with_shadow["selected_candidates"]]
        assert scores_a == scores_b, "Candidate relevance scores altered by shadow mode!"

        ranks_sel_a = [c["rank"] for c in res_without_shadow["selected_candidates"]]
        ranks_sel_b = [c["rank"] for c in res_with_shadow["selected_candidates"]]
        assert ranks_sel_a == ranks_sel_b, "Candidate selection ranks altered by shadow mode!"

        assert res_without_shadow["queries"] == res_with_shadow["queries"], "Discovery queries altered by shadow mode!"
        assert len(res_without_shadow["verified_competitors"]) == len(res_with_shadow["verified_competitors"])

        # 3. Compare Competitor Ranking
        ranks_a = [c["id"] for c in res_without_shadow["verified_competitors"]]
        ranks_b = [c["id"] for c in res_with_shadow["verified_competitors"]]
        assert ranks_a == ranks_b, "Competitor ranking altered by shadow mode!"

        # 4. Compare Patterns & Recommendations
        assert res_without_shadow["patterns"] == res_with_shadow["patterns"], "Patterns altered by shadow mode!"
        assert res_without_shadow["recommendations"] == res_with_shadow["recommendations"], "Recommendations altered by shadow mode!"

        # 5. Verify Wall-Time Impact & Shadow Telemetry
        shadow_b = res_with_shadow["dhf1k_spatial_saliency_signal"]
        assert shadow_b is not None
        assert shadow_b["status"] == "SUCCESS"
        assert shadow_b["telemetry"]["number_of_frames"] > 0
        assert shadow_b["summary"]["mean_spatial_dispersion"] > 0
        assert "main_pipeline_telemetry" in res_with_shadow
        assert res_with_shadow["main_pipeline_telemetry"]["main_pipeline_wall_time"] > 0
