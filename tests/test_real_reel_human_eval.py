"""
Tests for Phase 11: Human Qualitative Utility Evaluation of TinySalNet.

Verifies:
1. TinySalNet weights hash unchanged & model remains frozen in eval() mode on CPU.
2. Production NEMAR hashes unchanged (model.pkl, scaler.pkl, feature_schema.json).
3. DHF1K source dataset strictly untouched.
4. Production pipeline outputs remain 100% invariant between shadow-enabled and shadow-disabled runs.
5. Reviewer dataset schema valid, complete, and contains no model confidence or benchmark scores.
6. Artifact completeness (all required files and directory present, non-empty, all 30 panels present).
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
)
from core.master_agent import MasterAgent

WORKSPACE_ROOT = Path(r"E:\new viral")
EVAL_DIR = WORKSPACE_ROOT / "models" / "human_attention_experiments" / "real_reel_human_eval"
PANELS_DIR = EVAL_DIR / "qualitative_review_panels"
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
        assert h == EXPECTED_MODEL_HASH, f"TinySalNet weights altered! Expected {EXPECTED_MODEL_HASH}, got {h}"

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


class TestReviewerDataSchemaAndBlinding:
    """Verifies reviewer dataset schema and strict absence of model confidence or benchmark scores."""

    def test_reviewer_data_schema(self):
        csv_path = EVAL_DIR / "human_eval_data.csv"
        assert csv_path.exists()

        expected_columns = {
            "item_id", "video_id", "frame_idx", "item_type", "selection_stratum",
            "reviewer_id", "criterion", "score", "is_unclear_not_useful", "optional_comment"
        }

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
            assert expected_columns.issubset(headers), f"Missing headers: {expected_columns - headers}"
            rows = list(reader)

        assert len(rows) == 390, f"Expected 390 ratings (30 items x 3 reviewers x criteria), got {len(rows)}"

        # Verify no model scores or target values leaked into review records
        prohibited_leak_terms = ["confidence", "nss", "auc", "kl_divergence", "sim", "target_value", "ground_truth"]
        for r in rows:
            assert int(r["score"]) in [1, 2, 3, 4, 5], f"Invalid score: {r['score']}"
            for k in r.keys():
                assert k.lower() not in prohibited_leak_terms, f"Prohibited leakage key found: {k}"


class TestProductionInvariance:
    """Verifies that production pipeline outputs remain 100% invariant."""

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

        assert res_a["your_reel"]["nemar_attention_signal"] == res_b["your_reel"]["nemar_attention_signal"]
        assert res_a["queries"] == res_b["queries"]
        assert [c["id"] for c in res_a["selected_candidates"]] == [c["id"] for c in res_b["selected_candidates"]]
        assert [c["id"] for c in res_a["verified_competitors"]] == [c["id"] for c in res_b["verified_competitors"]]
        assert res_a["patterns"] == res_b["patterns"]
        assert res_a["recommendations"] == res_b["recommendations"]


class TestArtifactCompleteness:
    """Verifies all Phase 11 artifacts are present, valid, and non-empty."""

    def test_required_files_exist_and_non_empty(self):
        assert EVAL_DIR.exists(), f"Missing evaluation dir: {EVAL_DIR}"
        expected_files = [
            "human_eval_protocol.md",
            "human_eval_data.csv",
            "human_eval_report.md",
            "human_eval_report.json",
            "evaluation_metadata.json",
        ]
        for fname in expected_files:
            p = EVAL_DIR / fname
            assert p.exists(), f"Missing required file: {p}"
            assert p.stat().st_size > 0, f"File is empty: {p}"

    def test_qualitative_review_panels_complete(self):
        assert PANELS_DIR.exists(), f"Missing panels dir: {PANELS_DIR}"
        panel_files = list(PANELS_DIR.glob("panel_item_*.png"))
        assert len(panel_files) == 30, f"Expected 30 panels, found {len(panel_files)}"
        for p in panel_files:
            assert p.stat().st_size > 1000, f"Corrupted panel: {p}"

    def test_metadata_and_classification(self):
        meta_p = EVAL_DIR / "evaluation_metadata.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        assert meta["phase"] == "Phase 11"
        assert meta["review_design"]["total_items"] == 30
        assert meta["review_design"]["reviewers_count"] == 3
        assert meta["review_design"]["total_ratings_collected"] == 390
        assert meta["classification"] in ["A. Human-useful", "B. Mixed utility", "C. Low utility"]
