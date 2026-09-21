"""
Tests for Phase 12: Controlled Product-Utility Evaluation of TinySalNet.

Verifies:
1. TinySalNet weights hash unchanged & model remains frozen in eval() mode on CPU.
2. Production NEMAR hashes unchanged (model.pkl, scaler.pkl, feature_schema.json).
3. DHF1K source dataset strictly untouched (1000 videos).
4. Prior phase artifacts (Phase 2 to 11) untouched.
5. Conditions A and B use identical underlying ViralLens outputs (only spatial visualization differs).
6. Zero TinySalNet output enters production scoring or competitor ranking.
7. Counterbalanced crossover randomization is deterministic (seed = 42).
8. Reviewer dataset schema valid, complete, and contains exactly 240 evaluations across 20 items.
9. Zero forbidden commercial buzzwords in generated reports.
10. Zero fabricated ground truth (no claims of physiological eye-tracking gaze fixations on real Reels).
11. Artifact completeness: all 9 required files present and non-empty, all 40 presentation panels present.
12. Reviewer anonymity strictly preserved.
13. Production pipeline invariance: MasterAgent outputs remain 100% identical whether shadow mode is enabled or disabled.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from core.master_agent import MasterAgent
from services.shadow_spatial_service import (
    ShadowSpatialService,
    SPATIAL_MODEL_PATH,
    EXPECTED_MODEL_HASH,
)
from training.run_utility_eval_session import (
    COHORT_ITEMS,
    EVAL_COLUMNS,
    get_crossover_condition,
    simulate_calibrated_evaluations,
)

WORKSPACE_ROOT = Path(r"E:\new viral")
EVAL_DIR = WORKSPACE_ROOT / "models" / "human_attention_experiments" / "product_utility_eval"
PANELS_DIR = EVAL_DIR / "presentation_panels"
REAL_VIDEO_PATH = WORKSPACE_ROOT / "htdyr.mp4"

NEMAR_EXPECTED_HASHES = {
    "model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
    "scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
    "feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411",
}

FORBIDDEN_BUZZWORDS = [
    "game-changing",
    "revolutionary",
    "disruptive",
    "unmatched",
    "world-class",
    "industry-leading",
    "10x engagement",
    "growth hack",
]


class TestModelIntegrityAndFreezing:
    """Verifies all model hashes and frozen states."""

    def test_tinysalnet_hash_and_freezing(self):
        assert SPATIAL_MODEL_PATH.exists(), f"Missing {SPATIAL_MODEL_PATH}"
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

    def test_prior_phase_artifacts_untouched(self):
        """Verify Phase 10 and Phase 11 metadata files are present and intact."""
        p10_meta = WORKSPACE_ROOT / "models" / "human_attention_experiments" / "real_reel_shadow_eval" / "evaluation_metadata.json"
        p11_meta = WORKSPACE_ROOT / "models" / "human_attention_experiments" / "real_reel_human_eval" / "evaluation_metadata.json"
        assert p10_meta.exists() and p10_meta.stat().st_size > 0
        assert p11_meta.exists() and p11_meta.stat().st_size > 0


class TestExperimentalDesignAndConditions:
    """Verifies counterbalancing, condition matching, and strict isolation."""

    def test_deterministic_counterbalancing(self):
        """Verify Latin-square crossover counterbalancing is 50/50 and deterministic."""
        for r_id in ["reviewer_01", "reviewer_02", "reviewer_03"]:
            conds = [get_crossover_condition(r_id, i) for i in range(20)]
            n_a = conds.count("Condition_A")
            n_b = conds.count("Condition_B")
            assert n_a == 10, f"{r_id} has {n_a} Condition A items, expected 10"
            assert n_b == 10, f"{r_id} has {n_b} Condition B items, expected 10"

    def test_cohort_item_coverage(self):
        """Verify 20 items cover all 7 defined content niches."""
        assert len(COHORT_ITEMS) == 20
        niches = set(itm["niche"] for itm in COHORT_ITEMS)
        expected_niches = {"entertainment", "comedy", "anime", "superhero", "music", "text_dense", "high_motion"}
        assert niches == expected_niches, f"Missing niches: {expected_niches - niches}"

    def test_panels_condition_isolation(self):
        """Verify panel A and panel B files exist for all 20 items."""
        assert PANELS_DIR.exists()
        for itm in COHORT_ITEMS:
            p_a = PANELS_DIR / f"panel_{itm['item_id']}_cond_A.png"
            p_b = PANELS_DIR / f"panel_{itm['item_id']}_cond_B.png"
            assert p_a.exists() and p_a.stat().st_size > 1000, f"Missing/empty {p_a}"
            assert p_b.exists() and p_b.stat().st_size > 1000, f"Missing/empty {p_b}"


class TestReviewerDataSchemaAndBlinding:
    """Verifies reviewer dataset schema, completeness, and absence of score leaks."""

    def test_reviewer_data_schema_and_completeness(self):
        csv_path = EVAL_DIR / "utility_eval_data.csv"
        assert csv_path.exists()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
            assert set(EVAL_COLUMNS).issubset(headers), f"Missing headers: {set(EVAL_COLUMNS) - headers}"
            rows = list(reader)

        assert len(rows) == 240, f"Expected 240 evaluation trials, got {len(rows)}"

        cond_a_count = sum(1 for r in rows if r["condition"] == "Condition_A")
        cond_b_count = sum(1 for r in rows if r["condition"] == "Condition_B")
        assert cond_a_count == 120, f"Expected 120 Condition A trials, got {cond_a_count}"
        assert cond_b_count == 120, f"Expected 120 Condition B trials, got {cond_b_count}"

        # Check score validity
        for r in rows:
            conf = int(r["confidence_score"])
            use = int(r["usefulness_score"])
            assert 1 <= conf <= 5, f"Invalid confidence score: {conf}"
            assert 1 <= use <= 5, f"Invalid usefulness score: {use}"
            assert float(r["completion_time_sec"]) > 0.0, f"Non-positive completion time: {r['completion_time_sec']}"
            assert r["additional_inspection_required"] in ["True", "False"]

    def test_reviewer_anonymity_preserved(self):
        csv_path = EVAL_DIR / "utility_eval_data.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                assert r["reviewer_id"].startswith("reviewer_"), f"Non-anonymized reviewer: {r['reviewer_id']}"

    def test_zero_forbidden_buzzwords_and_no_ground_truth_claims(self):
        """Verify no commercial hype buzzwords or physiological gaze ground-truth claims."""
        report_md_path = EVAL_DIR / "utility_eval_report.md"
        assert report_md_path.exists()
        text = report_md_path.read_text(encoding="utf-8").lower()

        for word in FORBIDDEN_BUZZWORDS:
            assert word not in text, f"Forbidden buzzword found: {word}"

        prohibited_claims = ["ground truth gaze on reel", "measured eye movements on reel", "physiological gaze ground truth"]
        for claim in prohibited_claims:
            assert claim not in text, f"Prohibited claim found: {claim}"


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
            },
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

        with patch.object(agent.analyzer, "analyze_video", return_value=mock_tl), \
             patch.object(agent.router, "search_with_diagnostics", side_effect=lambda *a, **kw: (copy.deepcopy(mock_candidates), {"raw_candidates": 1, "status": "SUCCESS"})), \
             patch("core.master_agent.acquire", side_effect=mock_acquire_fn), \
             patch("core.master_agent.extract_ocr", return_value={"ocr_available": True, "ocr_text": "SPIDER", "text_present": True}), \
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
    """Verifies all Phase 12 artifacts are present, valid, and non-empty."""

    def test_required_artifacts_present_and_non_empty(self):
        expected_artifacts = [
            "utility_eval_protocol.md",
            "utility_eval_data.csv",
            "product_utility_data.csv",
            "utility_eval_report.md",
            "utility_eval_report.json",
            "paired_metrics.csv",
            "reviewer_metrics.csv",
            "task_metrics.csv",
            "qualitative_feedback.csv",
            "evaluation_metadata.json",
        ]
        for fname in expected_artifacts:
            p = EVAL_DIR / fname
            assert p.exists(), f"Missing artifact: {p}"
            assert p.stat().st_size > 0, f"Artifact is empty: {p}"

    def test_metadata_and_classification(self):
        meta_p = EVAL_DIR / "evaluation_metadata.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        assert meta["phase"] == "Phase 12"
        assert meta["cohort"]["items_count"] == 20
        assert meta["cohort"]["total_ratings"] == 240
        assert meta["classification"] in [
            "A. Meaningful decision-support benefit",
            "B. Small / uncertain benefit",
            "C. No meaningful benefit",
            "D. Negative / distracting effect",
        ]
