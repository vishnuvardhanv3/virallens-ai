"""
Tests for Phase 8 — DHF1K Spatial Saliency Dense Validation.

Verifies:
1. Phase 7 model hash unchanged (frozen spatial_model.pt).
2. Production model integrity (SHA-256 unchanged).
3. Phase 2 model integrity (SHA-256 unchanged).
4. DHF1K source data unchanged (1000 videos, 700 annotations).
5. Validation IDs strictly 601-700 only (no training videos 001-600 evaluated).
6. Deterministic 1 fps frame sampling (timestamps valid, no duplicate frame samples).
7. Valid metric bounds and NaN handling for uniform baseline NSS/CC.
8. Artifact completeness (all 9 required Phase 8 artifacts exist and non-empty).
9. Scientific terminology compliance (no 'ground-truth attention').
10. Final decision classification provenance.
"""

import os
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import torch

EXPECTED_HASHES = {
    r"models\human_attention\model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
    r"models\human_attention\scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
    r"models\human_attention\feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411",
    r"models\human_attention_experiments\dhf1k_independent\model_dhf1k.pkl": "faffcb161ac9f895763383364af52527d7fc3f468614936b3d998ed4a275c7a0",
    r"models\human_attention_experiments\dhf1k_independent\scaler_dhf1k.pkl": "8d83c2c336397f9dba5d05f68251f85837779a785875eb2df65b1fdf9f770cb8",
    r"models\human_attention_experiments\dhf1k_independent\feature_schema_dhf1k.json": "80f774f0ded8033b18132beb27fce88865298f005a11a07834c8728524645903",
    r"models\human_attention_experiments\dhf1k_spatial\spatial_model.pt": "fd6f27fd5ef6334c597ee3981086fc70f639cd59c7d57399fa72253d883b95c9",
}

PROJECT_ROOT = r"E:\new viral"
DHF1K_ROOT = r"E:\viral_attention_data\DHF1K"
ROBUSTNESS_DIR = os.path.join(PROJECT_ROOT, "models", "human_attention_experiments", "dhf1k_spatial_robustness")


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Model & Data Integrity Tests
# ---------------------------------------------------------------------------
def test_all_model_hashes_unchanged():
    """Verify production, Phase 2, and frozen Phase 7 model hashes remain bit-for-bit identical."""
    for rel_path, expected_hash in EXPECTED_HASHES.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        assert os.path.exists(full_path), f"Missing model file: {full_path}"
        actual_hash = sha256_file(full_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}!"


def test_dhf1k_source_data_unchanged():
    """Verify DHF1K source counts (1000 videos, 700 annotation folders)."""
    vid_dir = os.path.join(DHF1K_ROOT, "video")
    ann_dir = os.path.join(DHF1K_ROOT, "annotation")
    
    videos = [f for f in os.listdir(vid_dir) if f.endswith(".AVI")]
    annotations = [d for d in os.listdir(ann_dir) if os.path.isdir(os.path.join(ann_dir, d))]
    
    assert len(videos) == 1000, f"Expected 1000 videos, found {len(videos)}"
    assert len(annotations) == 700, f"Expected 700 annotations, found {len(annotations)}"


# ---------------------------------------------------------------------------
# Scope & Sampling Tests
# ---------------------------------------------------------------------------
def test_validation_video_scope_and_sampling():
    """Verify only validation videos 601-700 are evaluated, with deterministic non-duplicated sampling."""
    dense_csv = os.path.join(ROBUSTNESS_DIR, "dense_validation_metrics.csv")
    assert os.path.exists(dense_csv), "Missing dense_validation_metrics.csv"
    
    df = pd.read_csv(dense_csv)
    video_ids = sorted(df["video_id"].unique())
    assert min(video_ids) == 601, f"Expected min video 601, got {min(video_ids)}"
    assert max(video_ids) == 700, f"Expected max video 700, got {max(video_ids)}"
    assert len(video_ids) == 100, f"Expected exactly 100 videos, got {len(video_ids)}"
    
    # Check no duplicate (video_id, frame_idx) pairs
    dups = df.duplicated(subset=["video_id", "frame_idx"]).sum()
    assert dups == 0, f"Found {dups} duplicate frame samples!"
    
    # Check timestamps and non-negative frame indices
    assert (df["frame_idx"] >= 0).all()
    assert (df["timestamp_sec"] >= 0.0).all()


# ---------------------------------------------------------------------------
# Baseline Validity Tests
# ---------------------------------------------------------------------------
def test_uniform_baseline_validity():
    """Verify uniform baseline NSS and CC are NaN (undefined), not arbitrarily substituted with 0."""
    dense_csv = os.path.join(ROBUSTNESS_DIR, "dense_validation_metrics.csv")
    df = pd.read_csv(dense_csv)
    
    # baseline_nss and baseline_cc must be NaN
    assert df["baseline_nss"].isna().all(), "baseline_nss must be NaN because variance is zero!"
    assert df["baseline_cc"].isna().all(), "baseline_cc must be NaN because variance is zero!"
    
    # baseline_auc_judd must be valid in [0, 1] for non-empty fixation frames and average around 0.50
    valid_b_auc = df["baseline_auc_judd"].dropna()
    assert (valid_b_auc >= 0.0).all() and (valid_b_auc <= 1.0).all()
    mean_b_auc = valid_b_auc.mean()
    assert 0.45 <= mean_b_auc <= 0.58, f"Expected mean baseline AUC near 0.50, got {mean_b_auc}"
    # baseline_sim must be valid in [0, 1]
    assert (df["baseline_sim"] >= 0.0).all() and (df["baseline_sim"] <= 1.0).all()


# ---------------------------------------------------------------------------
# Metric Bounds Tests
# ---------------------------------------------------------------------------
def test_model_metric_bounds():
    """Verify model metrics fall within their theoretical mathematical domains."""
    dense_csv = os.path.join(ROBUSTNESS_DIR, "dense_validation_metrics.csv")
    df = pd.read_csv(dense_csv)
    
    # AUC-Judd in [0, 1]
    valid_auc = df["model_auc_judd"].dropna()
    assert (valid_auc >= 0.0).all() and (valid_auc <= 1.0).all()
    
    # SIM in [0, 1]
    assert (df["model_sim"] >= 0.0).all() and (df["model_sim"] <= 1.0).all()
    
    # CC in [-1, 1]
    assert (df["model_cc"] >= -1.0).all() and (df["model_cc"] <= 1.0).all()
    
    # KL >= 0
    assert (df["model_kl"] >= 0.0).all()


# ---------------------------------------------------------------------------
# Artifact Completeness Tests
# ---------------------------------------------------------------------------
def test_all_artifacts_exist_and_nonempty():
    """Verify all 9 Phase 8 required artifacts exist and are non-empty."""
    required_files = [
        "spatial_robustness_report.json",
        "spatial_robustness_report.md",
        "dense_validation_metrics.csv",
        "per_video_metrics.csv",
        "temporal_metrics.csv",
        "resolution_metrics.csv",
        "error_analysis.csv",
        "evaluation_metadata.json",
    ]
    for fname in required_files:
        path = os.path.join(ROBUSTNESS_DIR, fname)
        assert os.path.exists(path), f"Missing required artifact: {fname}"
        assert os.path.getsize(path) > 0, f"Artifact is empty: {fname}"
        
    qual_dir = os.path.join(ROBUSTNESS_DIR, "qualitative_examples")
    assert os.path.exists(qual_dir), "Missing qualitative_examples directory!"
    pngs = [f for f in os.listdir(qual_dir) if f.endswith(".png")]
    assert len(pngs) >= 5, f"Expected at least 5 qualitative PNGs, found {len(pngs)}"


def test_scientific_terminology():
    """Verify reports avoid forbidden phrase 'ground-truth attention'."""
    report_md = os.path.join(ROBUSTNESS_DIR, "spatial_robustness_report.md")
    report_json = os.path.join(ROBUSTNESS_DIR, "spatial_robustness_report.json")
    
    with open(report_md, "r", encoding="utf-8") as f:
        md_text = f.read()
    with open(report_json, "r", encoding="utf-8") as f:
        json_text = f.read()
        
    assert "ground-truth attention" not in md_text.lower(), "Found forbidden term 'ground-truth attention' in markdown report!"
    assert "ground-truth attention" not in json_text.lower(), "Found forbidden term 'ground-truth attention' in json report!"


def test_final_classification():
    """Verify final decision is properly recorded in spatial_robustness_report.json."""
    report_json = os.path.join(ROBUSTNESS_DIR, "spatial_robustness_report.json")
    with open(report_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["final_classification"].startswith("A. Robust under dense validation")
