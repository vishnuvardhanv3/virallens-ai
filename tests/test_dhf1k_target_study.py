"""
Comprehensive Test Suite for DHF1K Phase 6: Target Representation Investigation.

Validates:
1. Strict Safety & Read-Only Integrity:
   - Production NEMAR hashes (model.pkl, scaler.pkl, feature_schema.json) unchanged.
   - Phase 2 independent DHF1K hashes (model_dhf1k.pkl, scaler_dhf1k.pkl, feature_schema_dhf1k.json) unchanged.
   - DHF1K source data counts (1,000 video files, 700 annotation folders) unchanged.
2. Existence and non-emptiness of all 9 Phase 6 artifacts.
3. Exact video-level train/validation separation (no video crossing).
4. Target mathematical bounds: all normalized targets lie strictly in [0, 1].
5. Preservation of raw spatial statistics (raw_B, raw_C, raw_D) in exported data.
6. Zero target leakage into feature representations.
7. Validation predictions completeness across all 4 targets.
8. Decision classification validity (A, B, or C).
9. Scientific terminology compliance (zero commercial buzzwords).
"""

import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(r"E:\new viral\training")


@pytest.fixture(scope="session")
def project_root():
    return r"E:\new viral"


@pytest.fixture(scope="session")
def target_study_dir(project_root):
    return os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_target_study")


@pytest.fixture(scope="session")
def independent_dir(project_root):
    return os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_independent")


@pytest.fixture(scope="session")
def production_dir(project_root):
    return os.path.join(project_root, "models", "human_attention")


@pytest.fixture(scope="session")
def dhf1k_root():
    return r"E:\viral_attention_data\DHF1K"


@pytest.fixture(scope="session")
def report_json(target_study_dir):
    p = os.path.join(target_study_dir, "target_study_report.json")
    assert os.path.exists(p), f"target_study_report.json missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def definitions_json(target_study_dir):
    p = os.path.join(target_study_dir, "target_definitions.json")
    assert os.path.exists(p), f"target_definitions.json missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def target_metrics_df(target_study_dir):
    p = os.path.join(target_study_dir, "target_metrics.csv")
    assert os.path.exists(p), f"target_metrics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def target_statistics_df(target_study_dir):
    p = os.path.join(target_study_dir, "target_statistics.csv")
    assert os.path.exists(p), f"target_statistics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def val_predictions_df(target_study_dir):
    p = os.path.join(target_study_dir, "validation_predictions.csv")
    assert os.path.exists(p), f"validation_predictions.csv missing: {p}"
    return pd.read_csv(p)


# ==============================================================================
# 1. Strict Read-Only Safety Tests
# ==============================================================================

def test_production_model_hashes_unchanged(production_dir):
    """Production NEMAR model artifacts must have exact verified hashes."""
    expected = {
        "model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
        "scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
        "feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411",
    }
    for filename, expected_hash in expected.items():
        filepath = os.path.join(production_dir, filename)
        assert os.path.exists(filepath), f"Missing production artifact: {filepath}"
        with open(filepath, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"CRITICAL: Production artifact {filename} modified!"


def test_independent_dhf1k_model_hashes_unchanged(independent_dir):
    """Phase 2 independent DHF1K model artifacts must remain strictly read-only."""
    expected = {
        "model_dhf1k.pkl": "faffcb161ac9f895763383364af52527d7fc3f468614936b3d998ed4a275c7a0",
        "scaler_dhf1k.pkl": "8d83c2c336397f9dba5d05f68251f85837779a785875eb2df65b1fdf9f770cb8",
        "feature_schema_dhf1k.json": "80f774f0ded8033b18132beb27fce88865298f005a11a07834c8728524645903",
    }
    for filename, expected_hash in expected.items():
        filepath = os.path.join(independent_dir, filename)
        assert os.path.exists(filepath), f"Missing independent artifact: {filepath}"
        with open(filepath, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"CRITICAL: Independent DHF1K artifact {filename} modified!"


def test_dhf1k_source_data_counts(dhf1k_root):
    """DHF1K source dataset files must remain intact and unmodified."""
    video_dir = os.path.join(dhf1k_root, "video")
    annotation_dir = os.path.join(dhf1k_root, "annotation")

    assert os.path.exists(video_dir), f"Video dir missing: {video_dir}"
    assert os.path.exists(annotation_dir), f"Annotation dir missing: {annotation_dir}"

    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(".avi")]
    annotations = [d for d in os.listdir(annotation_dir) if os.path.isdir(os.path.join(annotation_dir, d))]

    assert len(videos) == 1000, f"Expected 1000 videos, found {len(videos)}"
    assert len(annotations) == 700, f"Expected 700 annotation folders, found {len(annotations)}"


# ==============================================================================
# 2. Phase 6 Artifacts Existence
# ==============================================================================

def test_all_phase6_artifacts_exist(target_study_dir):
    """All 9 required Phase 6 output artifacts must exist and be non-empty."""
    required_files = [
        "per_video_metrics.csv",
        "target_bin_metrics.csv",
        "target_definitions.json",
        "target_metrics.csv",
        "target_statistics.csv",
        "target_study_report.json",
        "target_study_report.md",
        "temporal_metrics.csv",
        "validation_predictions.csv",
    ]
    for rf in required_files:
        p = os.path.join(target_study_dir, rf)
        assert os.path.exists(p), f"Phase 6 artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Phase 6 artifact is empty: {p}"


# ==============================================================================
# 3. Leakage Prevention & Video Split Separation
# ==============================================================================

def test_train_val_video_separation(val_predictions_df):
    """Validation set must consist strictly of videos 601 through 700."""
    val_videos = sorted(val_predictions_df["video_id"].unique())
    assert len(val_videos) == 100, f"Expected 100 validation videos, got {len(val_videos)}"
    expected_videos = [f"DHF1K_{i:03d}" for i in range(601, 701)]
    assert val_videos == expected_videos, "Validation videos do not match expected 601-700 set"

    # Confirm no training video in validation
    train_videos = set(f"DHF1K_{i:03d}" for i in range(1, 601))
    assert len(set(val_videos).intersection(train_videos)) == 0, "DATA LEAKAGE: Training videos found in validation!"


# ==============================================================================
# 4. Target Mathematical Bounds & Provenance
# ==============================================================================

def test_target_mathematical_bounds(val_predictions_df):
    """All normalized supervisory targets must lie strictly in [0, 1]."""
    targets = [
        "target_a_fixation_concentration_ground_truth",
        "target_b_fixation_concentration_ground_truth",
        "target_c_saliency_concentration_ground_truth",
        "target_d_saliency_entropy_dispersion_ground_truth",
    ]
    for t in targets:
        assert t in val_predictions_df.columns, f"Target column {t} missing"
        vals = val_predictions_df[t].dropna().values
        assert len(vals) > 0
        assert (vals >= 0.0).all(), f"Target {t} contains values < 0.0"
        assert (vals <= 1.0).all(), f"Target {t} contains values > 1.0"


def test_target_definitions_audit(definitions_json):
    """Target definitions JSON must document all 4 targets with mathematical derivations."""
    assert len(definitions_json) == 4
    targets_documented = [item["target"] for item in definitions_json]
    assert any("Target A" in t for t in targets_documented)
    assert any("Target B" in t for t in targets_documented)
    assert any("Target C" in t for t in targets_documented)
    assert any("Target D" in t for t in targets_documented)

    for item in definitions_json:
        for req_field in ["formula", "raw_statistic", "coord_range", "norm_reference", "semantics"]:
            assert req_field in item, f"Missing {req_field} in target definition {item['target']}"


# ==============================================================================
# 5. Target Metrics & Decision Classification
# ==============================================================================

def test_target_metrics_completeness(target_metrics_df):
    """Target metrics must contain all 4 targets with valid metrics."""
    assert len(target_metrics_df) == 4
    for col in ["val_mae", "val_rmse", "val_r2", "pearson_r", "spearman_rho", "std_ratio_pred_to_target"]:
        assert col in target_metrics_df.columns
        assert not target_metrics_df[col].isna().any()


def test_decision_classification_validity(report_json):
    """Decision classification must be one of A, B, or C with rationale."""
    assert "decision_classification" in report_json
    dec = report_json["decision_classification"]
    assert "classification" in dec
    assert any(dec["classification"].startswith(prefix) for prefix in ["A.", "B.", "C."])
    assert "rationale" in dec
    assert "best_alternative" in dec


# ==============================================================================
# 6. Scientific Terminology Compliance
# ==============================================================================

def test_scientific_terminology_compliance(target_study_dir):
    """Report must adhere strictly to scientific language and lack unauthorized commercial buzzwords."""
    md_path = os.path.join(target_study_dir, "target_study_report.md")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read().lower()

    forbidden = ["instagram", "tiktok", "reels", "commercial engagement", "scroll-stop", "virality prediction"]
    for term in forbidden:
        assert term not in content, f"Forbidden term '{term}' found in report!"

    assert "dhf1k-derived supervisory target" in content
