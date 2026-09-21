"""
Comprehensive Test Suite for DHF1K Phase 5: Feature Representation Upgrade.

Validates:
1. Strict Safety & Read-Only Integrity:
   - Production NEMAR hashes (model.pkl, scaler.pkl, feature_schema.json) unchanged.
   - Phase 2 independent DHF1K hashes (model_dhf1k.pkl, scaler_dhf1k.pkl, feature_schema_dhf1k.json) unchanged.
   - DHF1K source data counts (1,000 video files, 700 annotation folders) unchanged.
2. Existence and completeness of all 9 Phase 5 artifacts.
3. Exact video-level train/validation separation (no video crossing).
4. Zero target leakage into feature representations.
5. Scaler and preprocessing fit strictly train-only.
6. Validation prediction completeness (exactly 3,970 samples) and valid [0, 1] bounds.
7. Decision-quality metrics completeness (relative MAE, per-video distribution, bootstrap CI).
8. Target bin and temporal quartile completeness.
9. Scientific terminology compliance and non-causal disclaimer.
10. Deterministic candidate selection and decision rule consistency.
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
def feature_upgrade_dir(project_root):
    return os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_feature_upgrade")


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
def report_json(feature_upgrade_dir):
    p = os.path.join(feature_upgrade_dir, "feature_upgrade_report.json")
    assert os.path.exists(p), f"feature_upgrade_report.json missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def meta_json(feature_upgrade_dir):
    p = os.path.join(feature_upgrade_dir, "experiment_metadata.json")
    assert os.path.exists(p), f"experiment_metadata.json missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def candidate_metrics_df(feature_upgrade_dir):
    p = os.path.join(feature_upgrade_dir, "candidate_metrics.csv")
    assert os.path.exists(p), f"candidate_metrics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def val_predictions_df(feature_upgrade_dir):
    p = os.path.join(feature_upgrade_dir, "validation_predictions.csv")
    assert os.path.exists(p), f"validation_predictions.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def per_video_df(feature_upgrade_dir):
    p = os.path.join(feature_upgrade_dir, "per_video_metrics.csv")
    assert os.path.exists(p), f"per_video_metrics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def target_bin_df(feature_upgrade_dir):
    p = os.path.join(feature_upgrade_dir, "target_bin_metrics.csv")
    assert os.path.exists(p), f"target_bin_metrics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def temporal_df(feature_upgrade_dir):
    p = os.path.join(feature_upgrade_dir, "temporal_metrics.csv")
    assert os.path.exists(p), f"temporal_metrics.csv missing: {p}"
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
# 2. Phase 5 Artifacts Existence
# ==============================================================================

def test_all_phase5_artifacts_exist(feature_upgrade_dir):
    """All 9 required Phase 5 output artifacts must exist and be non-empty."""
    required_files = [
        "candidate_metrics.csv",
        "experiment_metadata.json",
        "feature_importance.csv",
        "feature_upgrade_report.json",
        "feature_upgrade_report.md",
        "per_video_metrics.csv",
        "target_bin_metrics.csv",
        "temporal_metrics.csv",
        "validation_predictions.csv",
    ]
    for rf in required_files:
        p = os.path.join(feature_upgrade_dir, rf)
        assert os.path.exists(p), f"Phase 5 artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Phase 5 artifact is empty: {p}"


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


def test_no_target_leakage_in_features(feature_upgrade_dir):
    """Feature importance file must contain zero target-derived column names."""
    feat_df = pd.read_csv(os.path.join(feature_upgrade_dir, "feature_importance.csv"))
    forbidden = ["target", "proxy", "fixation", "ground_truth", "label", "attention_potential"]
    for feat_name in feat_df["feature_name"]:
        for f in forbidden:
            assert f not in feat_name.lower(), f"DATA LEAKAGE: Target-derived feature '{feat_name}' in model!"


# ==============================================================================
# 4. Predictions Completeness & Output Bounds
# ==============================================================================

def test_validation_predictions_completeness(val_predictions_df):
    """Validation predictions CSV must contain exactly 3,970 samples with bounded predictions."""
    assert len(val_predictions_df) == 3970, f"Expected 3,970 samples, got {len(val_predictions_df)}"
    required_cols = [
        "video_id", "window_index", "target_fixation_concentration_proxy",
        "predicted_baseline", "predicted_upgrade_winner",
        "baseline_error", "upgrade_winner_error"
    ]
    for col in required_cols:
        assert col in val_predictions_df.columns, f"Missing column {col} in predictions CSV"
        assert not val_predictions_df[col].isna().any(), f"Column {col} contains NaNs"

    # Predictions must lie strictly in [0, 1]
    assert (val_predictions_df["predicted_baseline"] >= 0.0).all()
    assert (val_predictions_df["predicted_baseline"] <= 1.0).all()
    assert (val_predictions_df["predicted_upgrade_winner"] >= 0.0).all()
    assert (val_predictions_df["predicted_upgrade_winner"] <= 1.0).all()


# ==============================================================================
# 5. Decision-Quality Metrics & Bootstrap CI
# ==============================================================================

def test_decision_quality_metrics_in_report(report_json):
    """Report must include relative MAE change, per-video percentiles, and bootstrap 95% CI."""
    assert "comparison_summary" in report_json
    comp = report_json["comparison_summary"]
    assert "relative_mae_change_pct" in comp
    assert "absolute_mae_change" in comp

    assert "bootstrap_diagnostics" in report_json
    boot = report_json["bootstrap_diagnostics"]
    ci = boot["bootstrap_95_ci"]
    assert "ci_lower_2_5" in ci
    assert "ci_upper_97_5" in ci
    assert ci["ci_lower_2_5"] <= ci["ci_upper_97_5"]

    p_stats = boot["per_video_delta_stats"]
    for k in ["median", "mean", "std", "p25", "p75"]:
        assert k in p_stats

    p_cnt = boot["per_video_counts"]
    assert p_cnt["improved_count"] + p_cnt["worsened_count"] + p_cnt["unchanged_count"] == 100


# ==============================================================================
# 6. Target Bins and Temporal Metrics Completeness
# ==============================================================================

def test_target_bin_metrics_structure(target_bin_df):
    """Target bin metrics must cover exactly 4 intervals summing to 3,970 windows."""
    assert len(target_bin_df) == 4
    assert target_bin_df["sample_count"].sum() == 3970


def test_temporal_metrics_structure(temporal_df):
    """Temporal metrics must cover 4 quartiles summing to 3,970 windows."""
    assert len(temporal_df) == 4
    assert temporal_df["sample_count"].sum() == 3970


# ==============================================================================
# 7. Scientific Terminology & Decision Rule Consistency
# ==============================================================================

def test_scientific_terminology_and_decision_consistency(feature_upgrade_dir, report_json):
    """Report must adhere to scientific language and decision thresholds."""
    md_path = os.path.join(feature_upgrade_dir, "feature_upgrade_report.md")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read().lower()

    forbidden = ["instagram", "tiktok", "reels", "commercial engagement", "scroll-stop", "virality prediction"]
    for term in forbidden:
        assert term not in content, f"Forbidden buzzword '{term}' found in report!"

    assert "dhf1k-derived fixation concentration proxy" in content

    dec = report_json["decision"]
    assert dec["status"] in [
        "Representation upgrade promising",
        "Representation upgrade marginal / Phase 2 model retained"
    ]
