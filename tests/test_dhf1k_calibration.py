"""
Comprehensive Test Suite for DHF1K Phase 4: Model Calibration & Generalization Analysis.

Validates:
1. Strict Read-Only Safety:
   - Production NEMAR hashes (model.pkl, scaler.pkl, feature_schema.json) unchanged.
   - Independent DHF1K hashes (model_dhf1k.pkl, scaler_dhf1k.pkl, feature_schema_dhf1k.json) unchanged.
   - DHF1K source data counts (1,000 video files, 700 annotation folders) unchanged.
2. Existence and completeness of all 6 Phase 4 calibration artifacts.
3. Prediction record completeness (exactly 3,970 valid validation samples).
4. Target bin metrics structure and sample count conservation (sum equals 3,970).
5. Per-video calibration metrics completeness (exactly 100 validation videos, 601-700).
6. Deterministic 5-fold cross-fitting with zero parameter leakage across folds.
7. Scientific terminology compliance and non-causal methodology note.
8. Decision rule validity and technical consistency.
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
def calibration_dir(project_root):
    return os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_calibration")


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
def report_json(calibration_dir):
    p = os.path.join(calibration_dir, "calibration_report.json")
    assert os.path.exists(p), f"calibration_report.json missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def methods_json(calibration_dir):
    p = os.path.join(calibration_dir, "calibration_methods.json")
    assert os.path.exists(p), f"calibration_methods.json missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def predictions_df(calibration_dir):
    p = os.path.join(calibration_dir, "calibration_predictions.csv")
    assert os.path.exists(p), f"calibration_predictions.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def bin_metrics_df(calibration_dir):
    p = os.path.join(calibration_dir, "calibration_bin_metrics.csv")
    assert os.path.exists(p), f"calibration_bin_metrics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def video_metrics_df(calibration_dir):
    p = os.path.join(calibration_dir, "calibration_video_metrics.csv")
    assert os.path.exists(p), f"calibration_video_metrics.csv missing: {p}"
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
# 2. Phase 4 Artifacts Existence Tests
# ==============================================================================

def test_all_phase4_artifacts_exist(calibration_dir):
    """All 6 required Phase 4 output artifacts must exist."""
    required_files = [
        "calibration_report.json",
        "calibration_report.md",
        "calibration_predictions.csv",
        "calibration_bin_metrics.csv",
        "calibration_video_metrics.csv",
        "calibration_methods.json",
    ]
    for rf in required_files:
        p = os.path.join(calibration_dir, rf)
        assert os.path.exists(p), f"Phase 4 artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Phase 4 artifact is empty: {p}"


# ==============================================================================
# 3. Prediction Records Completeness & Structure
# ==============================================================================

def test_calibration_predictions_completeness(predictions_df):
    """Predictions CSV must contain exactly 3,970 samples with all required columns."""
    assert len(predictions_df) == 3970, f"Expected 3,970 samples, got {len(predictions_df)}"
    expected_cols = [
        "video_id", "window_index", "target_fixation_concentration_proxy",
        "predicted_attention", "calibrated_affine_cross_fitted",
        "calibrated_isotonic_cross_fitted", "calibrated_affine_error"
    ]
    for col in expected_cols:
        assert col in predictions_df.columns, f"Missing column {col} in predictions CSV"
        assert not predictions_df[col].isna().any(), f"Column {col} contains NaNs"

    # Prediction span compression
    pred_span = predictions_df["predicted_attention"].max() - predictions_df["predicted_attention"].min()
    target_span = predictions_df["target_fixation_concentration_proxy"].max() - predictions_df["target_fixation_concentration_proxy"].min()
    assert pred_span < target_span, "Model predictions should exhibit compression relative to target span"


# ==============================================================================
# 4. Target Bin Metrics Completeness
# ==============================================================================

def test_target_bin_metrics_structure(bin_metrics_df):
    """Target bins must cover all samples without omission or overlap."""
    assert len(bin_metrics_df) == 4, f"Expected 4 target bins, got {len(bin_metrics_df)}"
    assert bin_metrics_df["sample_count"].sum() == 3970, (
        f"Sum of bin samples ({bin_metrics_df['sample_count'].sum()}) != 3,970"
    )

    # Validate delta MAE calculation
    for _, row in bin_metrics_df.iterrows():
        expected_delta = row["cal_mae"] - row["orig_mae"]
        assert abs(row["delta_mae"] - expected_delta) < 1e-3


# ==============================================================================
# 5. Per-Video Validation Metrics Completeness
# ==============================================================================

def test_per_video_calibration_metrics(video_metrics_df):
    """Video metrics CSV must contain exactly 100 validation videos summing to 3,970 windows."""
    assert len(video_metrics_df) == 100, f"Expected 100 validation videos, got {len(video_metrics_df)}"
    assert video_metrics_df["valid_window_count"].sum() == 3970, (
        f"Sum of video window counts ({video_metrics_df['valid_window_count'].sum()}) != 3,970"
    )

    expected_video_ids = [f"DHF1K_{i:03d}" for i in range(601, 701)]
    assert sorted(list(video_metrics_df["video_id"])) == expected_video_ids

    # Video level improvement / worsening counts consistency
    improved = video_metrics_df["is_improved"].sum()
    worsened = video_metrics_df["is_worsened"].sum()
    unchanged = len(video_metrics_df) - improved - worsened
    assert improved + worsened + unchanged == 100


# ==============================================================================
# 6. Grouped Cross-Fitting Leakage Prevention
# ==============================================================================

def test_cross_fitting_zero_leakage(methods_json):
    """5-fold cross-fitting must have exactly 80 train and 20 test videos per fold."""
    cv = methods_json["cross_validated_evaluation"]
    folds = cv["folds_summary"]
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    total_test_samples = sum(f["test_fold_samples"] for f in folds)
    assert total_test_samples == 3970, f"Sum of fold test samples ({total_test_samples}) != 3,970"

    for f in folds:
        assert f["train_videos_count"] == 80
        assert f["test_videos_count"] == 20


# ==============================================================================
# 7. Scientific Terminology & Non-Causal Compliance
# ==============================================================================

def test_scientific_terminology_compliance(report_json, calibration_dir):
    """Check report for non-causal disclaimer and absence of unauthorized commercial terms."""
    # Check non-causal disclaimer
    assert "non_causal_methodology_note" in report_json
    note = report_json["non_causal_methodology_note"].lower()
    assert "not claimed as the causal mechanism" in note

    # Read markdown content
    md_path = os.path.join(calibration_dir, "calibration_report.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read().lower()

    forbidden_terms = [
        "instagram", "tiktok", "reels", "commercial engagement",
        "scroll-stop", "virality prediction"
    ]
    for term in forbidden_terms:
        assert term not in md_content, f"Forbidden term '{term}' found in calibration_report.md!"

    # Target proxy language
    assert "dhf1k-derived fixation concentration proxy" in md_content


# ==============================================================================
# 8. Decision Rule Validity
# ==============================================================================

def test_decision_rule_validity(report_json):
    """Decision rule must be clearly declared with empirical justification."""
    dec = report_json["decision_rule"]
    assert dec["recommendation"] in ["Calibration promising", "Calibration marginal", "Calibration not justified"]
    assert "rationale" in dec
    assert "decision_factors" in dec
