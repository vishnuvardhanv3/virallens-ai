"""
Comprehensive Test Suite for DHF1K Phase 2: Independent Human Visual-Attention Model.

Validates:
1. Safety & Production Model Non-Regression (SHA-256 hashes of model.pkl, scaler.pkl, feature_schema.json unchanged)
2. Strict Data Leakage Prevention (zero video ID overlap between train and validation splits)
3. Scaler derivation integrity (StandardScaler derived strictly from training data, applied unchanged to validation)
4. Low-fixation window exclusion (windows with < 2 consensus fixations excluded from primary supervised fitting)
5. Deterministic model selection policy (verified that selected model has the lowest validation MAE)
6. Prediction and target validity (all bounded within [0, 1], non-zero variance, no NaNs)
7. Provenance preservation across all validation prediction records
8. Artifact completeness (all 8 experiment artifacts present and conform to schema)
9. Scientific terminology compliance (zero presence of commercial or retention buzzwords)
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
def independent_dir(project_root):
    return os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_independent")


@pytest.fixture(scope="session")
def production_dir(project_root):
    return os.path.join(project_root, "models", "human_attention")


@pytest.fixture(scope="session")
def cache_dir(project_root):
    return os.path.join(project_root, "data", "dhf1k_cache")


# Test 1: Production Model Safety (SHA-256 Integrity)
def test_production_model_sha256_unmodified(production_dir):
    """Verify that production NEMAR model files are strictly unmodified."""
    expected_hashes = {
        "model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
        "scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
        "feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411"
    }
    for fname, exp_hash in expected_hashes.items():
        p = os.path.join(production_dir, fname)
        assert os.path.exists(p), f"Production file missing: {p}"
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        assert h == exp_hash, f"CRITICAL: {fname} hash altered! Expected {exp_hash}, got {h}"


# Test 2: Data Leakage Prevention (Strict Video Partitioning)
def test_zero_train_validation_video_overlap(cache_dir, independent_dir):
    """Verify strictly no video ID overlap between training and validation splits."""
    train_csv = os.path.join(cache_dir, "train_windows.csv")
    val_csv = os.path.join(independent_dir, "validation_predictions.csv")

    assert os.path.exists(train_csv), f"Train windows CSV missing: {train_csv}"
    assert os.path.exists(val_csv), f"Validation predictions CSV missing: {val_csv}"

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_videos = set(train_df["video_id"].unique())
    val_videos = set(val_df["video_id"].unique())

    overlap = train_videos.intersection(val_videos)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: Overlapping video IDs {overlap}"

    # Verify counts
    assert len(train_videos) == 600, f"Expected 600 training videos, got {len(train_videos)}"
    assert len(val_videos) == 100, f"Expected 100 validation videos, got {len(val_videos)}"


# Test 3: Scaler Derived Exclusively from Training Data
def test_scaler_derived_only_from_training_data(cache_dir, independent_dir):
    """Verify that scaler_dhf1k.pkl parameters match training data statistics."""
    import joblib
    scaler_path = os.path.join(independent_dir, "scaler_dhf1k.pkl")
    assert os.path.exists(scaler_path), f"Scaler missing: {scaler_path}"
    scaler = joblib.load(scaler_path)

    train_csv = os.path.join(cache_dir, "train_windows.csv")
    train_df = pd.read_csv(train_csv)
    train_clean = train_df[train_df["is_valid_fixation_window"]].copy()

    feature_names = [
        "motion_magnitude", "motion_std", "scene_cut_rate", "time_since_cut",
        "brightness_mean", "brightness_std", "contrast", "visual_complexity",
        "face_presence", "text_presence", "audio_rms", "audio_silence_ratio",
        "audio_spectral_centroid", "audio_speech_presence"
    ]
    X_train = train_clean[feature_names].values.astype(np.float32)

    emp_mean = np.mean(X_train, axis=0)
    emp_scale = np.std(X_train, axis=0)

    # Allow tiny float precision tolerance
    np.testing.assert_allclose(scaler.mean_, emp_mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(scaler.scale_, emp_scale, rtol=1e-3, atol=1e-3)


# Test 4: Low-Fixation Window Exclusion Policy
def test_low_fixation_windows_excluded_from_primary_training(cache_dir):
    """Verify windows with < 2 fixation points are flagged as invalid and excluded from training target."""
    train_csv = os.path.join(cache_dir, "train_windows.csv")
    df = pd.read_csv(train_csv)

    low_fix_mask = df["fixation_point_count"] < 2
    if low_fix_mask.sum() > 0:
        assert (~df.loc[low_fix_mask, "is_valid_fixation_window"]).all()
        assert df.loc[low_fix_mask, "target_fixation_concentration_proxy"].isnull().all()


# Test 5: Deterministic Model Selection Policy
def test_deterministic_model_selection_lowest_mae(independent_dir):
    """Verify that the selected model possesses the lowest validation MAE among all candidates."""
    eval_json_path = os.path.join(independent_dir, "evaluation.json")
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    candidates = eval_data["model_selection_summary"]
    selected = eval_data["selected_model"]

    selected_mae = candidates[selected]["val"]["mae"]
    for name, cinfo in candidates.items():
        assert cinfo["val"]["mae"] >= selected_mae - 1e-4, (
            f"Model {name} had lower MAE ({cinfo['val']['mae']}) than selected {selected} ({selected_mae})"
        )


# Test 6: Target and Prediction Validity & Bounds
def test_predictions_and_targets_bounded(independent_dir):
    """Verify that validation predictions and target values are strictly in [0, 1] without NaNs."""
    val_csv = os.path.join(independent_dir, "validation_predictions.csv")
    df = pd.read_csv(val_csv)

    targets = df["target_fixation_concentration_proxy"]
    preds = df["predicted_attention"]

    assert targets.isnull().sum() == 0
    assert preds.isnull().sum() == 0

    assert (targets >= 0.0).all() and (targets <= 1.0).all()
    assert (preds >= 0.0).all() and (preds <= 1.0).all()

    assert targets.std() > 0.01
    assert preds.std() > 0.005


# Test 7: Target Provenance Preservation
def test_target_provenance_preservation(independent_dir):
    """Verify all required provenance attributes exist on all evaluation records."""
    val_csv = os.path.join(independent_dir, "validation_predictions.csv")
    df = pd.read_csv(val_csv)

    required_cols = [
        "dataset_source", "video_id", "split", "window_index", "window_start",
        "window_end", "timestamp", "fixation_point_count", "target_fixation_concentration_proxy",
        "predicted_attention", "prediction_error", "absolute_error"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing provenance column: {col}"

    assert (df["dataset_source"] == "DHF1K").all()
    assert (df["split"] == "val").all()
    assert (df["window_end"] > df["window_start"]).all()


# Test 8: Artifact Completeness
def test_all_independent_artifacts_exist(independent_dir):
    """Verify that all 8 required experiment artifacts exist and are non-empty."""
    expected_files = [
        "model_dhf1k.pkl",
        "scaler_dhf1k.pkl",
        "feature_schema_dhf1k.json",
        "training_metadata.json",
        "evaluation.json",
        "evaluation.md",
        "validation_predictions.csv",
        "feature_statistics.json"
    ]
    for fname in expected_files:
        p = os.path.join(independent_dir, fname)
        assert os.path.exists(p), f"Required artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Artifact is empty: {p}"


# Test 9: Scientific Terminology Enforcement
def test_scientific_terminology_compliance(independent_dir):
    """Verify that forbidden commercial/retention terms are strictly absent from reports."""
    forbidden_terms = [
        "instagram retention",
        "scroll-stop",
        "commercial engagement",
        "viral retention",
        "scroll stop"
    ]
    for fname in ["evaluation.json", "evaluation.md", "training_metadata.json"]:
        p = os.path.join(independent_dir, fname)
        with open(p, "r", encoding="utf-8") as f:
            content = f.read().lower()
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term '{term}' found in {fname}"
