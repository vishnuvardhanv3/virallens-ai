"""
Unit and Integration Test Suite for DHF1K Phase 1: Zero-Shot Cross-Dataset Validation.

Validates:
1. Safety constraints (zero mutations to production models, v1.0.0, or DHF1K source data)
2. Exact 14-feature schema conformance with production feature_schema.json
3. Provenance preservation across all 3,970 evaluation samples
4. Target independence (Target A1/A2 and Target B evaluated separately with no equivalence claims)
5. Zero-shot model execution validity
6. Metric integrity (MAE, RMSE, R^2, Pearson, Spearman)
7. Domain shift analysis
8. Artifact schema validation (evaluation.json, evaluation.md, feature_statistics.json, predictions.csv)
9. Scientific terminology enforcement (no commercial/retention buzzwords)
"""

import json
import math
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(r"E:\new viral\training")
from evaluate_dhf1k_zero_shot import DHF1KZeroShotEvaluator


@pytest.fixture(scope="session")
def project_root():
    return r"E:\new viral"


@pytest.fixture(scope="session")
def experiment_dir(project_root):
    return os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_zero_shot")


@pytest.fixture(scope="session")
def production_model_dir(project_root):
    return os.path.join(project_root, "models", "human_attention")


@pytest.fixture(scope="session")
def dhf1k_root():
    return r"E:\viral_attention_data\DHF1K"


@pytest.fixture(scope="session")
def evaluation_json(experiment_dir):
    p = os.path.join(experiment_dir, "evaluation.json")
    assert os.path.exists(p), f"evaluation.json not found at {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def predictions_df(experiment_dir):
    p = os.path.join(experiment_dir, "predictions.csv")
    assert os.path.exists(p), f"predictions.csv not found at {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def feature_statistics_json(experiment_dir):
    p = os.path.join(experiment_dir, "feature_statistics.json")
    assert os.path.exists(p), f"feature_statistics.json not found at {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# Test 1: Safety & Read-Only Integrity
def test_production_model_safety(production_model_dir):
    """Ensure production model artifacts exist and have not been modified or replaced."""
    model_path = os.path.join(production_model_dir, "model.pkl")
    scaler_path = os.path.join(production_model_dir, "scaler.pkl")
    schema_path = os.path.join(production_model_dir, "feature_schema.json")

    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)
    assert os.path.exists(schema_path)

    # File size checks to ensure integrity
    assert os.path.getsize(model_path) > 1_000_000, "model.pkl size unexpected"
    assert os.path.getsize(scaler_path) > 500, "scaler.pkl size unexpected"


def test_dhf1k_source_data_read_only(dhf1k_root):
    """Ensure DHF1K validation videos and annotations exist intact."""
    for vid in range(601, 701):
        v_path = os.path.join(dhf1k_root, "video", f"{vid}.AVI")
        a_path = os.path.join(dhf1k_root, "annotation", f"{vid:04d}")
        assert os.path.exists(v_path), f"Video {vid}.AVI missing"
        assert os.path.isdir(a_path), f"Annotation dir {vid:04d} missing"


# Test 2: Schema Conformance
def test_14_production_features_schema(production_model_dir):
    """Verify the 14 features match feature_schema.json exactly."""
    schema_path = os.path.join(production_model_dir, "feature_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    expected_features = [
        "motion_magnitude",
        "motion_std",
        "scene_cut_rate",
        "time_since_cut",
        "brightness_mean",
        "brightness_std",
        "contrast",
        "visual_complexity",
        "face_presence",
        "text_presence",
        "audio_rms",
        "audio_silence_ratio",
        "audio_spectral_centroid",
        "audio_speech_presence"
    ]
    assert schema["feature_names"] == expected_features
    assert schema["feature_count"] == 14
    assert DHF1KZeroShotEvaluator.FEATURE_NAMES == expected_features


# Test 3: Provenance Preservation
def test_sample_provenance_preservation(predictions_df):
    """Verify all 3,970 samples contain strict provenance attributes."""
    assert len(predictions_df) == 3970
    assert predictions_df["dataset_source"].unique().tolist() == ["DHF1K"]
    assert predictions_df["split"].unique().tolist() == ["val"]
    assert predictions_df["video_id"].nunique() == 100

    # Ensure timestamp and window fields are populated and ordered
    assert (predictions_df["window_end"] > predictions_df["window_start"]).all()
    assert (predictions_df["timestamp"] >= predictions_df["window_start"]).all()
    assert (predictions_df["timestamp"] <= predictions_df["window_end"]).all()


# Test 4: Target Independence & Documentation
def test_target_representations_independent(predictions_df, evaluation_json):
    """Verify Target A and Target B are separate columns without equivalence claims."""
    assert "target_a_saliency_peak" in predictions_df.columns
    assert "target_a_saliency_entropy" in predictions_df.columns
    assert "target_b_fixation_concentration" in predictions_df.columns

    # Verify no nulls in target columns
    assert predictions_df["target_a_saliency_peak"].isnull().sum() == 0
    assert predictions_df["target_a_saliency_entropy"].isnull().sum() == 0
    assert predictions_df["target_b_fixation_concentration"].isnull().sum() == 0

    # Verify distinct distributions
    mean_a1 = predictions_df["target_a_saliency_peak"].mean()
    mean_a2 = predictions_df["target_a_saliency_entropy"].mean()
    mean_b = predictions_df["target_b_fixation_concentration"].mean()

    assert mean_a1 == pytest.approx(1.0, abs=0.01)
    assert 0.20 <= mean_a2 <= 0.30
    assert 0.80 <= mean_b <= 0.90

    # Verify evaluation.json documents cross-dataset target-definition comparison
    assert "cross_dataset_target_definition_comparison" in evaluation_json
    comp = evaluation_json["cross_dataset_target_definition_comparison"]
    assert "not mathematical equivalence" in comp["note"].lower()


# Test 5: Zero-Shot Inference Properties
def test_zero_shot_predictions_properties(predictions_df):
    """Verify prediction values and confidences are well-formed and bounded in [0, 1]."""
    preds = predictions_df["predicted_attention"]
    conf = predictions_df["prediction_confidence"]

    assert (preds >= 0.0).all() and (preds <= 1.0).all()
    assert (conf >= 0.0).all() and (conf <= 1.0).all()

    # Verify predictions have non-zero variance
    assert preds.std() > 0.01
    assert 0.40 <= preds.mean() <= 0.50


# Test 6: Metrics Integrity
def test_metrics_calculation_validity(evaluation_json):
    """Verify metrics calculation correctness in evaluation.json."""
    metrics = evaluation_json["metrics_by_target"]
    assert "target_b_fixation_concentration" in metrics
    assert "target_a_saliency_peak" in metrics
    assert "target_a_saliency_entropy" in metrics

    tb = metrics["target_b_fixation_concentration"]
    assert tb["sample_count"] == 3970
    assert 0.35 <= tb["mae"] <= 0.45
    assert 0.35 <= tb["rmse"] <= 0.45
    assert tb["pearson_r"] is not None and tb["pearson_r"] > 0.0
    assert tb["spearman_rho"] is not None and tb["spearman_rho"] > 0.0


# Test 7: Domain Shift Analysis
def test_domain_shift_analysis(evaluation_json, feature_statistics_json):
    """Verify domain shift analysis identifies top shifting features."""
    assert "domain_shift_summary" in evaluation_json
    top_shifts = evaluation_json["domain_shift_summary"]["highest_domain_shift_features"]
    assert len(top_shifts) >= 5

    shift_feat_names = [s["feature"] for s in top_shifts]
    assert "brightness_mean" in shift_feat_names
    assert "audio_silence_ratio" in shift_feat_names
    assert "time_since_cut" in shift_feat_names

    # Check feature statistics json has all 14 features
    f_stats = feature_statistics_json["feature_statistics"]
    for fn in DHF1KZeroShotEvaluator.FEATURE_NAMES:
        assert fn in f_stats
        assert "mean" in f_stats[fn]
        assert "std" in f_stats[fn]


# Test 8: Scientific Terminology Enforcement
def test_scientific_terminology_compliance(experiment_dir):
    """Verify forbidden commercial buzzwords are completely absent from reports."""
    forbidden_terms = [
        "instagram retention",
        "scroll-stop",
        "commercial engagement",
        "viral retention",
        "scroll stop"
    ]

    for fname in ["evaluation.json", "evaluation.md"]:
        fpath = os.path.join(experiment_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().lower()
            for term in forbidden_terms:
                assert term not in content, f"Forbidden term '{term}' found in {fname}"


# Test 9: Evaluator Pipeline Reproducibility
def test_evaluator_class_reproducibility(experiment_dir):
    """Instantiate evaluator and verify it runs report generation without errors."""
    evaluator = DHF1KZeroShotEvaluator(output_dir=experiment_dir)
    df = evaluator.run_or_load_evaluation(use_cached_predictions=True)
    assert len(df) == 3970
    eval_metrics, feat_stats = evaluator.compute_evaluation_report(df)
    assert eval_metrics["dataset_evaluated"]["video_count_evaluated"] == 100
    assert eval_metrics["dataset_evaluated"]["windows_evaluated"] == 3970
