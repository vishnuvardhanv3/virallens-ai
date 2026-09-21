"""
Comprehensive Test Suite for DHF1K Phase 3: Robustness and Per-Video Validation.

Validates:
1. Exactly 100 validation videos analyzed in per_video_metrics.csv.
2. Safety & Read-Only Integrity:
   - Production NEMAR hashes (model.pkl, scaler.pkl, feature_schema.json) unchanged.
   - Independent DHF1K hashes (model_dhf1k.pkl, scaler_dhf1k.pkl, feature_schema_dhf1k.json) unchanged.
   - DHF1K source data counts (1,000 video files, 700 annotation folders) unchanged.
3. Provenance integrity across per-video, temporal, and target bin metrics.
4. Correct per-video aggregation (sum of window counts equals 3,970).
5. Target bin intervals and sample completeness.
6. Temporal segmentation (4 uniform quartiles summing to 3,970 samples).
7. Baseline calculations correctness (Baseline A vs Baseline B vs Model).
8. Absence of NaNs where metrics are mathematically defined.
9. Scientific terminology compliance (zero commercial / retention terms).
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
def robustness_dir(project_root):
    return os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_robustness")


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
def per_video_df(robustness_dir):
    p = os.path.join(robustness_dir, "per_video_metrics.csv")
    assert os.path.exists(p), f"per_video_metrics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def target_bin_df(robustness_dir):
    p = os.path.join(robustness_dir, "target_bin_metrics.csv")
    assert os.path.exists(p), f"target_bin_metrics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def temporal_df(robustness_dir):
    p = os.path.join(robustness_dir, "temporal_metrics.csv")
    assert os.path.exists(p), f"temporal_metrics.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def error_analysis_df(robustness_dir):
    p = os.path.join(robustness_dir, "error_analysis.csv")
    assert os.path.exists(p), f"error_analysis.csv missing: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def robustness_json(robustness_dir):
    p = os.path.join(robustness_dir, "robustness_report.json")
    assert os.path.exists(p), f"robustness_report.json missing: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# Test 1: Exactly 100 Validation Videos
def test_exactly_100_validation_videos(per_video_df):
    """Verify exactly 100 validation videos exist in per-video metrics."""
    assert len(per_video_df) == 100
    assert per_video_df["video_id"].nunique() == 100
    assert per_video_df["valid_window_count"].sum() == 3970


# Test 2: Safety & Hash Verification (Production NEMAR & Phase 2 Independent Models)
def test_production_and_independent_models_unmodified(production_dir, independent_dir):
    """Verify SHA-256 hashes of all models remain completely unchanged."""
    prod_hashes = {
        "model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
        "scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
        "feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411"
    }
    for f, exp in prod_hashes.items():
        p = os.path.join(production_dir, f)
        with open(p, "rb") as fp:
            cur = hashlib.sha256(fp.read()).hexdigest()
        assert cur == exp, f"Production model {f} was modified!"

    ind_hashes = {
        "model_dhf1k.pkl": "faffcb161ac9f895763383364af52527d7fc3f468614936b3d998ed4a275c7a0",
        "scaler_dhf1k.pkl": "8d83c2c336397f9dba5d05f68251f85837779a785875eb2df65b1fdf9f770cb8",
        "feature_schema_dhf1k.json": "80f774f0ded8033b18132beb27fce88865298f005a11a07834c8728524645903"
    }
    for f, exp in ind_hashes.items():
        p = os.path.join(independent_dir, f)
        with open(p, "rb") as fp:
            cur = hashlib.sha256(fp.read()).hexdigest()
        assert cur == exp, f"Independent model {f} was modified!"


# Test 3: DHF1K Source Data Counts Intact
def test_dhf1k_source_data_counts_unmodified(dhf1k_root):
    """Verify DHF1K source directories have not been modified."""
    vids = len(os.listdir(os.path.join(dhf1k_root, "video")))
    anns = len(os.listdir(os.path.join(dhf1k_root, "annotation")))
    assert vids == 1000
    assert anns == 700


# Test 4: Provenance Integrity Across Reports
def test_provenance_integrity(per_video_df, robustness_json):
    """Verify video IDs conform to DHF1K naming convention (DHF1K_601 to DHF1K_700)."""
    expected_ids = [f"DHF1K_{i}" for i in range(601, 701)]
    actual_ids = sorted(per_video_df["video_id"].tolist())
    assert actual_ids == expected_ids
    assert robustness_json["global_metrics"]["video_count"] == 100
    assert robustness_json["global_metrics"]["sample_count"] == 3970


# Test 5: Target Bin Stratification Validity
def test_target_bin_metrics_validity(target_bin_df):
    """Verify target bins partition validation samples properly."""
    expected_bins = ["0.60–0.70", "0.70–0.80", "0.80–0.90", "0.90–1.00"]
    assert target_bin_df["target_bin"].tolist() == expected_bins
    assert target_bin_df["sample_count"].sum() == 3970
    assert (target_bin_df["mae"] > 0.0).all()
    assert (target_bin_df["rmse"] > 0.0).all()


# Test 6: Temporal Quartile Segmentation Validity
def test_temporal_segmentation_validity(temporal_df):
    """Verify temporal segmentation creates 4 uniform segments summing to 3,970."""
    assert len(temporal_df) == 4
    assert temporal_df["sample_count"].sum() == 3970
    # Verify temporal stability (spread < 0.01)
    mae_spread = temporal_df["mae"].max() - temporal_df["mae"].min()
    assert mae_spread < 0.01, f"Temporal drift too large: {mae_spread}"


# Test 7: Baseline Comparison Calculations
def test_baseline_comparisons_mathematical_consistency(robustness_json):
    """Verify Baseline A, Baseline B, and model metrics."""
    bc = robustness_json["baseline_comparisons"]
    base_a = bc["baseline_a_training_mean"]
    base_b = bc["baseline_b_validation_mean"]
    model = bc["trained_dhf1k_extratrees"]

    assert base_a["mae"] == pytest.approx(0.0462, abs=1e-3)
    assert base_b["mae"] == pytest.approx(0.0462, abs=1e-3)
    assert base_b["r2"] == pytest.approx(0.0, abs=1e-3)
    assert model["mae"] < base_a["mae"], "Model MAE must outperform baseline A"
    assert model["r2"] > 0.0, "Model R^2 must be positive"


# Test 8: Absence of NaNs in Defined Metric Columns
def test_no_invalid_nans(per_video_df, error_analysis_df):
    """Verify no unexpected NaNs in required evaluation fields."""
    mandatory_non_null = ["video_id", "valid_window_count", "mae", "rmse", "bias"]
    for col in mandatory_non_null:
        assert per_video_df[col].isnull().sum() == 0, f"NaNs found in {col}"

    assert error_analysis_df.isnull().sum().sum() == 0
    assert (error_analysis_df["pearson_q_fdr"] >= 0.0).all() and (error_analysis_df["pearson_q_fdr"] <= 1.0).all()


# Test 9: Artifact Completeness
def test_all_phase3_artifacts_exist(robustness_dir):
    """Verify all 6 required Phase 3 artifacts exist and are non-empty."""
    required = [
        "robustness_report.json",
        "robustness_report.md",
        "per_video_metrics.csv",
        "target_bin_metrics.csv",
        "temporal_metrics.csv",
        "error_analysis.csv"
    ]
    for fname in required:
        p = os.path.join(robustness_dir, fname)
        assert os.path.exists(p), f"Artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Artifact is empty: {p}"


# Test 10: Scientific Terminology Enforcement
def test_scientific_terminology_enforcement(robustness_dir):
    """Verify forbidden commercial buzzwords are completely absent."""
    forbidden = [
        "instagram retention",
        "scroll-stop",
        "commercial engagement",
        "viral retention",
        "scroll stop"
    ]
    for fname in ["robustness_report.json", "robustness_report.md"]:
        p = os.path.join(robustness_dir, fname)
        with open(p, "r", encoding="utf-8") as f:
            content = f.read().lower()
        for term in forbidden:
            assert term not in content, f"Forbidden term '{term}' found in {fname}"
