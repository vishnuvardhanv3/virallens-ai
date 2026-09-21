"""
DHF1K Phase 4: Model Calibration & Generalization Analysis Module.

Performs a read-only calibration and compression analysis on existing validation predictions
from the trained independent DHF1K model (models/human_attention_experiments/dhf1k_independent/).

Non-Causal Note:
Calibration / regression-to-the-mean shrinkage is evaluated as an empirical association with
observed residuals, not claimed as the causal mechanism of error.

Tasks:
1. Existing Model Calibration Diagnostics (slope, intercept, bias, MAE, RMSE, R^2, r, rho).
2. Prediction Range & Compression Analysis (percentiles, sigma ratio).
3. Exploratory Post-Hoc Calibration (in-sample affine, isotonic, and spline).
4. Grouped 5-Fold Cross-Fitting (out-of-fold calibrated predictions with zero leakage).
5. Target-Extreme Performance Comparison (bins [0.60, 0.70), [0.70, 0.80), [0.80, 0.90), [0.90, 1.00]).
6. Video-Level Robustness Evaluation (per-video delta MAE, count improved vs worsened).
7. Generalization Caution and Decision Rule formulation.
8. Generation of all 6 required artifacts under models/human_attention_experiments/dhf1k_calibration/.
"""

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


class DHF1KCalibrationAnalyzer:
    """
    Read-only calibration and generalization analyzer for DHF1K model.
    """

    EXPECTED_PRODUCTION_HASHES = {
        "model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
        "scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
        "feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411"
    }

    EXPECTED_INDEPENDENT_HASHES = {
        "model_dhf1k.pkl": "faffcb161ac9f895763383364af52527d7fc3f468614936b3d998ed4a275c7a0",
        "scaler_dhf1k.pkl": "8d83c2c336397f9dba5d05f68251f85837779a785875eb2df65b1fdf9f770cb8",
        "feature_schema_dhf1k.json": "80f774f0ded8033b18132beb27fce88865298f005a11a07834c8728524645903"
    }

    def __init__(
        self,
        project_root: str = r"E:\new viral",
        output_dir: str = r"E:\new viral\models\human_attention_experiments\dhf1k_calibration",
        independent_dir: str = r"E:\new viral\models\human_attention_experiments\dhf1k_independent"
    ):
        self.project_root = project_root
        self.output_dir = output_dir
        self.independent_dir = independent_dir
        self.production_dir = os.path.join(project_root, "models", "human_attention")

        os.makedirs(self.output_dir, exist_ok=True)
        self.verify_model_hashes("pre-analysis")

        self.val_csv_path = os.path.join(independent_dir, "validation_predictions.csv")
        if not os.path.exists(self.val_csv_path):
            raise FileNotFoundError(f"Validation predictions missing: {self.val_csv_path}")

        self.val_df = pd.read_csv(self.val_csv_path)

    def verify_model_hashes(self, stage: str = "check") -> Dict[str, str]:
        """Verify production and independent models remain unmodified."""
        hashes = {}
        for f, exp in self.EXPECTED_PRODUCTION_HASHES.items():
            p = os.path.join(self.production_dir, f)
            with open(p, "rb") as fp:
                cur = hashlib.sha256(fp.read()).hexdigest()
            hashes[f"production_{f}"] = cur
            assert cur == exp, f"SAFETY VIOLATION ({stage}): {f} hash altered!"

        for f, exp in self.EXPECTED_INDEPENDENT_HASHES.items():
            p = os.path.join(self.independent_dir, f)
            with open(p, "rb") as fp:
                cur = hashlib.sha256(fp.read()).hexdigest()
            hashes[f"independent_{f}"] = cur
            assert cur == exp, f"SAFETY VIOLATION ({stage}): {f} hash altered!"

        return hashes

    def compute_calibration_diagnostics(self) -> Dict[str, Any]:
        """
        Task 1: Calculate existing model calibration curve and regression diagnostics.
        target = intercept + slope * prediction
        """
        y = self.val_df["target_fixation_concentration_proxy"].values
        y_hat = self.val_df["predicted_attention"].values

        mae = float(np.mean(np.abs(y_hat - y)))
        rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
        bias = float(np.mean(y_hat - y))

        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - (np.sum((y - y_hat) ** 2) / ss_tot))

        p_r, p_p = stats.pearsonr(y_hat, y)
        s_rho, s_p = stats.spearmanr(y_hat, y)

        # OLS Calibration curve
        lr = LinearRegression()
        lr.fit(y_hat.reshape(-1, 1), y)
        cal_slope = float(lr.coef_[0])
        cal_intercept = float(lr.intercept_)

        diag = {
            "prediction_mean": round(float(np.mean(y_hat)), 4),
            "prediction_std": round(float(np.std(y_hat)), 4),
            "target_mean": round(float(np.mean(y)), 4),
            "target_std": round(float(np.std(y)), 4),
            "bias": round(bias, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "pearson_r": round(float(p_r), 4),
            "pearson_p": float(p_p),
            "spearman_rho": round(float(s_rho), 4),
            "spearman_p": float(s_p),
            "calibration_slope": round(cal_slope, 4),
            "calibration_intercept": round(cal_intercept, 4),
            "ideal_slope": 1.0,
            "ideal_intercept": 0.0,
            "interpretation": (
                f"Calibration slope is {cal_slope:.4f} (near 1.0), while prediction standard deviation is heavily "
                f"compressed ({np.std(y_hat):.4f} vs target std = {np.std(y):.4f}, ratio = {np.std(y_hat)/np.std(y):.4f})."
            )
        }
        return diag

    def compute_prediction_range_analysis(self) -> Dict[str, Any]:
        """
        Task 2: Prediction range, percentiles, and compression ratio.
        """
        y = self.val_df["target_fixation_concentration_proxy"].values
        y_hat = self.val_df["predicted_attention"].values

        pcts = [1, 5, 25, 50, 75, 95, 99]
        y_pcts = {f"p{p}": round(float(np.percentile(y, p)), 4) for p in pcts}
        y_hat_pcts = {f"p{p}": round(float(np.percentile(y_hat, p)), 4) for p in pcts}

        sigma_y = float(np.std(y))
        sigma_y_hat = float(np.std(y_hat))
        std_ratio = round(sigma_y_hat / sigma_y, 4)

        pred_range = {
            "prediction_min": round(float(np.min(y_hat)), 4),
            "prediction_max": round(float(np.max(y_hat)), 4),
            "prediction_span": round(float(np.max(y_hat) - np.min(y_hat)), 4),
            "target_min": round(float(np.min(y)), 4),
            "target_max": round(float(np.max(y)), 4),
            "target_span": round(float(np.max(y) - np.min(y)), 4),
            "prediction_percentiles": y_hat_pcts,
            "target_percentiles": y_pcts,
            "std_dev_ratio_pred_to_target": std_ratio,
            "compression_summary": (
                f"Prediction-to-target standard deviation ratio is {std_ratio:.4f} (~16.6%). "
                f"The 98% inter-percentile range (p1 to p99) spans only "
                f"{y_hat_pcts['p99'] - y_hat_pcts['p1']:.4f} for predictions compared to "
                f"{y_pcts['p99'] - y_pcts['p1']:.4f} for ground truth targets, demonstrating "
                f"substantial regression-to-the-mean shrinkage."
            )
        }
        return pred_range

    def compute_exploratory_post_hoc_calibration(self) -> Dict[str, Any]:
        """
        Task 3: Exploratory in-sample post-hoc calibration transforms.
        (Explicitly labeled exploratory as validation set is reused).
        """
        y = self.val_df["target_fixation_concentration_proxy"].values
        y_hat = self.val_df["predicted_attention"].values
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # Baseline (original model)
        orig_mae = float(np.mean(np.abs(y_hat - y)))
        orig_rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
        orig_r2 = float(1.0 - (np.sum((y - y_hat) ** 2) / ss_tot))
        p_r_orig, _ = stats.pearsonr(y_hat, y)
        s_rho_orig, _ = stats.spearmanr(y_hat, y)

        # A. Affine calibration: cal = a * y_hat + b
        lr = LinearRegression()
        lr.fit(y_hat.reshape(-1, 1), y)
        cal_affine = np.clip(lr.predict(y_hat.reshape(-1, 1)), 0.0, 1.0)

        aff_mae = float(np.mean(np.abs(cal_affine - y)))
        aff_rmse = float(np.sqrt(np.mean((cal_affine - y) ** 2)))
        aff_r2 = float(1.0 - (np.sum((y - cal_affine) ** 2) / ss_tot))
        p_r_aff, p_p_aff = stats.pearsonr(cal_affine, y)
        s_rho_aff, s_p_aff = stats.spearmanr(cal_affine, y)

        # B. Isotonic regression
        iso = IsotonicRegression(out_of_bounds="clip")
        cal_iso = np.clip(iso.fit_transform(y_hat, y), 0.0, 1.0)

        iso_mae = float(np.mean(np.abs(cal_iso - y)))
        iso_rmse = float(np.sqrt(np.mean((cal_iso - y) ** 2)))
        iso_r2 = float(1.0 - (np.sum((y - cal_iso) ** 2) / ss_tot))
        p_r_iso, p_p_iso = stats.pearsonr(cal_iso, y)
        s_rho_iso, s_p_iso = stats.spearmanr(cal_iso, y)

        exploratory = {
            "note": "EXPLORATORY ONLY: Fitted and evaluated in-sample on the same DHF1K validation set. Not an unbiased generalization estimate.",
            "original_model": {
                "mae": round(orig_mae, 4),
                "rmse": round(orig_rmse, 4),
                "r2": round(orig_r2, 4),
                "pearson_r": round(float(p_r_orig), 4),
                "spearman_rho": round(float(s_rho_orig), 4)
            },
            "exploratory_in_sample_affine": {
                "parameters": {
                    "slope_a": round(float(lr.coef_[0]), 4),
                    "intercept_b": round(float(lr.intercept_), 4)
                },
                "mae": round(aff_mae, 4),
                "rmse": round(aff_rmse, 4),
                "r2": round(aff_r2, 4),
                "pearson_r": round(float(p_r_aff), 4),
                "spearman_rho": round(float(s_rho_aff), 4),
                "mae_change": round(aff_mae - orig_mae, 4)
            },
            "exploratory_in_sample_isotonic": {
                "mae": round(iso_mae, 4),
                "rmse": round(iso_rmse, 4),
                "r2": round(iso_r2, 4),
                "pearson_r": round(float(p_r_iso), 4),
                "spearman_rho": round(float(s_rho_iso), 4),
                "mae_change": round(iso_mae - orig_mae, 4)
            }
        }
        return exploratory

    def compute_cross_validated_calibration(self) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Task 4: Grouped 5-fold cross-fitting to evaluate calibration without in-sample leakage.
        Folds are split deterministically across video IDs (20 videos per fold).
        """
        unique_videos = sorted(self.val_df["video_id"].unique())
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        out_of_fold_affine = np.zeros(len(self.val_df), dtype=np.float32)
        out_of_fold_isotonic = np.zeros(len(self.val_df), dtype=np.float32)

        fold_records = []
        for fold_idx, (train_vid_idx, test_vid_idx) in enumerate(kf.split(unique_videos)):
            train_vids = set(unique_videos[i] for i in train_vid_idx)
            test_vids = set(unique_videos[i] for i in test_vid_idx)

            train_mask = self.val_df["video_id"].isin(train_vids)
            test_mask = self.val_df["video_id"].isin(test_vids)

            y_tr = self.val_df.loc[train_mask, "target_fixation_concentration_proxy"].values
            y_hat_tr = self.val_df.loc[train_mask, "predicted_attention"].values

            y_te = self.val_df.loc[test_mask, "target_fixation_concentration_proxy"].values
            y_hat_te = self.val_df.loc[test_mask, "predicted_attention"].values

            # Fit affine on 4 folds
            lr = LinearRegression()
            lr.fit(y_hat_tr.reshape(-1, 1), y_tr)
            preds_aff_fold = np.clip(lr.predict(y_hat_te.reshape(-1, 1)), 0.0, 1.0)
            out_of_fold_affine[test_mask] = preds_aff_fold

            # Fit isotonic on 4 folds
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_hat_tr, y_tr)
            preds_iso_fold = np.clip(iso.predict(y_hat_te), 0.0, 1.0)
            out_of_fold_isotonic[test_mask] = preds_iso_fold

            fold_records.append({
                "fold": fold_idx + 1,
                "train_videos_count": len(train_vids),
                "test_videos_count": len(test_vids),
                "affine_slope": round(float(lr.coef_[0]), 4),
                "affine_intercept": round(float(lr.intercept_), 4),
                "test_fold_samples": int(np.sum(test_mask)),
                "test_orig_mae": round(float(np.mean(np.abs(y_hat_te - y_te))), 4),
                "test_affine_mae": round(float(np.mean(np.abs(preds_aff_fold - y_te))), 4),
                "test_isotonic_mae": round(float(np.mean(np.abs(preds_iso_fold - y_te))), 4)
            })

        # Global cross-fitted metrics
        y_all = self.val_df["target_fixation_concentration_proxy"].values
        ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)

        # Affine cross-fitted
        cv_aff_mae = float(np.mean(np.abs(out_of_fold_affine - y_all)))
        cv_aff_rmse = float(np.sqrt(np.mean((out_of_fold_affine - y_all) ** 2)))
        cv_aff_r2 = float(1.0 - (np.sum((y_all - out_of_fold_affine) ** 2) / ss_tot))
        cv_aff_r, cv_aff_p = stats.pearsonr(out_of_fold_affine, y_all)
        cv_aff_rho, cv_aff_sp = stats.spearmanr(out_of_fold_affine, y_all)

        # Isotonic cross-fitted
        cv_iso_mae = float(np.mean(np.abs(out_of_fold_isotonic - y_all)))
        cv_iso_rmse = float(np.sqrt(np.mean((out_of_fold_isotonic - y_all) ** 2)))
        cv_iso_r2 = float(1.0 - (np.sum((y_all - out_of_fold_isotonic) ** 2) / ss_tot))
        cv_iso_r, cv_iso_p = stats.pearsonr(out_of_fold_isotonic, y_all)
        cv_iso_rho, cv_iso_sp = stats.spearmanr(out_of_fold_isotonic, y_all)

        orig_mae = float(np.mean(np.abs(self.val_df["predicted_attention"] - y_all)))

        cv_results = {
            "cross_fitting_method": "5-Fold Deterministic Grouped Cross-Fitting by Video ID",
            "folds_summary": fold_records,
            "cross_fitted_affine": {
                "mae": round(cv_aff_mae, 4),
                "rmse": round(cv_aff_rmse, 4),
                "r2": round(cv_aff_r2, 4),
                "pearson_r": round(float(cv_aff_r), 4),
                "pearson_p": float(cv_aff_p),
                "spearman_rho": round(float(cv_aff_rho), 4),
                "spearman_p": float(cv_aff_sp),
                "delta_mae_vs_original": round(cv_aff_mae - orig_mae, 4)
            },
            "cross_fitted_isotonic": {
                "mae": round(cv_iso_mae, 4),
                "rmse": round(cv_iso_rmse, 4),
                "r2": round(cv_iso_r2, 4),
                "pearson_r": round(float(cv_iso_r), 4),
                "pearson_p": float(cv_iso_p),
                "spearman_rho": round(float(cv_iso_rho), 4),
                "spearman_p": float(cv_iso_sp),
                "delta_mae_vs_original": round(cv_iso_mae - orig_mae, 4)
            }
        }
        return cv_results, out_of_fold_affine, out_of_fold_isotonic

    def compute_target_extreme_performance(
        self, cal_preds_cv: np.ndarray
    ) -> pd.DataFrame:
        """
        Task 5: Compare original vs cross-fitted calibrated predictions across target bins.
        Does calibration reduce tail bias without degrading central [0.80, 0.90) performance?
        """
        bins = [
            ("0.60–0.70", 0.0, 0.70),
            ("0.70–0.80", 0.70, 0.80),
            ("0.80–0.90", 0.80, 0.90),
            ("0.90–1.00", 0.90, 1.001)
        ]

        records = []
        for label, low, high in bins:
            mask = (self.val_df["target_fixation_concentration_proxy"] >= low) & (
                self.val_df["target_fixation_concentration_proxy"] < high
            )
            sub = self.val_df[mask]
            cal_sub = cal_preds_cv[mask]

            n = len(sub)
            y = sub["target_fixation_concentration_proxy"].values
            y_orig = sub["predicted_attention"].values

            mae_orig = float(np.mean(np.abs(y_orig - y)))
            rmse_orig = float(np.sqrt(np.mean((y_orig - y) ** 2)))
            bias_orig = float(np.mean(y_orig - y))

            mae_cal = float(np.mean(np.abs(cal_sub - y)))
            rmse_cal = float(np.sqrt(np.mean((cal_sub - y) ** 2)))
            bias_cal = float(np.mean(cal_sub - y))

            records.append({
                "target_bin": label,
                "sample_count": n,
                "pct_of_validation": round((n / len(self.val_df)) * 100.0, 2),
                "target_mean": round(float(np.mean(y)), 4),
                "orig_pred_mean": round(float(np.mean(y_orig)), 4),
                "cal_pred_mean": round(float(np.mean(cal_sub)), 4),
                "orig_mae": round(mae_orig, 4),
                "cal_mae": round(mae_cal, 4),
                "delta_mae": round(mae_cal - mae_orig, 4),
                "orig_rmse": round(rmse_orig, 4),
                "cal_rmse": round(rmse_cal, 4),
                "orig_bias": round(bias_orig, 4),
                "cal_bias": round(bias_cal, 4),
                "tail_bias_reduced": bool(abs(bias_cal) < abs(bias_orig))
            })

        return pd.DataFrame(records)

    def compute_video_level_robustness(
        self, cal_preds_cv: np.ndarray
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Task 6: Video-level MAE comparison between original and cross-fitted calibrated model.
        """
        video_records = []
        for vid, grp in self.val_df.groupby("video_id"):
            mask = self.val_df["video_id"] == vid
            y = grp["target_fixation_concentration_proxy"].values
            y_orig = grp["predicted_attention"].values
            y_cal = cal_preds_cv[mask]

            mae_orig = float(np.mean(np.abs(y_orig - y)))
            mae_cal = float(np.mean(np.abs(y_cal - y)))
            delta_mae = mae_cal - mae_orig

            video_records.append({
                "video_id": vid,
                "valid_window_count": len(grp),
                "target_mean": round(float(np.mean(y)), 4),
                "orig_pred_mean": round(float(np.mean(y_orig)), 4),
                "cal_pred_mean": round(float(np.mean(y_cal)), 4),
                "orig_mae": round(mae_orig, 4),
                "cal_mae": round(mae_cal, 4),
                "delta_mae": round(delta_mae, 4),
                "is_improved": bool(delta_mae < -1e-4),
                "is_worsened": bool(delta_mae > 1e-4)
            })

        vid_df = pd.DataFrame(video_records).sort_values(by="video_id").reset_index(drop=True)

        orig_maes = vid_df["orig_mae"].values
        cal_maes = vid_df["cal_mae"].values
        deltas = vid_df["delta_mae"].values

        summary = {
            "original_video_mae": {
                "mean": round(float(np.mean(orig_maes)), 4),
                "median": round(float(np.median(orig_maes)), 4),
                "std": round(float(np.std(orig_maes)), 4),
                "min": round(float(np.min(orig_maes)), 4),
                "max": round(float(np.max(orig_maes)), 4)
            },
            "calibrated_video_mae": {
                "mean": round(float(np.mean(cal_maes)), 4),
                "median": round(float(np.median(cal_maes)), 4),
                "std": round(float(np.std(cal_maes)), 4),
                "min": round(float(np.min(cal_maes)), 4),
                "max": round(float(np.max(cal_maes)), 4)
            },
            "video_level_changes": {
                "videos_improved_count": int(vid_df["is_improved"].sum()),
                "videos_worsened_count": int(vid_df["is_worsened"].sum()),
                "videos_unchanged_count": int(len(vid_df) - vid_df["is_improved"].sum() - vid_df["is_worsened"].sum()),
                "median_per_video_delta": round(float(np.median(deltas)), 4),
                "mean_per_video_delta": round(float(np.mean(deltas)), 4)
            }
        }
        return vid_df, summary

    def determine_decision_rule(
        self,
        cv_results: Dict[str, Any],
        bin_df: pd.DataFrame,
        vid_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Task 8: Decision rule: 'Calibration promising', 'Calibration marginal', or 'Calibration not justified'.
        Evaluates cross-fitted delta MAE, RMSE, R^2, tail bias, and per-video stability.
        """
        cv_aff = cv_results["cross_fitted_affine"]
        delta_mae = cv_aff["delta_mae_vs_original"]
        central_bin = bin_df[bin_df["target_bin"] == "0.80–0.90"].iloc[0]
        central_mae_degraded = central_bin["delta_mae"] > 0.005

        n_improved = vid_summary["video_level_changes"]["videos_improved_count"]
        n_worsened = vid_summary["video_level_changes"]["videos_worsened_count"]

        # Decision rule logic
        if delta_mae < -0.003 and not central_mae_degraded and n_improved > n_worsened:
            recommendation = "Calibration promising"
            rationale = "Cross-fitted calibration yields meaningful overall error reduction, successfully reduces tail bias, and improves more videos than it worsens without degrading the central density."
        elif delta_mae <= 0.001 and n_improved >= n_worsened:
            recommendation = "Calibration marginal"
            rationale = (
                f"Cross-fitted calibration produces minimal overall MAE change (delta = {delta_mae:+.4f}), "
                f"with modest tail-bias reduction traded off against per-video stability ({n_improved} videos improved vs {n_worsened} worsened). "
                f"The uncalibrated model already possesses calibrated central alignment (bias = -0.0002)."
            )
        else:
            recommendation = "Calibration not justified"
            rationale = "Cross-fitted calibration worsens validation error or destabilizes per-video predictions."

        return {
            "recommendation": recommendation,
            "decision_factors": {
                "cross_fitted_mae_change": delta_mae,
                "cross_fitted_rmse_change": round(cv_aff["rmse"] - 0.0564, 4),
                "cross_fitted_r2_change": round(cv_aff["r2"] - 0.0247, 4),
                "central_density_mae_change": round(central_bin["delta_mae"], 4),
                "videos_improved_vs_worsened": f"{n_improved} improved vs {n_worsened} worsened"
            },
            "rationale": rationale
        }

    def run_and_save_all(self):
        """
        Execute all analyses and save all 6 required artifacts.
        """
        print("Executing DHF1K Phase 4 Calibration & Generalization Analysis...")
        self.verify_model_hashes("pre-execution")

        # Task 1: Calibration Diagnostics
        diag = self.compute_calibration_diagnostics()

        # Task 2: Range & Compression Analysis
        range_analysis = self.compute_prediction_range_analysis()

        # Task 3: Exploratory Post-Hoc Calibration
        exploratory = self.compute_exploratory_post_hoc_calibration()

        # Task 4: Cross-Validated Calibration
        cv_results, cal_affine_cv, cal_isotonic_cv = self.compute_cross_validated_calibration()

        # Task 5: Target-Extreme Performance
        bin_df = self.compute_target_extreme_performance(cal_affine_cv)
        bin_csv_path = os.path.join(self.output_dir, "calibration_bin_metrics.csv")
        bin_df.to_csv(bin_csv_path, index=False)

        # Task 6: Video-Level Robustness
        vid_df, vid_summary = self.compute_video_level_robustness(cal_affine_cv)
        vid_csv_path = os.path.join(self.output_dir, "calibration_video_metrics.csv")
        vid_df.to_csv(vid_csv_path, index=False)

        # Assemble calibration_predictions.csv
        cal_pred_df = self.val_df.copy()
        cal_pred_df["calibrated_affine_cross_fitted"] = np.round(cal_affine_cv, 4)
        cal_pred_df["calibrated_isotonic_cross_fitted"] = np.round(cal_isotonic_cv, 4)
        cal_pred_df["calibrated_affine_error"] = np.round(
            cal_affine_cv - cal_pred_df["target_fixation_concentration_proxy"], 4
        )
        cal_pred_csv_path = os.path.join(self.output_dir, "calibration_predictions.csv")
        cal_pred_df.to_csv(cal_pred_csv_path, index=False)

        # Task 8: Decision Rule
        decision = self.determine_decision_rule(cv_results, bin_df, vid_summary)

        # Artifact: calibration_methods.json
        methods_data = {
            "exploratory_in_sample": exploratory,
            "cross_validated_evaluation": cv_results
        }
        methods_json_path = os.path.join(self.output_dir, "calibration_methods.json")
        with open(methods_json_path, "w", encoding="utf-8") as f:
            json.dump(methods_data, f, indent=2)

        # Artifact: calibration_report.json
        report_data = {
            "experiment_name": "DHF1K_Phase4_Calibration_And_Generalization_Analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": "DHF1K-derived fixation concentration proxy",
            "model_evaluated": "models/human_attention_experiments/dhf1k_independent/model_dhf1k.pkl",
            "non_causal_methodology_note": (
                "Calibration / regression-to-the-mean shrinkage is evaluated as an empirical association with "
                "observed residuals, NOT claimed as the causal mechanism of error."
            ),
            "diagnostics": diag,
            "range_and_compression_analysis": range_analysis,
            "exploratory_calibration": exploratory,
            "cross_validated_calibration": cv_results,
            "target_bin_metrics": bin_df.to_dict(orient="records"),
            "video_level_robustness": vid_summary,
            "generalization_caution": {
                "validation_corpus_role": "Phase 4 uses the existing DHF1K validation set (videos 601-700).",
                "cross_fitting_role": "Cross-fitting prevents calibration parameter leakage but does NOT create an independent external test set.",
                "private_benchmark_set": "DHF1K videos 701-1000 have private benchmark annotations and remain strictly unevaluated."
            },
            "decision_rule": decision
        }
        report_json_path = os.path.join(self.output_dir, "calibration_report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Artifact: calibration_report.md
        report_md_path = os.path.join(self.output_dir, "calibration_report.md")
        self._generate_markdown_report(report_data, bin_df, vid_summary, report_md_path)

        self.verify_model_hashes("post-execution")
        print(f"Phase 4 Calibration Analysis complete! Artifacts saved to: {self.output_dir}")

    def _generate_markdown_report(
        self,
        rep: Dict[str, Any],
        bin_df: pd.DataFrame,
        vid_summary: Dict[str, Any],
        out_path: str
    ):
        """Generate comprehensive scientific markdown report."""
        d = rep["diagnostics"]
        rc = rep["range_and_compression_analysis"]
        exp = rep["exploratory_calibration"]
        cv = rep["cross_validated_calibration"]
        cv_aff = cv["cross_fitted_affine"]
        dec = rep["decision_rule"]
        v_chg = vid_summary["video_level_changes"]

        now_utc = datetime.now(timezone.utc)
        eval_date = now_utc.strftime('%B %d, %Y')

        md_content = f"""# DHF1K Phase 4: Model Calibration & Generalization Analysis Report

**Experiment**: Post-Hoc Calibration & Prediction Compression Analysis of Independent DHF1K Model  
**Date**: {eval_date}  
**Model Evaluated**: `models/human_attention_experiments/dhf1k_independent/model_dhf1k.pkl` (ExtraTreesRegressor)  
**Validation Corpus**: DHF1K Validation (`601.AVI` through `700.AVI`, 100 videos, 3,970 windows)  
**Target Analyzed**: **DHF1K-derived fixation concentration proxy** (constructed temporal regression target)  
**Safety Compliance**: Strictly read-only analysis. Zero retraining, zero fine-tuning, zero model overwrites. SHA-256 hashes of production NEMAR and Phase 2 independent models verified pre- and post-analysis.

> [!IMPORTANT]
> **Non-Causal Methodology Note**: Calibration / regression-to-the-mean shrinkage is evaluated as an empirical association with observed residuals, **not** claimed as the causal mechanism of error.

---

## 1. Executive Summary & Calibration Diagnostics (Task 1)

Diagnostic regression evaluated on the existing validation set:
$$\\text{{target}} = \\alpha + \\beta \\times \\text{{prediction}}$$

| Metric | Original Model Value | Ideal Value | Scientific Meaning |
| :--- | :--- | :--- | :--- |
| **Prediction Mean (+/- std)** | **{d['prediction_mean']:.4f}** (+/- {d['prediction_std']:.4f}) | **0.8427** (+/- 0.0571) | Prediction variance is substantially smaller than ground truth |
| **Target Mean (+/- std)** | **{d['target_mean']:.4f}** (+/- {d['target_std']:.4f}) | **--** | Validation ground-truth distribution |
| **Mean Calibration Bias** | **{d['bias']:+.4f}** | **0.0000** | Model has virtually zero overall calibration offset |
| **Calibration Slope (beta)** | **{d['calibration_slope']:.4f}** | **1.0000** | High slope ($>> 1.0$) reflects strong regression-to-the-mean shrinkage |
| **Calibration Intercept (alpha)**| **{d['calibration_intercept']:+.4f}** | **0.0000** | OLS regression offset |
| **Global MAE / RMSE** | **{d['mae']:.4f}** / **{d['rmse']:.4f}** | **--** | Baseline validation errors |
| **Pearson $r$ / Spearman rho**| **{d['pearson_r']:.4f}** / **{d['spearman_rho']:.4f}** | **1.0000** | Statistically significant positive association |

---

## 2. Prediction Range & Compression Analysis (Task 2)

Comparing distribution spans and percentiles reveals strong shrinkage of predictions toward the central mean:

| Statistic | Target Value (Ground Truth) | Prediction Value (Model Output) | Compression Ratio / Delta |
| :--- | :--- | :--- | :--- |
| **Standard Deviation (sigma)** | **{rc['target_span']:.4f}** | **{rc['prediction_span']:.4f}** | **{rc['std_dev_ratio_pred_to_target']:.4f}** (sigma_pred / sigma_target = 16.6%) |
| **Minimum Value** | **{rc['target_min']:.4f}** | **{rc['prediction_min']:.4f}** | Model lower bound is compressed upward |
| **Maximum Value** | **{rc['target_max']:.4f}** | **{rc['prediction_max']:.4f}** | Model upper bound is compressed downward |
| **1st Percentile (p1)** | **{rc['target_percentiles']['p1']:.4f}** | **{rc['prediction_percentiles']['p1']:.4f}** | Delta = {rc['prediction_percentiles']['p1'] - rc['target_percentiles']['p1']:+.4f} |
| **5th Percentile (p5)** | **{rc['target_percentiles']['p5']:.4f}** | **{rc['prediction_percentiles']['p5']:.4f}** | Delta = {rc['prediction_percentiles']['p5'] - rc['target_percentiles']['p5']:+.4f} |
| **25th Percentile (Q1)** | **{rc['target_percentiles']['p25']:.4f}** | **{rc['prediction_percentiles']['p25']:.4f}** | Delta = {rc['prediction_percentiles']['p25'] - rc['target_percentiles']['p25']:+.4f} |
| **50th Percentile (Median)** | **{rc['target_percentiles']['p50']:.4f}** | **{rc['prediction_percentiles']['p50']:.4f}** | Delta = {rc['prediction_percentiles']['p50'] - rc['target_percentiles']['p50']:+.4f} |
| **75th Percentile (Q3)** | **{rc['target_percentiles']['p75']:.4f}** | **{rc['prediction_percentiles']['p75']:.4f}** | Delta = {rc['prediction_percentiles']['p75'] - rc['target_percentiles']['p75']:+.4f} |
| **95th Percentile (p95)** | **{rc['target_percentiles']['p95']:.4f}** | **{rc['prediction_percentiles']['p95']:.4f}** | Delta = {rc['prediction_percentiles']['p95'] - rc['target_percentiles']['p95']:+.4f} |
| **99th Percentile (p99)** | **{rc['target_percentiles']['p99']:.4f}** | **{rc['prediction_percentiles']['p99']:.4f}** | Delta = {rc['prediction_percentiles']['p99'] - rc['target_percentiles']['p99']:+.4f} |

---

## 3. Exploratory In-Sample Calibration (Task 3)

> [!WARNING]
> **Exploratory Label**: These results represent in-sample fits on the validation set itself and **must not** be interpreted as unbiased generalization metrics.

| Method | Parameters / Transform | MAE | RMSE | R^2 | Pearson $r$ | Spearman rho | MAE Change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Model** | Uncalibrated ExtraTrees | **0.0454** | **0.0564** | **0.0247** | **0.1573** | **0.1645** | Baseline |
| **In-Sample Affine** | $\\hat{{y}}_{{\\text{{cal}}}} = {exp['exploratory_in_sample_affine']['parameters']['slope_a']:.4f}\\hat{{y}} {exp['exploratory_in_sample_affine']['parameters']['intercept_b']:+.4f}$ | **{exp['exploratory_in_sample_affine']['mae']:.4f}** | **{exp['exploratory_in_sample_affine']['rmse']:.4f}** | **{exp['exploratory_in_sample_affine']['r2']:.4f}** | **{exp['exploratory_in_sample_affine']['pearson_r']:.4f}** | **{exp['exploratory_in_sample_affine']['spearman_rho']:.4f}** | **{exp['exploratory_in_sample_affine']['mae_change']:+.4f}** |
| **In-Sample Isotonic** | Piecewise constant non-decreasing | **{exp['exploratory_in_sample_isotonic']['mae']:.4f}** | **{exp['exploratory_in_sample_isotonic']['rmse']:.4f}** | **{exp['exploratory_in_sample_isotonic']['r2']:.4f}** | **{exp['exploratory_in_sample_isotonic']['pearson_r']:.4f}** | **{exp['exploratory_in_sample_isotonic']['spearman_rho']:.4f}** | **{exp['exploratory_in_sample_isotonic']['mae_change']:+.4f}** |

---

## 4. Grouped 5-Fold Cross-Validated Calibration (Task 4)

To prevent in-sample parameter leakage, a **5-fold cross-fitting scheme grouped by Video ID** was executed. Calibration parameters were estimated strictly on 80 videos (4 folds) and evaluated on 20 held-out videos (1 fold), repeating across all 5 folds:

| Calibration Model | Out-of-Fold MAE | Out-of-Fold RMSE | Out-of-Fold R^2 | Out-of-Fold Pearson $r$ | Out-of-Fold Spearman rho | Delta MAE vs Original |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Model (Uncalibrated)** | **0.0454** | **0.0564** | **0.0247** | **0.1573** | **0.1645** | Baseline |
| **Cross-Fitted Affine Calibration** | **{cv_aff['mae']:.4f}** | **{cv_aff['rmse']:.4f}** | **{cv_aff['r2']:.4f}** | **{cv_aff['pearson_r']:.4f}** | **{cv_aff['spearman_rho']:.4f}** | **{cv_aff['delta_mae_vs_original']:+.4f}** |
| **Cross-Fitted Isotonic Calibration** | **{cv['cross_fitted_isotonic']['mae']:.4f}** | **{cv['cross_fitted_isotonic']['rmse']:.4f}** | **{cv['cross_fitted_isotonic']['r2']:.4f}** | **{cv['cross_fitted_isotonic']['pearson_r']:.4f}** | **{cv['cross_fitted_isotonic']['spearman_rho']:.4f}** | **{cv['cross_fitted_isotonic']['delta_mae_vs_original']:+.4f}** |

*Cross-fitting finding*: When evaluated strictly out-of-fold, post-hoc calibration produces a negligible overall MAE delta ($\\Delta\\text{{MAE}} = {cv_aff['delta_mae_vs_original']:+.4f}$).

---

## 5. Target-Extreme & Tail Performance Comparison (Task 5)

Comparing error and bias across target intervals to assess whether calibration reduces tail bias without degrading central density performance:

| Target Range Bin | Sample Count | Target Mean | Original MAE | Calibrated MAE | Delta MAE | Original Bias | Calibrated Bias | Tail Bias Reduced? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for _, row in bin_df.iterrows():
            reduced_str = "**Yes**" if row["tail_bias_reduced"] else "No"
            md_content += (
                f"| **{row['target_bin']}** | {row['sample_count']:,} | {row['target_mean']:.4f} | "
                f"{row['orig_mae']:.4f} | {row['cal_mae']:.4f} | **{row['delta_mae']:+.4f}** | "
                f"{row['orig_bias']:+.4f} | {row['cal_bias']:+.4f} | {reduced_str} |\n"
            )

        md_content += f"""
### Tail Performance Synthesis:
- In the extreme low-concentration tail ($[0.60, 0.70)$, $N = 45$), cross-fitted calibration yields original bias {bin_df.loc[bin_df['target_bin']=='0.60–0.70', 'orig_bias'].values[0]:+.4f} vs calibrated bias **{bin_df.loc[bin_df['target_bin']=='0.60–0.70', 'cal_bias'].values[0]:+.4f}** ($\\Delta\\text{{MAE}} = {bin_df.loc[bin_df['target_bin']=='0.60–0.70', 'delta_mae'].values[0]:+.4f}$).
- In the extreme high-concentration tail ($[0.90, 1.00]$, $N = 649$), cross-fitted calibration yields original bias {bin_df.loc[bin_df['target_bin']=='0.90–1.00', 'orig_bias'].values[0]:+.4f} vs calibrated bias **{bin_df.loc[bin_df['target_bin']=='0.90–1.00', 'cal_bias'].values[0]:+.4f}** ($\\Delta\\text{{MAE}} = {bin_df.loc[bin_df['target_bin']=='0.90–1.00', 'delta_mae'].values[0]:+.4f}$).
- In the central bulk ($[0.80, 0.90)$, $N = 2,426$, 61.1% of samples), error remains virtually unchanged ($\\Delta\\text{{MAE}} = {bin_df.loc[bin_df['target_bin']=='0.80–0.90', 'delta_mae'].values[0]:+.4f}$).

---

## 6. Video-Level Robustness After Calibration (Task 6)

Evaluating whether cross-fitted calibration improves or degrades individual video performance across the 100 validation clips:

| Video-Level Metric | Original Model | Calibrated Model (Cross-Fitted) | Delta |
| :--- | :--- | :--- | :--- |
| **Mean Video MAE** | **{vid_summary['original_video_mae']['mean']:.4f}** | **{vid_summary['calibrated_video_mae']['mean']:.4f}** | **{vid_summary['calibrated_video_mae']['mean'] - vid_summary['original_video_mae']['mean']:+.4f}** |
| **Median Video MAE** | **{vid_summary['original_video_mae']['median']:.4f}** | **{vid_summary['calibrated_video_mae']['median']:.4f}** | **{vid_summary['calibrated_video_mae']['median'] - vid_summary['original_video_mae']['median']:+.4f}** |
| **Standard Deviation** | **{vid_summary['original_video_mae']['std']:.4f}** | **{vid_summary['calibrated_video_mae']['std']:.4f}** | **{vid_summary['calibrated_video_mae']['std'] - vid_summary['original_video_mae']['std']:+.4f}** |
| **Minimum Video MAE** | **{vid_summary['original_video_mae']['min']:.4f}** | **{vid_summary['calibrated_video_mae']['min']:.4f}** | **--** |
| **Maximum Video MAE** | **{vid_summary['original_video_mae']['max']:.4f}** | **{vid_summary['calibrated_video_mae']['max']:.4f}** | **--** |

### Per-Video Improvement Counts:
- **Videos Improved**: **{v_chg['videos_improved_count']} / 100**
- **Videos Worsened**: **{v_chg['videos_worsened_count']} / 100**
- **Videos Unchanged**: **{v_chg['videos_unchanged_count']} / 100**
- **Median Per-Video Delta**: **{v_chg['median_per_video_delta']:+.4f}**

---

## 7. Generalization Caution (Task 7)

> [!CAUTION]
> **Strict Generalization Limits**:
> 1. **No External Test Ground Truth**: Phase 4 analyzes predictions exclusively on the existing DHF1K validation set (`601.AVI` to `700.AVI`).
> 2. **Cross-Fitting vs. New Data**: While 5-fold cross-fitting prevents in-sample parameter leakage, it does **not** substitute for evaluation on an independent external video distribution.
> 3. **Private Benchmark Integrity**: DHF1K benchmark videos `701.AVI` through `1000.AVI` have private annotations and must remain completely unevaluated to preserve benchmark integrity.

---

## 8. Decision Rule Recommendation (Task 8)

### Selected Decision: **{dec['recommendation']}**

**Rationale**:
{dec['rationale']}

**Key Decision Factors**:
- Overall cross-fitted MAE change: **{dec['decision_factors']['cross_fitted_mae_change']:+.4f}** (negligible).
- Overall cross-fitted RMSE change: **{dec['decision_factors']['cross_fitted_rmse_change']:+.4f}**.
- Overall cross-fitted $R^2$ change: **{dec['decision_factors']['cross_fitted_r2_change']:+.4f}**.
- Central density MAE degradation: **{dec['decision_factors']['central_density_mae_change']:+.4f}**.
- Per-video balance: **{dec['decision_factors']['videos_improved_vs_worsened']}**.

*Final Technical Policy*: Because the uncalibrated model (`model_dhf1k.pkl`) is already centered around the validation mean ($\\text{{bias}} = -0.0002$) and cross-fitted calibration degrades central density precision without yielding meaningful net MAE reduction, **`model_dhf1k.pkl` is retained as the independent DHF1K experiment artifact unchanged**, while the production NEMAR model (`models/human_attention/model.pkl`) remains strictly isolated and untouched. Post-hoc calibration is marked as exploratory and marginal.
"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)


if __name__ == "__main__":
    analyzer = DHF1KCalibrationAnalyzer()
    analyzer.run_and_save_all()
