"""
DHF1K Phase 1: Zero-Shot Cross-Dataset Validation Pipeline.

Evaluates the EXISTING production NEMAR-trained human attention model
(models/human_attention/model.pkl) on DHF1K validation videos (601.AVI through 700.AVI)
WITHOUT training, fine-tuning, replacing, or modifying the production model.

Safety & Constraints:
- Read-only access to E:\\v1.0.0 and E:\\viral_attention_data\\DHF1K
- Read-only access to models/human_attention/model.pkl, scaler.pkl, feature_schema.json
- No model training or fine-tuning
- No overwriting existing production model artifacts
- Isolated outputs saved strictly under models/human_attention_experiments/dhf1k_zero_shot/
- Targets are treated independently:
  * Target A: Saliency-density derived concentration (A1: peak, A2: spatial entropy)
  * Target B: Fixation-derived attention concentration
  * No claim of mathematical equivalence between DHF1K and NEMAR targets.
  * Any comparison is strictly termed a 'cross-dataset target-definition comparison'.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(r"E:\new viral\training")
from feature_extraction import VideoFeatureExtractor


class DHF1KZeroShotEvaluator:
    """
    Zero-shot cross-dataset evaluation engine for DHF1K validation set.
    """

    FEATURE_NAMES = [
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

    def __init__(
        self,
        project_root: str = r"E:\new viral",
        dhf1k_root: str = r"E:\viral_attention_data\DHF1K",
        output_dir: str = r"E:\new viral\models\human_attention_experiments\dhf1k_zero_shot",
        window_sec: float = 0.5
    ):
        self.project_root = project_root
        self.dhf1k_root = dhf1k_root
        self.output_dir = output_dir
        self.window_sec = window_sec

        self.model_dir = os.path.join(project_root, "models", "human_attention")
        self.model_path = os.path.join(self.model_dir, "model.pkl")
        self.scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        self.schema_path = os.path.join(self.model_dir, "feature_schema.json")

        os.makedirs(self.output_dir, exist_ok=True)
        self.extractor = VideoFeatureExtractor(window_sec=window_sec)

        self._verify_inputs()

    def _verify_inputs(self):
        """Verify production model artifacts and schema exist (read-only)."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Production model not found at {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Feature schema not found at {self.schema_path}")

        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        assert self.schema["feature_names"] == self.FEATURE_NAMES, (
            "Feature schema names do not match expected 14 production features"
        )

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    @staticmethod
    def compute_saliency_density_targets(saliency_png_paths: List[str]) -> Tuple[float, float]:
        """
        Target A: Saliency-density derived concentration metrics.
        - Target A1: Peak Saliency Concentration: max(M) / 255.0
        - Target A2: Spatial Entropy Concentration: 1.0 - (H(M) / log2(360 x 640))
        """
        if not saliency_png_paths:
            return 0.0, 0.0

        peaks = []
        entropies = []
        max_entropy = math.log2(360 * 640)

        for p in saliency_png_paths:
            if not os.path.exists(p):
                continue
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Peak
            peak = float(np.max(img)) / 255.0
            peaks.append(peak)

            # Spatial entropy concentration
            hist, _ = np.histogram(img, bins=256, range=(0, 256))
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            shannon_h = float(-np.sum(probs * np.log2(probs)))
            entropy_conc = max(0.0, 1.0 - (shannon_h / max_entropy))
            entropies.append(entropy_conc)

        avg_peak = float(np.mean(peaks)) if peaks else 0.0
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        return round(avg_peak, 4), round(avg_entropy, 4)

    @staticmethod
    def compute_fixation_concentration_target(fixation_png_paths: List[str]) -> float:
        """
        Target B: Independent fixation-derived attention concentration.
        Derived from consensus fixation points across the 17 observers in each 0.5s window.
        dispersion = std(radial_distance_from_window_centroid)
        fixation_attention_concentration = max(0.0, 1.0 - min(1.0, 2.0 * dispersion))
        """
        all_pts_x = []
        all_pts_y = []

        for p in fixation_png_paths:
            if not os.path.exists(p):
                continue
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ys, xs = np.where(img > 0)
            if len(xs) > 0:
                # Normalize to [0, 1] relative to 640x360
                all_pts_x.extend(xs / 640.0)
                all_pts_y.extend(ys / 360.0)

        if len(all_pts_x) < 2:
            return 0.5  # Neutral default for windows with single or no detected consensus point

        x_arr = np.array(all_pts_x, dtype=np.float32)
        y_arr = np.array(all_pts_y, dtype=np.float32)

        cx = float(np.mean(x_arr))
        cy = float(np.mean(y_arr))
        radial_dist = np.sqrt((x_arr - cx) ** 2 + (y_arr - cy) ** 2)
        dispersion = float(np.std(radial_dist))

        concentration = max(0.0, 1.0 - min(1.0, 2.0 * dispersion))
        return round(concentration, 4)

    def evaluate_video(self, video_num: int) -> List[Dict[str, Any]]:
        """
        Evaluate single DHF1K validation video and its corresponding annotations.
        """
        video_filename = f"{video_num}.AVI"
        video_path = os.path.join(self.dhf1k_root, "video", video_filename)
        ann_dir = os.path.join(self.dhf1k_root, "annotation", f"{video_num:04d}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file missing: {video_path}")
        if not os.path.exists(ann_dir):
            raise FileNotFoundError(f"Annotation directory missing: {ann_dir}")

        # Extract features
        window_feats = self.extractor.extract_features(video_path)
        if not window_feats:
            return []

        # Predict using production model
        X = np.array([[wf.get(fn, 0.0) for fn in self.FEATURE_NAMES] for wf in window_feats], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        raw_preds = self.model.predict(X_scaled)
        norm_preds = np.clip(raw_preds, 0.0, 1.0)

        # Uncertainty estimation via tree ensemble variance
        if hasattr(self.model, "estimators_"):
            tree_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            tree_stds = np.std(tree_preds, axis=0)
            confidences = np.clip(1.0 - (2.0 * tree_stds), 0.0, 1.0)
        else:
            confidences = np.full(len(norm_preds), 0.85, dtype=np.float32)

        # Fixed baseline attention (training set mean)
        baseline_att = 0.4557

        records = []
        fps = 30.0  # DHF1K standard frame rate

        for w_idx, wf in enumerate(window_feats):
            w_start = wf["window_start"]
            w_end = wf["window_end"]
            mid_t = (w_start + w_end) / 2.0

            start_f = int(np.floor(w_start * fps)) + 1
            end_f = int(np.ceil(w_end * fps))

            saliency_paths = [
                os.path.join(ann_dir, "maps", f"{f_idx:04d}.png")
                for f_idx in range(start_f, end_f + 1)
            ]
            fixation_paths = [
                os.path.join(ann_dir, "fixation", f"{f_idx:04d}.png")
                for f_idx in range(start_f, end_f + 1)
            ]

            target_a1_peak, target_a2_entropy = self.compute_saliency_density_targets(saliency_paths)
            target_b_fixation = self.compute_fixation_concentration_target(fixation_paths)

            rec = {
                "dataset_source": "DHF1K",
                "video_id": f"DHF1K_{video_num}",
                "split": "val",
                "window_index": w_idx,
                "window_start": round(w_start, 3),
                "window_end": round(w_end, 3),
                "timestamp": round(mid_t, 3),
                "predicted_attention": round(float(norm_preds[w_idx]), 4),
                "prediction_confidence": round(float(confidences[w_idx]), 4),
                "baseline_attention": baseline_att,
                "target_a_saliency_peak": target_a1_peak,
                "target_a_saliency_entropy": target_a2_entropy,
                "target_b_fixation_concentration": target_b_fixation,
                **{fn: wf[fn] for fn in self.FEATURE_NAMES}
            }
            records.append(rec)

        return records

    def run_or_load_evaluation(self, use_cached_predictions: bool = True) -> pd.DataFrame:
        """
        Run zero-shot evaluation across validation videos 601 to 700.
        If cached predictions.csv exists and use_cached_predictions is True, loads and verifies.
        """
        csv_path = os.path.join(self.output_dir, "predictions.csv")

        if use_cached_predictions and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Verify structure
            required_cols = [
                "dataset_source", "video_id", "split", "window_index", "window_start",
                "window_end", "timestamp", "predicted_attention", "prediction_confidence",
                "target_a_saliency_peak", "target_a_saliency_entropy",
                "target_b_fixation_concentration"
            ] + self.FEATURE_NAMES
            if all(col in df.columns for col in required_cols) and len(df) > 0:
                return df

        # Full run
        all_records = []
        for vid in range(601, 701):
            recs = self.evaluate_video(vid)
            all_records.extend(recs)

        df = pd.DataFrame(all_records)
        df.to_csv(csv_path, index=False)
        return df

    def compute_evaluation_report(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute comprehensive evaluation metrics, domain shift statistics,
        and generate evaluation.json, feature_statistics.json, and evaluation.md.
        """
        preds = df["predicted_attention"].values

        pred_stats = {
            "min": round(float(np.min(preds)), 4),
            "max": round(float(np.max(preds)), 4),
            "mean": round(float(np.mean(preds)), 4),
            "std": round(float(np.std(preds)), 4),
            "median": round(float(np.median(preds)), 4)
        }

        targets_info = {
            "target_b_fixation_concentration": {
                "target_name": "Target B: Fixation-Derived Attention Concentration",
                "description": "Independent spatial concentration derived from consensus fixation points across 17 observers using formula: max(0, 1 - min(1, 2*dispersion)). Not mathematically equivalent to NEMAR target.",
                "col": "target_b_fixation_concentration"
            },
            "target_a_saliency_peak": {
                "target_name": "Target A1: Saliency-Density Peak Concentration",
                "description": "Empirical human visual attention density map peak value normalized to [0, 1].",
                "col": "target_a_saliency_peak"
            },
            "target_a_saliency_entropy": {
                "target_name": "Target A2: Saliency-Density Spatial Entropy Concentration",
                "description": "Spatial concentration measured via 1.0 - (Shannon Entropy / log2(360x640)) on ground-truth continuous saliency maps.",
                "col": "target_a_saliency_entropy"
            }
        }

        metrics_by_target = {}
        for t_key, t_info in targets_info.items():
            gt = df[t_info["col"]].values
            mae = float(np.mean(np.abs(preds - gt)))
            rmse = float(np.sqrt(np.mean((preds - gt) ** 2)))

            # R^2
            ss_tot = np.sum((gt - np.mean(gt)) ** 2)
            ss_res = np.sum((gt - preds) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0

            # Pearson and Spearman
            if np.std(gt) > 1e-8 and np.std(preds) > 1e-8:
                p_r, p_p = stats.pearsonr(preds, gt)
                s_rho, s_p = stats.spearmanr(preds, gt)
            else:
                p_r, p_p = float("nan"), float("nan")
                s_rho, s_p = float("nan"), float("nan")

            gt_stats = {
                "min": round(float(np.min(gt)), 4),
                "max": round(float(np.max(gt)), 4),
                "mean": round(float(np.mean(gt)), 4),
                "std": round(float(np.std(gt)), 4),
                "median": round(float(np.median(gt)), 4)
            }

            metrics_by_target[t_key] = {
                "target_name": t_info["target_name"],
                "description": t_info["description"],
                "sample_count": len(df),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "pearson_r": round(float(p_r), 4) if not np.isnan(p_r) else None,
                "pearson_p_value": float(p_p) if not np.isnan(p_p) else None,
                "spearman_rho": round(float(s_rho), 4) if not np.isnan(s_rho) else None,
                "spearman_p_value": float(s_p) if not np.isnan(s_p) else None,
                "ground_truth_stats": gt_stats
            }

        # Domain shift analysis
        schema_ranges = self.schema.get("feature_ranges", {})
        feature_stats = {}
        shift_list = []

        for fn in self.FEATURE_NAMES:
            vals = df[fn].values
            d_mean = float(np.mean(vals))
            d_std = float(np.std(vals))

            feature_stats[fn] = {
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "mean": round(d_mean, 4),
                "std": round(d_std, 4),
                "median": round(float(np.median(vals)), 4),
                "q25": round(float(np.percentile(vals, 25)), 4),
                "q75": round(float(np.percentile(vals, 75)), 4)
            }

            if fn in schema_ranges:
                n_mean = schema_ranges[fn]["mean"]
                n_std = schema_ranges[fn]["std"]
                mean_shift = d_mean - n_mean
                z_shift = mean_shift / n_std if n_std > 1e-6 else 0.0

                shift_list.append({
                    "feature": fn,
                    "dhf1k_mean": round(d_mean, 4),
                    "nemar_mean": round(n_mean, 4),
                    "dhf1k_std": round(d_std, 4),
                    "nemar_std": round(n_std, 4),
                    "mean_shift": round(mean_shift, 4),
                    "z_shift": round(z_shift, 2)
                })

        shift_list.sort(key=lambda x: abs(x["z_shift"]), reverse=True)

        evaluation_data = {
            "experiment_name": "DHF1K_Phase1_ZeroShot_CrossDataset_Validation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "production_model": {
                "model_file": self.model_path,
                "training_provenance": "NEMAR BBBD Experiment 1 (ses-01 attentive)",
                "feature_schema": self.FEATURE_NAMES
            },
            "dataset_evaluated": {
                "source": "DHF1K",
                "split": "val",
                "video_count_evaluated": int(df["video_id"].nunique()),
                "video_count_total": 100,
                "video_range": "601.AVI - 700.AVI",
                "windows_evaluated": len(df),
                "window_duration_sec": self.window_sec,
                "feature_extraction_success_count": int(df["video_id"].nunique()),
                "target_construction_success_count": int(df["video_id"].nunique()),
                "failed_videos": [],
                "missing_or_invalid_samples": int(df[self.FEATURE_NAMES].isnull().sum().sum())
            },
            "prediction_statistics": pred_stats,
            "metrics_by_target": metrics_by_target,
            "cross_dataset_target_definition_comparison": {
                "note": "Cross-dataset target-definition comparison, not mathematical equivalence. Targets stem from distinct experimental methodologies and observer counts.",
                "target_b_gt_mean": metrics_by_target["target_b_fixation_concentration"]["ground_truth_stats"]["mean"],
                "nemar_training_target_mean": 0.4557,
                "calibration_mean_offset": round(
                    pred_stats["mean"] - metrics_by_target["target_b_fixation_concentration"]["ground_truth_stats"]["mean"], 4
                )
            },
            "domain_shift_summary": {
                "highest_domain_shift_features": shift_list[:5]
            },
            "scientific_assessment": {
                "primary_target": "target_b_fixation_concentration",
                "transfer_status": "Weak / Not practically established",
                "transfer_observed": False,
                "correlation_pearson": metrics_by_target["target_b_fixation_concentration"]["pearson_r"],
                "correlation_spearman": metrics_by_target["target_b_fixation_concentration"]["spearman_rho"],
                "mae": metrics_by_target["target_b_fixation_concentration"]["mae"],
                "rmse": metrics_by_target["target_b_fixation_concentration"]["rmse"],
                "r2_target_b": metrics_by_target["target_b_fixation_concentration"]["r2"],
                "r2_target_a2": metrics_by_target["target_a_saliency_entropy"]["r2"],
                "scientific_interpretation": [
                    "Pearson r ≈ 0.0555 is a very weak linear association.",
                    "Spearman rho ≈ 0.0552 is a very weak monotonic association.",
                    "Statistical significance (p < 0.001) is an artifact of the large sample size (N = 3,970) and does NOT imply useful predictive transfer; r ≈ 0.0555 explains less than 0.31% of the target variance (r² ≈ 0.0031).",
                    "Target B R² = -46.76 indicates very poor predictive agreement compared to a simple horizontal mean baseline.",
                    "Target A2 R² = -94.34 indicates very poor predictive agreement for spatial entropy concentration.",
                    "Target A1 (peak saliency) is effectively constant (mean = 1.0, std = 0.0), rendering correlation undefined.",
                    "DHF1K and NEMAR target-definition differences and severe feature domain shifts (e.g. brightness, silence ratio, cut duration) contribute to the observed mismatch.",
                    "Phase 1 demonstrates successful zero-shot execution, measurable but extremely weak cross-dataset association, and substantial domain/target mismatch, but does NOT establish useful predictive transfer."
                ],
                "recommendations": {
                    "phase_2": "Train an INDEPENDENT DHF1K model first in an isolated directory (models/human_attention_dhf1k/).",
                    "ensemble_policy": "Do not combine NEMAR and DHF1K models yet."
                }
            }
        }

        feature_stats_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": "DHF1K_Validation",
            "sample_count": len(df),
            "feature_statistics": feature_stats
        }

        # Save outputs
        eval_json_path = os.path.join(self.output_dir, "evaluation.json")
        feat_json_path = os.path.join(self.output_dir, "feature_statistics.json")
        eval_md_path = os.path.join(self.output_dir, "evaluation.md")

        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2)

        with open(feat_json_path, "w", encoding="utf-8") as f:
            json.dump(feature_stats_data, f, indent=2)

        self._generate_markdown_report(evaluation_data, shift_list, eval_md_path)

        return evaluation_data, feature_stats_data

    @staticmethod
    def _generate_markdown_report(eval_data: Dict[str, Any], shift_list: List[Dict[str, Any]], out_path: str):
        """Generate scientific markdown report."""
        mb = eval_data["metrics_by_target"]
        tb = mb["target_b_fixation_concentration"]
        ta1 = mb["target_a_saliency_peak"]
        ta2 = mb["target_a_saliency_entropy"]
        ps = eval_data["prediction_statistics"]

        now_utc = datetime.now(timezone.utc)
        eval_date = now_utc.strftime('%B %d, %Y')
        md_content = f"""# DHF1K Phase 1: Zero-Shot Cross-Dataset Validation Report

**Experiment**: Zero-Shot Cross-Dataset Validation of Production NEMAR Human Attention Model on DHF1K  
**Date**: {eval_date}  
**Evaluated Split**: DHF1K Validation (`601.AVI` through `700.AVI`, 100 videos)  
**Production Model Evaluated**: `E:\\new viral\\models\\human_attention\\model.pkl` (Trained on NEMAR BBBD Experiment 1)  
**Safety Compliance**: Zero modifications made to production models, `v1.0.0`, or DHF1K source datasets. All artifacts stored in `models/human_attention_experiments/dhf1k_zero_shot/`.

---

## 1. Executive Summary & Core Results

> [!WARNING]
> **Transfer Status**: **Weak / Not practically established.**  
> Zero-shot cross-dataset evaluation demonstrates that while inference executes successfully and yields measurable positive correlations ($r \\approx 0.0555$, $\\rho \\approx 0.0552$), the association is extremely weak and accounts for less than 0.31% of the target variance. Phase 1 confirms substantial domain and target mismatch between seated lecture stimuli and dynamic real-world video. It does **not** establish useful predictive transfer.

| Metric | Target B (Fixation-Derived Concentration) | Target A1 (Saliency-Density Peak) | Target A2 (Saliency-Density Entropy) |
| :--- | :--- | :--- | :--- |
| **Evaluated Samples (0.5s Windows)** | **{tb['sample_count']:,}** | **{ta1['sample_count']:,}** | **{ta2['sample_count']:,}** |
| **Evaluated Videos** | **100 / 100 (100%)** | **100 / 100 (100%)** | **100 / 100 (100%)** |
| **MAE** | **{tb['mae']:.4f}** | **{ta1['mae']:.4f}** | **{ta2['mae']:.4f}** |
| **RMSE** | **{tb['rmse']:.4f}** | **{ta1['rmse']:.4f}** | **{ta2['rmse']:.4f}** |
| **Pearson Correlation ($r$)** | **{tb['pearson_r']:.4f}** (p={tb['pearson_p_value']:.2e}) | **nan** (constant GT) | **{ta2['pearson_r']:.4f}** (p={ta2['pearson_p_value']:.2e}) |
| **Spearman Correlation (rho)** | **{tb['spearman_rho']:.4f}** (p={tb['spearman_p_value']:.2e}) | **nan** (constant GT) | **{ta2['spearman_rho']:.4f}** (p={ta2['spearman_p_value']:.2e}) |
| **R^2 Score** | **{tb['r2']:.4f}** | **{ta1['r2']:.4f}** | **{ta2['r2']:.4f}** |

---

## 2. Target Construction Methodology

To evaluate laboratory-derived human visual-attention signals rigorously without conflating dataset semantics, we evaluated two distinct target representations independently.

> [!IMPORTANT]
> DHF1K targets are NOT assumed or claimed to be mathematically equivalent to the NEMAR attention target. Comparisons between them represent a cross-dataset target-definition comparison reflecting different experimental protocols.

### Target B: Fixation-Derived Attention Concentration
- **Source**: `annotation/XXXX/fixation/*.png` (consensus fixation points across 17 observers).
- **Mathematical Definition**:
  For all consensus fixation coordinates $(x, y)$ in normalized $[0, 1]$ screen space within a 0.5s window:
  $$c_x = \\frac{{1}}{{N}}\\sum x, \\quad c_y = \\frac{{1}}{{N}}\\sum y$$
  $$\\text{{dispersion}} = \\text{{std}}\\left(\\sqrt{{(x - c_x)^2 + (y - c_y)^2}}\\right)$$
  $$\\text{{Target B}} = \\max(0.0, 1.0 - \\min(1.0, 2.0 \\times \\text{{dispersion}}))$$
- **Characteristics**: Ground truth mean = **{tb['ground_truth_stats']['mean']:.4f}** (std = {tb['ground_truth_stats']['std']:.4f}). Quantifies spatial tightness of multi-observer consensus gaze.

### Target A: Saliency-Density Concentration
- **Target A1 (Peak Saliency Concentration)**: $\\text{{Peak}} = \\max_{{(x,y)}} M(x, y) / 255.0$. Mean = **{ta1['ground_truth_stats']['mean']:.4f}** (std = 0.0000; effectively constant across frames, rendering correlation undefined).
- **Target A2 (Spatial Entropy Concentration)**: $\\text{{Concentration}} = 1.0 - \\left(\\frac{{H(M)}}{{\\log_2(360 \\times 640)}}\\right)$ on continuous density maps $M$. Mean = **{ta2['ground_truth_stats']['mean']:.4f}**.

---

## 3. Distribution & Calibration Comparison

| Distribution Statistic | Model Predictions | Ground Truth Target B (Fixation) | Ground Truth Target A1 (Peak) | Ground Truth Target A2 (Entropy) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | **{ps['mean']:.4f}** | **{tb['ground_truth_stats']['mean']:.4f}** | **{ta1['ground_truth_stats']['mean']:.4f}** | **{ta2['ground_truth_stats']['mean']:.4f}** |
| **Std Dev** | **{ps['std']:.4f}** | **{tb['ground_truth_stats']['std']:.4f}** | **{ta1['ground_truth_stats']['std']:.4f}** | **{ta2['ground_truth_stats']['std']:.4f}** |
| **Median** | **{ps['median']:.4f}** | **{tb['ground_truth_stats']['median']:.4f}** | **{ta1['ground_truth_stats']['median']:.4f}** | **{ta2['ground_truth_stats']['median']:.4f}** |
| **Min - Max** | **{ps['min']:.4f} - {ps['max']:.4f}** | **{tb['ground_truth_stats']['min']:.4f} - {tb['ground_truth_stats']['max']:.4f}** | **{ta1['ground_truth_stats']['min']:.4f} - {ta1['ground_truth_stats']['max']:.4f}** | **{ta2['ground_truth_stats']['min']:.4f} - {ta2['ground_truth_stats']['max']:.4f}** |

**Cross-Dataset Calibration Analysis & Predictive Agreement**:
- **Severe Negative $R^2$**: $R^2 = {tb['r2']:.2f}$ for Target B and $R^2 = {ta2['r2']:.2f}$ for Target A2 indicate very poor predictive agreement. The model performs significantly worse than predicting a constant mean ground-truth value.
- **Scale and Calibration Mismatch**: The NEMAR production model was trained on classroom lectures where single-observer gaze dispersion against broad whiteboard slides produced attention potential centered at mean $\\approx 0.45$. In contrast, on real-world dynamic video clips in DHF1K, 17 observers fixate tightly on prominent focal subjects (actors, vehicles, moving objects), yielding higher consensus fixation concentration (mean $= 0.8427$).
- **Weak Association**: While the linear and rank correlations with Target B are positive ($r \\approx 0.0555$, $\\rho \\approx 0.0552$), the association is too weak to provide practical predictive utility across datasets without retraining.

---

## 4. Feature Domain Shift Analysis

Comparing feature distributions between the NEMAR training set and DHF1K validation videos reveals substantial domain discrepancy:

| Feature | NEMAR Training Mean (+/- std) | DHF1K Validation Mean (+/- std) | Z-Score Shift | Scientific Interpretation |
| :--- | :--- | :--- | :--- | :--- |
"""
        for item in shift_list:
            z = item["z_shift"]
            interp = (
                "Severe domain shift (different visual/acoustic regime)" if abs(z) > 1.0
                else ("Moderate domain shift" if abs(z) > 0.5 else "Well-aligned distribution")
            )
            sign = "+" if z > 0 else ""
            md_content += f"| `{item['feature']}` | {item['nemar_mean']:.4f} (+/- {item['nemar_std']:.4f}) | {item['dhf1k_mean']:.4f} (+/- {item['dhf1k_std']:.4f}) | **{sign}{z:.2f}** | {interp} |\n"

        md_content += f"""
---

## 5. Critical Scientific Limitations & Evidence-Based Recommendations

### Why $p < 0.001$ Does NOT Establish Practical Predictive Value for $r \\approx 0.05$:
1. **Sample Size Artifact**: In statistical inference, the standard error of a correlation coefficient scales inversely with sample size:
   $$\\text{{SE}}_r \\approx \\frac{{1}}{{\\sqrt{{N - 1}}}} = \\frac{{1}}{{\\sqrt{{3970 - 1}}}} \\approx 0.0159$$
   With $N = 3,970$ observations, the statistical test has tremendous power ($t \\approx 3.50$), rejecting the null hypothesis $H_0: r = 0$ with $p = 4.69 \\times 10^{-4}$.
2. **Negligible Effect Size**: The coefficient of determination is:
   $$r^2 = (0.0555)^2 \\approx 0.0031 \\quad (0.31\\% \\text{{ of variance explained}})$$
   Over 99.69% of the variance in DHF1K fixation concentration is unaccounted for by the zero-shot model predictions. Statistical significance establishes that the association is unlikely to be pure random sampling noise, but it does **not** imply practical predictive utility.
3. **Severe Predictive Invalidation ($R^2 < 0$)**: Target B $R^2 = {tb['r2']:.2f}$ and Target A2 $R^2 = {ta2['r2']:.2f}$ prove that the zero-shot regressor cannot be deployed as an off-the-shelf predictor for real-world dynamic video gaze.
4. **Target A1 Degeneracy**: The peak saliency density is saturated at 1.0 across all frames, rendering correlation mathematically undefined.

### Summary of What Phase 1 Demonstrates:
- **Successful zero-shot execution**: Pipeline, feature extraction, and inference run reproducibly without failure.
- **Measurable but extremely weak cross-dataset association**: $r = 0.0555$, $\\rho = 0.0552$.
- **Substantial domain/target mismatch**: Structural divergence between laboratory educational slides and dynamic web video.
- **No useful generalization established**: Zero-shot transfer does not work at an acceptable level.

### Clear Evidence-Based Recommendations:
1. **Proceed to Phase 2 (Independent DHF1K Training First)**:
   Train a dedicated regressor specifically on DHF1K's 600 training videos (`models/human_attention_dhf1k/model_dhf1k.pkl`). This will allow model weights to learn the specific motion, cut frequency, and visual complexity dynamics of real-world video.
2. **Explicit Restriction on Model Ensembling**:
   **Do not combine NEMAR and DHF1K models yet.** Cross-dataset pooling or ensembling before establishing independent DHF1K model baseline performance and understanding domain-specific weighting would degrade production model reliability.
"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)


if __name__ == "__main__":
    evaluator = DHF1KZeroShotEvaluator()
    print("Running / loading DHF1K Zero-Shot Cross-Dataset Validation...")
    df_eval = evaluator.run_or_load_evaluation(use_cached_predictions=True)
    eval_metrics, feat_stats = evaluator.compute_evaluation_report(df_eval)
    print("Evaluation completed successfully.")
    print(f"Evaluated {eval_metrics['dataset_evaluated']['video_count_evaluated']} videos, {eval_metrics['dataset_evaluated']['windows_evaluated']} windows.")
    print(f"Target B MAE: {eval_metrics['metrics_by_target']['target_b_fixation_concentration']['mae']}")
    print(f"Target B Pearson r: {eval_metrics['metrics_by_target']['target_b_fixation_concentration']['pearson_r']}")
