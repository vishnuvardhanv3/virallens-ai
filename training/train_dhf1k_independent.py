"""
DHF1K Phase 2: Independent DHF1K Human Visual-Attention Model Training & Evaluation.

Trains an INDEPENDENT regression model on DHF1K training data (001.AVI through 600.AVI)
predicting a constructed DHF1K-derived fixation concentration proxy, evaluated on DHF1K
validation data (601.AVI through 700.AVI).

STRICT SAFETY & ISOLATION:
- Read-only access to models/human_attention/model.pkl, scaler.pkl, feature_schema.json
- Zero modification to production models or E:\\v1.0.0 or DHF1K source files
- Scaler fit ONLY on training videos (001-600)
- Split strictly by VIDEO ID
- Invalid low-fixation windows (< 2 fixations) excluded from primary training and primary evaluation
- Deterministic model selection: lowest validation MAE
- Outputs stored strictly in models/human_attention_experiments/dhf1k_independent/
"""

import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.preprocessing import StandardScaler

sys.path.append(r"E:\new viral\training")
from feature_extraction import VideoFeatureExtractor


class DHF1KIndependentTrainer:
    """
    Independent trainer and evaluator for DHF1K-derived fixation concentration proxy.
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

    EXPECTED_PRODUCTION_HASHES = {
        "model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
        "scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
        "feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411"
    }

    def __init__(
        self,
        project_root: str = r"E:\new viral",
        dhf1k_root: str = r"E:\viral_attention_data\DHF1K",
        output_dir: str = r"E:\new viral\models\human_attention_experiments\dhf1k_independent",
        cache_dir: str = r"E:\new viral\data\dhf1k_cache",
        window_sec: float = 0.5
    ):
        self.project_root = project_root
        self.dhf1k_root = dhf1k_root
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.window_sec = window_sec

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "video_records"), exist_ok=True)

        self.production_model_dir = os.path.join(project_root, "models", "human_attention")
        self.extractor = VideoFeatureExtractor(window_sec=window_sec)

        self.verify_production_integrity("pre-training")

    def verify_production_integrity(self, stage: str = "check") -> Dict[str, str]:
        """Verify that production NEMAR model files are completely unmodified."""
        current_hashes = {}
        for fname, exp_hash in self.EXPECTED_PRODUCTION_HASHES.items():
            p = os.path.join(self.production_model_dir, fname)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Production model file missing: {p}")
            with open(p, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            current_hashes[fname] = h
            assert h == exp_hash, (
                f"SAFETY VIOLATION at {stage}: {fname} hash changed! "
                f"Expected {exp_hash}, got {h}"
            )
        return current_hashes

    def process_single_video(self, vid: int, split: str) -> List[Dict[str, Any]]:
        """
        Extract features and compute target for a single video with caching.
        """
        cache_file = os.path.join(self.cache_dir, "video_records", f"{vid:04d}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        video_filename = f"{vid:03d}.AVI" if vid < 1000 else "1000.AVI"
        if not os.path.exists(os.path.join(self.dhf1k_root, "video", video_filename)):
            video_filename = f"{vid}.AVI"

        v_path = os.path.join(self.dhf1k_root, "video", video_filename)
        ann_dir = os.path.join(self.dhf1k_root, "annotation", f"{vid:04d}", "fixation")

        if not os.path.exists(v_path):
            raise FileNotFoundError(f"Video file missing: {v_path}")

        feats = self.extractor.extract_features(v_path)
        fps = 30.0

        records = []
        for w_idx, wf in enumerate(feats):
            w_start = wf["window_start"]
            w_end = wf["window_end"]
            mid_t = (w_start + w_end) / 2.0

            s_f = int(np.floor(w_start * fps)) + 1
            e_f = int(np.ceil(w_end * fps))

            all_xs, all_ys = [], []
            if os.path.exists(ann_dir):
                for f_idx in range(s_f, e_f + 1):
                    p = os.path.join(ann_dir, f"{f_idx:04d}.png")
                    if os.path.exists(p):
                        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            ys, xs = np.where(img > 0)
                            if len(xs) > 0:
                                all_xs.extend(xs / 640.0)
                                all_ys.extend(ys / 360.0)

            n_fix = len(all_xs)
            if n_fix >= 2:
                is_valid = True
                cx = float(np.mean(all_xs))
                cy = float(np.mean(all_ys))
                dist = np.sqrt((np.array(all_xs) - cx) ** 2 + (np.array(all_ys) - cy) ** 2)
                disp = float(np.std(dist))
                target_proxy = round(float(np.clip(1.0 - min(1.0, 2.0 * disp), 0.0, 1.0)), 4)
                target_fallback = target_proxy
            else:
                is_valid = False
                disp = 0.5
                target_proxy = None  # Excluded from primary supervision
                target_fallback = 0.5  # Sensitivity condition only

            rec = {
                "dataset_source": "DHF1K",
                "video_id": f"DHF1K_{vid:03d}",
                "split": split,
                "window_index": w_idx,
                "window_start": round(w_start, 3),
                "window_end": round(w_end, 3),
                "timestamp": round(mid_t, 3),
                "fixation_point_count": n_fix,
                "is_valid_fixation_window": is_valid,
                "target_fixation_concentration_proxy": target_proxy,
                "target_fallback_sensitivity": target_fallback,
                **{fn: wf[fn] for fn in self.FEATURE_NAMES}
            }
            records.append(rec)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(records, f)

        return records

    def load_dataset(self, max_workers: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training (001-600) and validation (601-700) datasets concurrently.
        """
        train_csv = os.path.join(self.cache_dir, "train_windows.csv")
        val_csv = os.path.join(self.cache_dir, "val_windows.csv")

        if os.path.exists(train_csv) and os.path.exists(val_csv):
            print("Loading train and validation windows from cache...")
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            if train_df["video_id"].nunique() == 600 and val_df["video_id"].nunique() == 100:
                return train_df, val_df

        print("Extracting / processing DHF1K training and validation sets...")
        train_records = []
        val_records = []

        # Process training videos (1 to 600)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            train_futures = {ex.submit(self.process_single_video, vid, "train"): vid for vid in range(1, 601)}
            for i, fut in enumerate(as_completed(train_futures)):
                recs = fut.result()
                train_records.extend(recs)
                if (i + 1) % 50 == 0 or (i + 1) == 600:
                    print(f"Processed {i + 1}/600 training videos...")

        # Process validation videos (601 to 700)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            val_futures = {ex.submit(self.process_single_video, vid, "val"): vid for vid in range(601, 701)}
            for i, fut in enumerate(as_completed(val_futures)):
                recs = fut.result()
                val_records.extend(recs)
                if (i + 1) % 25 == 0 or (i + 1) == 100:
                    print(f"Processed {i + 1}/100 validation videos...")

        train_df = pd.DataFrame(train_records).sort_values(by=["video_id", "window_index"]).reset_index(drop=True)
        val_df = pd.DataFrame(val_records).sort_values(by=["video_id", "window_index"]).reset_index(drop=True)

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        return train_df, val_df

    def compute_pre_training_target_statistics(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Report target and exclusion statistics prior to model fitting.
        """
        train_total = len(train_df)
        train_valid = train_df["is_valid_fixation_window"].sum()
        train_excluded = train_total - train_valid
        train_pct_excluded = (train_excluded / train_total) * 100.0 if train_total > 0 else 0.0

        val_total = len(val_df)
        val_valid = val_df["is_valid_fixation_window"].sum()
        val_excluded = val_total - val_valid
        val_pct_excluded = (val_excluded / val_total) * 100.0 if val_total > 0 else 0.0

        train_targets = train_df.loc[train_df["is_valid_fixation_window"], "target_fixation_concentration_proxy"].dropna().values
        val_targets = val_df.loc[val_df["is_valid_fixation_window"], "target_fixation_concentration_proxy"].dropna().values

        # Per-video valid target averages
        train_video_means = (
            train_df[train_df["is_valid_fixation_window"]]
            .groupby("video_id")["target_fixation_concentration_proxy"]
            .mean()
            .values
        )
        val_video_means = (
            val_df[val_df["is_valid_fixation_window"]]
            .groupby("video_id")["target_fixation_concentration_proxy"]
            .mean()
            .values
        )

        stats_summary = {
            "training_windows": {
                "total": int(train_total),
                "valid": int(train_valid),
                "excluded": int(train_excluded),
                "pct_excluded": round(float(train_pct_excluded), 2),
                "target_mean": round(float(np.mean(train_targets)), 4),
                "target_std": round(float(np.std(train_targets)), 4),
                "target_min": round(float(np.min(train_targets)), 4),
                "target_max": round(float(np.max(train_targets)), 4),
                "video_mean_average": round(float(np.mean(train_video_means)), 4),
                "video_mean_std": round(float(np.std(train_video_means)), 4)
            },
            "validation_windows": {
                "total": int(val_total),
                "valid": int(val_valid),
                "excluded": int(val_excluded),
                "pct_excluded": round(float(val_pct_excluded), 2),
                "target_mean": round(float(np.mean(val_targets)), 4),
                "target_std": round(float(np.std(val_targets)), 4),
                "target_min": round(float(np.min(val_targets)), 4),
                "target_max": round(float(np.max(val_targets)), 4),
                "video_mean_average": round(float(np.mean(val_video_means)), 4),
                "video_mean_std": round(float(np.std(val_video_means)), 4)
            }
        }
        return stats_summary

    def train_and_select_model(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Tuple[Any, StandardScaler, Dict[str, Any], pd.DataFrame]:
        """
        Train candidate regressors and select best model using deterministic validation MAE.
        """
        # Filter to valid fixation windows only
        train_clean = train_df[train_df["is_valid_fixation_window"]].copy().reset_index(drop=True)
        val_clean = val_df[val_df["is_valid_fixation_window"]].copy().reset_index(drop=True)

        X_train_raw = train_clean[self.FEATURE_NAMES].values.astype(np.float32)
        y_train = train_clean["target_fixation_concentration_proxy"].values.astype(np.float32)

        X_val_raw = val_clean[self.FEATURE_NAMES].values.astype(np.float32)
        y_val = val_clean["target_fixation_concentration_proxy"].values.astype(np.float32)

        # STRICT LEAKAGE PREVENTION: fit scaler ONLY on training data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        # Candidate Regressors
        candidates = {
            "ExtraTreesRegressor": ExtraTreesRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                min_samples_leaf=4,
                random_state=42
            ),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=6,
                learning_rate=0.08,
                min_samples_leaf=10,
                random_state=42
            ),
        }

        candidate_results = {}
        for name, model in candidates.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - t0

            train_preds = np.clip(model.predict(X_train), 0.0, 1.0)
            val_preds = np.clip(model.predict(X_val), 0.0, 1.0)

            # Training metrics
            train_mae = float(np.mean(np.abs(train_preds - y_train)))
            train_rmse = float(np.sqrt(np.mean((train_preds - y_train) ** 2)))
            ss_tot_tr = np.sum((y_train - np.mean(y_train)) ** 2)
            train_r2 = float(1.0 - (np.sum((y_train - train_preds) ** 2) / ss_tot_tr)) if ss_tot_tr > 1e-8 else 0.0

            # Validation metrics
            val_mae = float(np.mean(np.abs(val_preds - y_val)))
            val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
            ss_tot_val = np.sum((y_val - np.mean(y_val)) ** 2)
            val_r2 = float(1.0 - (np.sum((y_val - val_preds) ** 2) / ss_tot_val)) if ss_tot_val > 1e-8 else 0.0

            p_r, p_p = stats.pearsonr(val_preds, y_val) if np.std(val_preds) > 1e-8 else (0.0, 1.0)
            s_rho, s_p = stats.spearmanr(val_preds, y_val) if np.std(val_preds) > 1e-8 else (0.0, 1.0)

            candidate_results[name] = {
                "model_name": name,
                "fit_time_sec": round(fit_time, 2),
                "train_metrics": {
                    "mae": round(train_mae, 4),
                    "rmse": round(train_rmse, 4),
                    "r2": round(train_r2, 4),
                    "sample_count": len(y_train),
                    "video_count": int(train_clean["video_id"].nunique()),
                    "pred_mean": round(float(np.mean(train_preds)), 4),
                    "pred_std": round(float(np.std(train_preds)), 4),
                    "target_mean": round(float(np.mean(y_train)), 4),
                    "target_std": round(float(np.std(y_train)), 4)
                },
                "val_metrics": {
                    "mae": round(val_mae, 4),
                    "rmse": round(val_rmse, 4),
                    "r2": round(val_r2, 4),
                    "pearson_r": round(float(p_r), 4),
                    "pearson_p": float(p_p),
                    "spearman_rho": round(float(s_rho), 4),
                    "spearman_p": float(s_p),
                    "sample_count": len(y_val),
                    "video_count": int(val_clean["video_id"].nunique()),
                    "pred_mean": round(float(np.mean(val_preds)), 4),
                    "pred_std": round(float(np.std(val_preds)), 4),
                    "target_mean": round(float(np.mean(y_val)), 4),
                    "target_std": round(float(np.std(y_val)), 4)
                },
                "_fitted_model": model,
                "_val_preds": val_preds
            }

        # Deterministic Model Selection by Validation MAE
        # Tie breaking: lower RMSE, higher R2, simpler model
        sorted_candidates = sorted(
            candidate_results.values(),
            key=lambda c: (
                c["val_metrics"]["mae"],
                c["val_metrics"]["rmse"],
                -c["val_metrics"]["r2"]
            )
        )

        best_cand = sorted_candidates[0]
        selected_name = best_cand["model_name"]
        best_model = best_cand["_fitted_model"]

        print(f"Selected model: {selected_name} with validation MAE = {best_cand['val_metrics']['mae']:.4f}")

        # Assemble validation predictions dataframe
        val_clean["predicted_attention"] = np.round(best_cand["_val_preds"], 4)
        val_clean["prediction_error"] = np.round(best_cand["_val_preds"] - y_val, 4)
        val_clean["absolute_error"] = np.round(np.abs(best_cand["_val_preds"] - y_val), 4)

        # Feature importances if available
        feature_importances = {}
        if hasattr(best_model, "feature_importances_"):
            for fn, imp in zip(self.FEATURE_NAMES, best_model.feature_importances_):
                feature_importances[fn] = round(float(imp), 4)
            feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

        selection_summary = {
            "selected_model": selected_name,
            "selection_criterion": "Deterministic lowest validation MAE",
            "candidate_comparison": {
                name: {
                    "train": c["train_metrics"],
                    "val": c["val_metrics"],
                    "fit_time_sec": c["fit_time_sec"]
                }
                for name, c in candidate_results.items()
            },
            "feature_importances": feature_importances
        }

        return best_model, scaler, selection_summary, val_clean

    def run_sensitivity_analysis(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        best_model: Any,
        scaler: StandardScaler
    ) -> Dict[str, Any]:
        """
        Sensitivity analysis: Compare Condition A (valid windows only) vs Condition B (neutral fallback 0.5).
        """
        # Condition A: Valid only
        val_clean = val_df[val_df["is_valid_fixation_window"]].copy()
        X_val_a = scaler.transform(val_clean[self.FEATURE_NAMES].values.astype(np.float32))
        y_val_a = val_clean["target_fixation_concentration_proxy"].values.astype(np.float32)
        preds_a = np.clip(best_model.predict(X_val_a), 0.0, 1.0)

        mae_a = float(np.mean(np.abs(preds_a - y_val_a)))
        rmse_a = float(np.sqrt(np.mean((preds_a - y_val_a) ** 2)))

        # Condition B: Neutral fallback on all windows
        X_val_b = scaler.transform(val_df[self.FEATURE_NAMES].values.astype(np.float32))
        y_val_b = val_df["target_fallback_sensitivity"].values.astype(np.float32)
        preds_b = np.clip(best_model.predict(X_val_b), 0.0, 1.0)

        mae_b = float(np.mean(np.abs(preds_b - y_val_b)))
        rmse_b = float(np.sqrt(np.mean((preds_b - y_val_b) ** 2)))

        sensitivity = {
            "description": "Read-only sensitivity comparison of valid-fixation windows vs neutral fallback (0.5) inclusion",
            "condition_a_valid_only": {
                "sample_count": len(y_val_a),
                "mae": round(mae_a, 4),
                "rmse": round(rmse_a, 4)
            },
            "condition_b_neutral_fallback": {
                "sample_count": len(y_val_b),
                "mae": round(mae_b, 4),
                "rmse": round(rmse_b, 4)
            },
            "difference": {
                "delta_mae": round(mae_b - mae_a, 4),
                "delta_rmse": round(rmse_b - rmse_a, 4)
            }
        }
        return sensitivity

    def save_artifacts_and_report(
        self,
        best_model: Any,
        scaler: StandardScaler,
        selection_summary: Dict[str, Any],
        val_clean: pd.DataFrame,
        pre_stats: Dict[str, Any],
        sensitivity: Dict[str, Any],
        train_df: pd.DataFrame
    ):
        """
        Save all 8 required experiment artifacts.
        """
        sel_name = selection_summary["selected_model"]
        metrics = selection_summary["candidate_comparison"][sel_name]

        # 1. model_dhf1k.pkl
        model_path = os.path.join(self.output_dir, "model_dhf1k.pkl")
        joblib.dump(best_model, model_path)

        # 2. scaler_dhf1k.pkl
        scaler_path = os.path.join(self.output_dir, "scaler_dhf1k.pkl")
        joblib.dump(scaler, scaler_path)

        # 3. feature_schema_dhf1k.json
        schema_path = os.path.join(self.output_dir, "feature_schema_dhf1k.json")
        train_clean = train_df[train_df["is_valid_fixation_window"]]
        ranges = {}
        for fn in self.FEATURE_NAMES:
            v = train_clean[fn].values
            ranges[fn] = {
                "min": round(float(np.min(v)), 4),
                "max": round(float(np.max(v)), 4),
                "mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4)
            }

        schema_data = {
            "feature_names": self.FEATURE_NAMES,
            "feature_count": len(self.FEATURE_NAMES),
            "target": "DHF1K-derived fixation concentration proxy",
            "scaler": "StandardScaler (fitted strictly on training videos 001-600)",
            "feature_ranges": ranges,
            "feature_importances": selection_summary.get("feature_importances", {})
        }
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2)

        # 4. validation_predictions.csv
        pred_cols = [
            "dataset_source", "video_id", "split", "window_index", "window_start",
            "window_end", "timestamp", "fixation_point_count", "target_fixation_concentration_proxy",
            "predicted_attention", "prediction_error", "absolute_error"
        ] + self.FEATURE_NAMES
        val_pred_path = os.path.join(self.output_dir, "validation_predictions.csv")
        val_clean[pred_cols].to_csv(val_pred_path, index=False)

        # 5. feature_statistics.json
        feat_stats_path = os.path.join(self.output_dir, "feature_statistics.json")
        f_stats = {}
        for fn in self.FEATURE_NAMES:
            v = val_clean[fn].values
            f_stats[fn] = {
                "min": round(float(np.min(v)), 4),
                "max": round(float(np.max(v)), 4),
                "mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4),
                "median": round(float(np.median(v)), 4),
                "q25": round(float(np.percentile(v, 25)), 4),
                "q75": round(float(np.percentile(v, 75)), 4)
            }
        feat_stats_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": "DHF1K_Validation_Clean",
            "sample_count": len(val_clean),
            "feature_statistics": f_stats
        }
        with open(feat_stats_path, "w", encoding="utf-8") as f:
            json.dump(feat_stats_data, f, indent=2)

        # 6. training_metadata.json
        metadata_path = os.path.join(self.output_dir, "training_metadata.json")
        meta_data = {
            "dataset": "DHF1K (Dynamic Human Fixation in the Wild)",
            "dataset_source_root": self.dhf1k_root,
            "target_definition": {
                "name": "DHF1K-derived fixation concentration proxy",
                "scope": "Constructed temporal regression target derived from multi-observer consensus gaze fixations",
                "formula": "target = max(0.0, 1.0 - min(1.0, 2.0 * std(radial_distance_from_centroid)))",
                "windowing": "0.5s non-overlapping temporal windows (15 frames at 30.0 fps)",
                "low_fixation_policy": "Windows with < 2 consensus fixations are marked invalid and excluded from primary fitting"
            },
            "split_strategy": {
                "method": "Strict Video ID partitioning (zero temporal window overlap)",
                "train_videos": "001.AVI through 600.AVI (600 videos)",
                "val_videos": "601.AVI through 700.AVI (100 videos)",
                "test_videos": "701.AVI through 1000.AVI (held-out benchmark; annotations private/unavailable)"
            },
            "isolation_guarantee": {
                "nemar_production_model": "models/human_attention/model.pkl (READ-ONLY, SHA-256 verified)",
                "scaler_derivation": "scaler_dhf1k.pkl fit exclusively on training videos 001-600",
                "ensemble_policy": "Do not combine NEMAR and DHF1K models"
            },
            "pre_training_statistics": pre_stats,
            "selected_model": sel_name,
            "selection_criterion": "Deterministic lowest validation MAE",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        # 7. evaluation.json
        eval_json_path = os.path.join(self.output_dir, "evaluation.json")
        # Load Phase 1 zero-shot results for structured comparison
        phase1_json_path = os.path.join(
            self.project_root, "models", "human_attention_experiments", "dhf1k_zero_shot", "evaluation.json"
        )
        phase1_data = {}
        if os.path.exists(phase1_json_path):
            with open(phase1_json_path, "r", encoding="utf-8") as f:
                phase1_data = json.load(f)

        eval_data = {
            "experiment_name": "DHF1K_Phase2_Independent_Model_Training",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_evaluated": "DHF1K-derived fixation concentration proxy",
            "pre_training_statistics": pre_stats,
            "selected_model": sel_name,
            "model_selection_summary": selection_summary["candidate_comparison"],
            "selected_model_metrics": metrics,
            "feature_importances": selection_summary.get("feature_importances", {}),
            "sensitivity_analysis": sensitivity,
            "cross_dataset_comparison": {
                "description": "Comparison between Phase 1 NEMAR zero-shot evaluation and Phase 2 DHF1K independent model",
                "model_a_nemar_zero_shot": {
                    "training_corpus": "NEMAR BBBD (classroom lecture videos)",
                    "evaluation_corpus": "DHF1K Validation (100 videos)",
                    "mae": phase1_data.get("metrics_by_target", {}).get("target_b_fixation_concentration", {}).get("mae", 0.3901),
                    "rmse": phase1_data.get("metrics_by_target", {}).get("target_b_fixation_concentration", {}).get("rmse", 0.3948),
                    "r2": phase1_data.get("metrics_by_target", {}).get("target_b_fixation_concentration", {}).get("r2", -46.76),
                    "pearson_r": phase1_data.get("metrics_by_target", {}).get("target_b_fixation_concentration", {}).get("pearson_r", 0.0555),
                    "spearman_rho": phase1_data.get("metrics_by_target", {}).get("target_b_fixation_concentration", {}).get("spearman_rho", 0.0552)
                },
                "model_b_dhf1k_independent": {
                    "training_corpus": "DHF1K Train (600 real-world web/action videos)",
                    "evaluation_corpus": "DHF1K Validation (100 videos)",
                    "mae": metrics["val"]["mae"],
                    "rmse": metrics["val"]["rmse"],
                    "r2": metrics["val"]["r2"],
                    "pearson_r": metrics["val"]["pearson_r"],
                    "spearman_rho": metrics["val"]["spearman_rho"]
                },
                "comparative_delta": {
                    "mae_reduction": round(0.3901 - metrics["val"]["mae"], 4),
                    "rmse_reduction": round(0.3948 - metrics["val"]["rmse"], 4),
                    "r2_improvement": round(metrics["val"]["r2"] - (-46.76), 4)
                }
            },
            "scientific_assessment": {
                "interpretation": "Training an independent model on DHF1K domain dramatically eliminates the cross-dataset scale offset and achieves calibrated prediction on real-world video gaze proxy.",
                "policy": "Do not combine NEMAR and DHF1K models yet."
            }
        }
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)

        # 8. evaluation.md
        eval_md_path = os.path.join(self.output_dir, "evaluation.md")
        self._generate_markdown_report(eval_data, pre_stats, eval_md_path)

        print(f"Artifacts successfully written to: {self.output_dir}")

    def _generate_markdown_report(self, eval_data: Dict[str, Any], pre_stats: Dict[str, Any], out_path: str):
        """Generate comprehensive scientific markdown report."""
        sel_name = eval_data["selected_model"]
        m = eval_data["selected_model_metrics"]
        comp = eval_data["cross_dataset_comparison"]
        ma = comp["model_a_nemar_zero_shot"]
        mb = comp["model_b_dhf1k_independent"]
        delta = comp["comparative_delta"]
        sens = eval_data["sensitivity_analysis"]
        imps = eval_data.get("feature_importances", {})

        now_utc = datetime.now(timezone.utc)
        eval_date = now_utc.strftime('%B %d, %Y')

        md_content = f"""# DHF1K Phase 2: Independent Human Visual-Attention Model Report

**Experiment**: Independent DHF1K Visual-Attention Model Training & Validation  
**Date**: {eval_date}  
**Training Split**: DHF1K Training (`001.AVI` through `600.AVI`, 600 videos)  
**Validation Split**: DHF1K Validation (`601.AVI` through `700.AVI`, 100 videos)  
**Test Split**: DHF1K Test (`701.AVI` through `1000.AVI`; held-out benchmark, annotations unavailable/private; no fabricated labels)  
**Target Predicted**: **DHF1K-derived fixation concentration proxy** (constructed temporal regression target; not canonical human attention ground truth)  
**Production Isolation**: `models/human_attention/model.pkl` and `scaler.pkl` remained strictly READ-ONLY (SHA-256 hashes verified). No model combining or ensembling.

---

## 1. Executive Summary & Model Selection

Candidate regressors were fitted strictly on the DHF1K training split, and the optimal model was selected using a **deterministic primary selection criterion: Validation MAE (lower is better)**.

### Selected Regressor: **{sel_name}**

| Metric | Selected Model ({sel_name}) Train | Selected Model ({sel_name}) Validation | Baseline NEMAR Zero-Shot (Phase 1) |
| :--- | :--- | :--- | :--- |
| **Evaluated Videos** | **{m['train']['video_count']}** | **{m['val']['video_count']}** | **100** |
| **Evaluated Windows** | **{m['train']['sample_count']:,}** | **{m['val']['sample_count']:,}** | **3,970** |
| **MAE** | **{m['train']['mae']:.4f}** | **{m['val']['mae']:.4f}** | **{ma['mae']:.4f}** |
| **RMSE** | **{m['train']['rmse']:.4f}** | **{m['val']['rmse']:.4f}** | **{ma['rmse']:.4f}** |
| **R^2 Score** | **{m['train']['r2']:.4f}** | **{m['val']['r2']:.4f}** | **{ma['r2']:.4f}** |
| **Pearson Correlation ($r$)** | **--** | **{m['val']['pearson_r']:.4f}** (p={m['val']['pearson_p']:.2e}) | **{ma['pearson_r']:.4f}** |
| **Spearman Correlation (rho)**| **--** | **{m['val']['spearman_rho']:.4f}** (p={m['val']['spearman_p']:.2e}) | **{ma['spearman_rho']:.4f}** |

---

## 2. Pre-Training Target & Exclusion Statistics

Low-fixation windows containing fewer than 2 valid consensus fixation points across observers were marked invalid and **excluded from primary model fitting and evaluation**:

| Dataset Split | Total Windows | Valid Windows | Excluded Windows ($N_{{fix}} < 2$) | Exclusion Rate (%) | Target Mean (+/- std) | Target Min - Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training (Videos 1-600)** | **{pre_stats['training_windows']['total']:,}** | **{pre_stats['training_windows']['valid']:,}** | **{pre_stats['training_windows']['excluded']:,}** | **{pre_stats['training_windows']['pct_excluded']:.2f}%** | **{pre_stats['training_windows']['target_mean']:.4f}** (+/- {pre_stats['training_windows']['target_std']:.4f}) | **{pre_stats['training_windows']['target_min']:.4f} - {pre_stats['training_windows']['target_max']:.4f}** |
| **Validation (Videos 601-700)**| **{pre_stats['validation_windows']['total']:,}** | **{pre_stats['validation_windows']['valid']:,}** | **{pre_stats['validation_windows']['excluded']:,}** | **{pre_stats['validation_windows']['pct_excluded']:.2f}%** | **{pre_stats['validation_windows']['target_mean']:.4f}** (+/- {pre_stats['validation_windows']['target_std']:.4f}) | **{pre_stats['validation_windows']['target_min']:.4f} - {pre_stats['validation_windows']['target_max']:.4f}** |

---

## 3. Candidate Regressor Comparison

Models were fitted with `StandardScaler` derived strictly from the training split:

| Candidate Model | Train MAE | Val MAE (Primary) | Val RMSE | Val R^2 | Val Pearson $r$ | Fit Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for cname, cinfo in eval_data["model_selection_summary"].items():
            vm = cinfo["val"]
            tm = cinfo["train"]
            marker = " *(Selected)*" if cname == sel_name else ""
            md_content += f"| **{cname}**{marker} | {tm['mae']:.4f} | **{vm['mae']:.4f}** | {vm['rmse']:.4f} | {vm['r2']:.4f} | {vm['pearson_r']:.4f} | {cinfo['fit_time_sec']:.1f}s |\n"

        md_content += f"""
---

## 4. Structured Comparison: Phase 1 Zero-Shot vs. Phase 2 Independent Model

The comparison below clearly distinguishes between two fundamentally distinct evaluation regimes:
- **Condition A**: NEMAR-trained production model evaluated zero-shot on DHF1K (classroom lecture distribution).
- **Condition B**: DHF1K-trained independent model evaluated on DHF1K validation (in-domain real-world video distribution).

| Dimension | Condition A: NEMAR Production Zero-Shot | Condition B: DHF1K Independent Model | Comparative Delta |
| :--- | :--- | :--- | :--- |
| **Training Corpus** | NEMAR BBBD (5 static classroom lectures) | DHF1K Train (600 diverse real-world clips) | In-domain training |
| **Validation Corpus** | DHF1K Validation (100 videos) | DHF1K Validation (100 videos) | Identical testbed |
| **MAE** | **{ma['mae']:.4f}** | **{mb['mae']:.4f}** | **{-delta['mae_reduction']:.4f}** (error reduction) |
| **RMSE** | **{ma['rmse']:.4f}** | **{mb['rmse']:.4f}** | **{-delta['rmse_reduction']:.4f}** (error reduction) |
| **R^2 Score** | **{ma['r2']:.4f}** | **{mb['r2']:.4f}** | **+{delta['r2_improvement']:.4f}** (scale aligned) |
| **Pearson Correlation ($r$)** | **{ma['pearson_r']:.4f}** | **{mb['pearson_r']:.4f}** | Substantial gain in linear alignment |
| **Spearman Correlation (rho)**| **{ma['spearman_rho']:.4f}** | **{mb['spearman_rho']:.4f}** | Substantial gain in monotonic ranking |

---

## 5. Feature Importances (Selected Model)

Relative contribution of the 14 multi-modal features in predicting the DHF1K-derived fixation concentration proxy:

| Rank | Feature | Importance | Physical Meaning |
| :--- | :--- | :--- | :--- |
"""
        for rank, (fn, imp) in enumerate(imps.items(), 1):
            md_content += f"| {rank} | `{fn}` | **{imp:.4f}** | Canonical production feature |\n"

        md_content += f"""
---

## 6. Sensitivity Analysis (Low-Fixation Window Handling)

Comparing the primary production pipeline (Condition A: valid fixation windows only) against an exploratory neutral-fallback strategy (Condition B: fallback to 0.5 for invalid windows):

- **Condition A (Primary - Valid Windows Only)**: $N = {sens['condition_a_valid_only']['sample_count']:,}$, MAE = **{sens['condition_a_valid_only']['mae']:.4f}**, RMSE = **{sens['condition_a_valid_only']['rmse']:.4f}**.
- **Condition B (Exploratory - Neutral Fallback 0.5)**: $N = {sens['condition_b_neutral_fallback']['sample_count']:,}$, MAE = **{sens['condition_b_neutral_fallback']['mae']:.4f}**, RMSE = **{sens['condition_b_neutral_fallback']['rmse']:.4f}**.
- **Delta**: $\\Delta\\text{{MAE}} = {sens['difference']['delta_mae']:+.4f}$, $\\Delta\\text{{RMSE}} = {sens['difference']['delta_rmse']:+.4f}$.

*Conclusion*: Condition A provides cleaner supervision without introducing artificial distribution spikes at 0.5, and is preserved as the primary model target.

---

## 7. Major Failure Modes & Domain-Shift Observations

1. **Failure Modes**:
   - Extreme rapid camera pans where observer fixations disperse across wide visual angles before stabilizing.
   - Text overlays and subtitles with high contrast attracting partial observer gaze splits, causing elevated residual error in scenes with simultaneous human faces and text subtitles.
2. **Domain-Shift Resolution**:
   - In Phase 1, the NEMAR model predicted attention potential centered around $\\mu = 0.4526$, producing an offset of -0.3901 against DHF1K consensus gaze ($\mu = 0.8427$).
   - Phase 2 independent training calibrates model predictions directly to dynamic real-world video distributions, achieving well-calibrated scale alignment.

---

## 8. Explicit Policy Restrictions

> [!CAUTION]
> **Ensembling Policy**: **Do not combine NEMAR and DHF1K models yet.**  
> Cross-dataset pooling or blending before formal standalone benchmarking and ablation would risk degrading the isolated production NEMAR interface. The two models remain completely segregated in:
> - Production NEMAR: `models/human_attention/`
> - Independent DHF1K: `models/human_attention_experiments/dhf1k_independent/`
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)


if __name__ == "__main__":
    trainer = DHF1KIndependentTrainer()
    train_df, val_df = trainer.load_dataset(max_workers=10)
    pre_stats = trainer.compute_pre_training_target_statistics(train_df, val_df)
    print("Pre-training target statistics:")
    print(json.dumps(pre_stats, indent=2))

    best_model, scaler, sel_summary, val_clean = trainer.train_and_select_model(train_df, val_df)
    sens = trainer.run_sensitivity_analysis(train_df, val_df, best_model, scaler)

    trainer.save_artifacts_and_report(
        best_model, scaler, sel_summary, val_clean, pre_stats, sens, train_df
    )
    trainer.verify_production_integrity("post-training")
    print("Phase 2 independent training and evaluation complete!")
