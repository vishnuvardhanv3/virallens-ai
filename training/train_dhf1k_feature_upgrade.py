"""
DHF1K Phase 5: Feature Representation Upgrade Training & Evaluation Module.

Evaluates whether richer temporal dynamics, temporal context windowing,
and model capacity improve prediction of the DHF1K-derived fixation concentration proxy
over the Phase 2 ExtraTrees baseline.

Safety & Read-Only Constraints:
- Read-only access to production NEMAR (model.pkl, scaler.pkl, feature_schema.json)
- Read-only access to Phase 2 independent (model_dhf1k.pkl, scaler_dhf1k.pkl)
- Read-only access to DHF1K source data
- Scaler fit ONLY on training videos (001-600)
- Temporal features computed strictly per-video
- Outputs stored exclusively in models/human_attention_experiments/dhf1k_feature_upgrade/
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler


class DHF1KFeatureUpgradeTrainer:
    """
    Independent trainer and evaluator for DHF1K feature representation upgrades.
    """

    BASE_FEATURE_NAMES = [
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

    EXPECTED_INDEPENDENT_HASHES = {
        "model_dhf1k.pkl": "faffcb161ac9f895763383364af52527d7fc3f468614936b3d998ed4a275c7a0",
        "scaler_dhf1k.pkl": "8d83c2c336397f9dba5d05f68251f85837779a785875eb2df65b1fdf9f770cb8",
        "feature_schema_dhf1k.json": "80f774f0ded8033b18132beb27fce88865298f005a11a07834c8728524645903"
    }

    def __init__(
        self,
        project_root: str = r"E:\new viral",
        output_dir: str = r"E:\new viral\models\human_attention_experiments\dhf1k_feature_upgrade",
        cache_dir: str = r"E:\new viral\data\dhf1k_cache\video_records"
    ):
        self.project_root = project_root
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.production_dir = os.path.join(project_root, "models", "human_attention")
        self.independent_dir = os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_independent")

        os.makedirs(self.output_dir, exist_ok=True)
        self.verify_model_hashes("pre-execution")

    def verify_model_hashes(self, stage: str = "check") -> Dict[str, str]:
        """Verify production and Phase 2 independent model artifacts remain unmodified."""
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

    def load_and_engineer_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        """
        Load cached video records and compute temporal features and context strictly per video.
        Returns:
            train_df, val_df, feature_sets_dict
        """
        all_train_records = []
        all_val_records = []

        for vid in range(1, 701):
            cache_file = os.path.join(self.cache_dir, f"{vid:04d}.json")
            if not os.path.exists(cache_file):
                raise FileNotFoundError(f"Missing cached video record: {cache_file}")

            with open(cache_file, "r", encoding="utf-8") as f:
                records = json.load(f)

            if len(records) == 0:
                continue

            # Compute temporal features strictly within this video
            records = self._add_video_temporal_features(records)

            if vid <= 600:
                all_train_records.extend(records)
            else:
                all_val_records.extend(records)

        train_df = pd.DataFrame(all_train_records)
        val_df = pd.DataFrame(all_val_records)

        # Filter out invalid fixation windows (< 2 valid fixations)
        train_clean = train_df[train_df["is_valid_fixation_window"]].copy().reset_index(drop=True)
        val_clean = val_df[val_df["is_valid_fixation_window"]].copy().reset_index(drop=True)

        assert len(train_clean) == 22804, f"Expected 22,804 valid train windows, got {len(train_clean)}"
        assert len(val_clean) == 3970, f"Expected 3,970 valid val windows, got {len(val_clean)}"

        feature_sets = self._define_feature_sets()
        return train_clean, val_clean, feature_sets

    def _add_video_temporal_features(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute causal temporal deltas, trends, and context windows strictly per video sequence.
        """
        n = len(records)
        base = self.BASE_FEATURE_NAMES

        # Array of base features for fast vector operations: shape (n, 14)
        base_arr = np.zeros((n, len(base)), dtype=np.float32)
        for i, r in enumerate(records):
            base_arr[i] = [r[col] for col in base]

        # 1. First-order deltas: x_t - x_{t-1} (0 for t=0)
        deltas = np.zeros_like(base_arr)
        deltas[1:] = base_arr[1:] - base_arr[:-1]

        # 2. Trailing trends: (x_t - x_{t-2}) / 2 if t >= 2, else x_1 - x_0 if t == 1, else 0
        trends = np.zeros_like(base_arr)
        if n > 1:
            trends[1] = base_arr[1] - base_arr[0]
        if n > 2:
            trends[2:] = (base_arr[2:] - base_arr[:-2]) / 2.0

        # 3. Speech transition rate: |speech_t - speech_{t-1}|
        speech_idx = base.index("audio_speech_presence")
        speech_transitions = np.abs(deltas[:, speech_idx])

        # 4. Context C1: lag-1 base features (for t=0, pad with t=0)
        lag1_arr = np.zeros_like(base_arr)
        lag1_arr[0] = base_arr[0]
        if n > 1:
            lag1_arr[1:] = base_arr[:-1]

        # 5. Context C2: lead-1 base features (for t=n-1, pad with t=n-1) [Offline / Bidirectional]
        lead1_arr = np.zeros_like(base_arr)
        lead1_arr[-1] = base_arr[-1]
        if n > 1:
            lead1_arr[:-1] = base_arr[1:]

        for i, r in enumerate(records):
            # Dynamic features
            r["motion_magnitude_delta"] = float(deltas[i, base.index("motion_magnitude")])
            r["motion_magnitude_trend"] = float(trends[i, base.index("motion_magnitude")])
            r["brightness_delta"] = float(deltas[i, base.index("brightness_mean")])
            r["brightness_trend"] = float(trends[i, base.index("brightness_mean")])
            r["contrast_delta"] = float(deltas[i, base.index("contrast")])
            r["contrast_trend"] = float(trends[i, base.index("contrast")])
            r["visual_complexity_delta"] = float(deltas[i, base.index("visual_complexity")])
            r["audio_rms_delta"] = float(deltas[i, base.index("audio_rms")])
            r["audio_silence_delta"] = float(deltas[i, base.index("audio_silence_ratio")])
            r["audio_spectral_centroid_delta"] = float(deltas[i, base.index("audio_spectral_centroid")])
            r["audio_speech_transition_rate"] = float(speech_transitions[i])
            r["face_presence_delta"] = float(deltas[i, base.index("face_presence")])
            r["text_presence_delta"] = float(deltas[i, base.index("text_presence")])

            # Context features
            for b_idx, b_name in enumerate(base):
                r[f"{b_name}_lag1"] = float(lag1_arr[i, b_idx])
                r[f"{b_name}_lead1"] = float(lead1_arr[i, b_idx])

        return records

    def _define_feature_sets(self) -> Dict[str, List[str]]:
        """Define feature configurations for Experiments A, B, C1, and C2."""
        base = list(self.BASE_FEATURE_NAMES)

        temporal_dynamics = [
            "motion_magnitude_delta",
            "motion_magnitude_trend",
            "brightness_delta",
            "brightness_trend",
            "contrast_delta",
            "contrast_trend",
            "visual_complexity_delta",
            "audio_rms_delta",
            "audio_silence_delta",
            "audio_spectral_centroid_delta",
            "audio_speech_transition_rate",
            "face_presence_delta",
            "text_presence_delta"
        ]

        set_a = base
        set_b = base + temporal_dynamics
        set_c1 = base + [f"{b}_lag1" for b in base]
        set_c2 = base + [f"{b}_lag1" for b in base] + [f"{b}_lead1" for b in base]

        return {
            "set_a_baseline": set_a,
            "set_b_temporal_dynamics": set_b,
            "set_c1_causal_context": set_c1,
            "set_c2_bidirectional_context": set_c2
        }

    def train_and_evaluate_candidate(
        self,
        model_name: str,
        model: Any,
        feature_set_name: str,
        features: List[str],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Train candidate model on training set and evaluate on validation set with strict scaler isolation."""
        X_tr_raw = train_df[features].values.astype(np.float32)
        y_tr = train_df["target_fixation_concentration_proxy"].values.astype(np.float32)

        X_val_raw = val_df[features].values.astype(np.float32)
        y_val = val_df["target_fixation_concentration_proxy"].values.astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_val = scaler.transform(X_val_raw)

        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_time = time.time() - t0

        tr_preds = np.clip(model.predict(X_tr), 0.0, 1.0)
        val_preds = np.clip(model.predict(X_val), 0.0, 1.0)

        # Train metrics
        tr_mae = float(np.mean(np.abs(tr_preds - y_tr)))
        tr_rmse = float(np.sqrt(np.mean((tr_preds - y_tr) ** 2)))
        ss_tr_tot = np.sum((y_tr - np.mean(y_tr)) ** 2)
        tr_r2 = float(1.0 - (np.sum((y_tr - tr_preds) ** 2) / ss_tr_tot)) if ss_tr_tot > 1e-8 else 0.0

        # Val metrics
        val_mae = float(np.mean(np.abs(val_preds - y_val)))
        val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
        ss_val_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        val_r2 = float(1.0 - (np.sum((y_val - val_preds) ** 2) / ss_val_tot)) if ss_val_tot > 1e-8 else 0.0

        p_r, _ = stats.pearsonr(val_preds, y_val) if np.std(val_preds) > 1e-8 else (0.0, 1.0)
        s_rho, _ = stats.spearmanr(val_preds, y_val) if np.std(val_preds) > 1e-8 else (0.0, 1.0)

        return {
            "model_name": model_name,
            "feature_set_name": feature_set_name,
            "feature_count": len(features),
            "features": features,
            "fit_time_sec": round(fit_time, 2),
            "train_mae": round(tr_mae, 4),
            "train_rmse": round(tr_rmse, 4),
            "train_r2": round(tr_r2, 4),
            "val_mae": round(val_mae, 4),
            "val_rmse": round(val_rmse, 4),
            "val_r2": round(val_r2, 4),
            "pearson_r": round(float(p_r), 4),
            "spearman_rho": round(float(s_rho), 4),
            "val_predictions": val_preds,
            "model_instance": model,
            "scaler_instance": scaler
        }

    def compute_paired_video_bootstrap(
        self,
        val_df: pd.DataFrame,
        base_preds: np.ndarray,
        new_preds: np.ndarray,
        n_boot: int = 2000,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Compute paired deterministic bootstrap of 100 video-level MAEs.
        Difference: mean(MAE_new) - mean(MAE_base). Negative means new is better.
        """
        video_ids = sorted(val_df["video_id"].unique())
        n_videos = len(video_ids)

        base_maes = []
        new_maes = []

        for vid in video_ids:
            mask = (val_df["video_id"] == vid).values
            y = val_df.loc[mask, "target_fixation_concentration_proxy"].values
            b_p = base_preds[mask]
            n_p = new_preds[mask]

            base_maes.append(float(np.mean(np.abs(b_p - y))))
            new_maes.append(float(np.mean(np.abs(n_p - y))))

        base_maes = np.array(base_maes)
        new_maes = np.array(new_maes)
        deltas = new_maes - base_maes

        # Deterministic paired bootstrap
        rng = np.random.RandomState(random_seed)
        boot_diffs = np.zeros(n_boot, dtype=np.float32)

        for b in range(n_boot):
            sample_indices = rng.choice(n_videos, size=n_videos, replace=True)
            boot_diffs[b] = np.mean(deltas[sample_indices])

        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))
        boot_mean = float(np.mean(boot_diffs))
        boot_std = float(np.std(boot_diffs))

        # Per-video delta summary statistics
        improved_count = int(np.sum(deltas < -1e-4))
        worsened_count = int(np.sum(deltas > 1e-4))
        unchanged_count = int(n_videos - improved_count - worsened_count)

        return {
            "per_video_delta_stats": {
                "median": round(float(np.median(deltas)), 4),
                "mean": round(float(np.mean(deltas)), 4),
                "std": round(float(np.std(deltas)), 4),
                "p25": round(float(np.percentile(deltas, 25)), 4),
                "p75": round(float(np.percentile(deltas, 75)), 4),
            },
            "per_video_counts": {
                "improved_count": improved_count,
                "improved_pct": round(improved_count / n_videos * 100.0, 1),
                "worsened_count": worsened_count,
                "worsened_pct": round(worsened_count / n_videos * 100.0, 1),
                "unchanged_count": unchanged_count,
                "unchanged_pct": round(unchanged_count / n_videos * 100.0, 1),
            },
            "bootstrap_95_ci": {
                "n_iterations": n_boot,
                "random_seed": random_seed,
                "ci_lower_2_5": round(ci_lower, 4),
                "ci_upper_97_5": round(ci_upper, 4),
                "boot_mean": round(boot_mean, 4),
                "boot_std": round(boot_std, 4),
                "zero_in_ci": bool(ci_lower <= 0.0 <= ci_upper),
                "interpretation": (
                    f"95% bootstrap CI for mean video-level MAE difference is [{ci_lower:+.4f}, {ci_upper:+.4f}]. "
                    + ("Interval contains zero, indicating difference is not statistically distinguishable from zero at alpha=0.05."
                       if ci_lower <= 0.0 <= ci_upper else
                       "Interval excludes zero.")
                )
            },
            "per_video_records": [
                {
                    "video_id": vid,
                    "valid_windows": int(np.sum(val_df["video_id"] == vid)),
                    "base_mae": round(float(base_maes[idx]), 4),
                    "new_mae": round(float(new_maes[idx]), 4),
                    "delta_mae": round(float(deltas[idx]), 4),
                    "status": "improved" if deltas[idx] < -1e-4 else ("worsened" if deltas[idx] > 1e-4 else "unchanged")
                }
                for idx, vid in enumerate(video_ids)
            ]
        }

    def compute_target_bin_metrics(
        self, val_df: pd.DataFrame, base_preds: np.ndarray, new_preds: np.ndarray
    ) -> pd.DataFrame:
        """Compute error metrics across 4 target concentration bins."""
        y = val_df["target_fixation_concentration_proxy"].values
        bins = [
            ("0.60–0.70", (y >= 0.0) & (y < 0.70)),
            ("0.70–0.80", (y >= 0.70) & (y < 0.80)),
            ("0.80–0.90", (y >= 0.80) & (y < 0.90)),
            ("0.90–1.00", (y >= 0.90) & (y <= 1.00)),
        ]

        records = []
        for name, mask in bins:
            sub_y = y[mask]
            sub_b = base_preds[mask]
            sub_n = new_preds[mask]

            b_mae = float(np.mean(np.abs(sub_b - sub_y)))
            n_mae = float(np.mean(np.abs(sub_n - sub_y)))
            b_rmse = float(np.sqrt(np.mean((sub_b - sub_y) ** 2)))
            n_rmse = float(np.sqrt(np.mean((sub_n - sub_y) ** 2)))
            b_bias = float(np.mean(sub_b - sub_y))
            n_bias = float(np.mean(sub_n - sub_y))

            records.append({
                "target_bin": name,
                "sample_count": int(np.sum(mask)),
                "target_mean": round(float(np.mean(sub_y)), 4),
                "base_pred_mean": round(float(np.mean(sub_b)), 4),
                "new_pred_mean": round(float(np.mean(sub_n)), 4),
                "base_mae": round(b_mae, 4),
                "new_mae": round(n_mae, 4),
                "delta_mae": round(n_mae - b_mae, 4),
                "base_rmse": round(b_rmse, 4),
                "new_rmse": round(n_rmse, 4),
                "base_bias": round(b_bias, 4),
                "new_bias": round(n_bias, 4)
            })

        return pd.DataFrame(records)

    def compute_temporal_quartile_metrics(
        self, val_df: pd.DataFrame, base_preds: np.ndarray, new_preds: np.ndarray
    ) -> pd.DataFrame:
        """Compute performance across 4 temporal playback quartiles."""
        val_work = val_df.copy()
        # Compute normalized playback progress [0, 1] per video
        progress = []
        for vid, grp in val_work.groupby("video_id"):
            w_indices = grp["window_index"].values
            max_idx = np.max(w_indices)
            p = w_indices / max(max_idx, 1)
            progress.extend(p)

        val_work["playback_progress"] = progress
        val_work["base_pred"] = base_preds
        val_work["new_pred"] = new_preds
        val_work["target"] = val_df["target_fixation_concentration_proxy"].values

        quartiles = [
            ("Q1 (0.00–0.25)", (val_work["playback_progress"] >= 0.0) & (val_work["playback_progress"] < 0.25)),
            ("Q2 (0.25–0.50)", (val_work["playback_progress"] >= 0.25) & (val_work["playback_progress"] < 0.50)),
            ("Q3 (0.50–0.75)", (val_work["playback_progress"] >= 0.50) & (val_work["playback_progress"] < 0.75)),
            ("Q4 (0.75–1.00)", (val_work["playback_progress"] >= 0.75) & (val_work["playback_progress"] <= 1.00)),
        ]

        records = []
        for q_name, mask in quartiles:
            sub = val_work[mask]
            y = sub["target"].values
            b = sub["base_pred"].values
            n = sub["new_pred"].values

            b_mae = float(np.mean(np.abs(b - y)))
            n_mae = float(np.mean(np.abs(n - y)))

            records.append({
                "quartile": q_name,
                "sample_count": len(sub),
                "target_mean": round(float(np.mean(y)), 4),
                "base_mae": round(b_mae, 4),
                "new_mae": round(n_mae, 4),
                "delta_mae": round(n_mae - b_mae, 4)
            })

        return pd.DataFrame(records)

    def run_experiments(self) -> Dict[str, Any]:
        """Execute Experiments A, B, C1, C2, and D."""
        print("Loading cached video records and generating temporal feature sets...")
        train_df, val_df, feature_sets = self.load_and_engineer_features()

        results = {}

        # ---------------------------------------------------------
        # Experiment A: Baseline ExtraTrees with 14 features
        # ---------------------------------------------------------
        print("Running Experiment A: Phase 2 Baseline Reproduction...")
        model_et_base = ExtraTreesRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
        res_a = self.train_and_evaluate_candidate(
            "ExtraTreesRegressor", model_et_base, "set_a_baseline", feature_sets["set_a_baseline"],
            train_df, val_df
        )
        results["exp_a_baseline"] = res_a

        # ---------------------------------------------------------
        # Experiment B: Temporal Dynamics (27 features)
        # ---------------------------------------------------------
        print("Running Experiment B: Temporal Dynamics Features (Causal)...")
        model_et_b = ExtraTreesRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
        res_b = self.train_and_evaluate_candidate(
            "ExtraTreesRegressor", model_et_b, "set_b_temporal_dynamics", feature_sets["set_b_temporal_dynamics"],
            train_df, val_df
        )
        results["exp_b_temporal_dynamics"] = res_b

        # ---------------------------------------------------------
        # Experiment C: Temporal Context Windowing
        # ---------------------------------------------------------
        print("Running Experiment C1: Causal Lag Context [t-1, t] (28 features)...")
        model_et_c1 = ExtraTreesRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
        res_c1 = self.train_and_evaluate_candidate(
            "ExtraTreesRegressor", model_et_c1, "set_c1_causal_context", feature_sets["set_c1_causal_context"],
            train_df, val_df
        )
        results["exp_c1_causal_context"] = res_c1

        print("Running Experiment C2: Bidirectional Context [t-1, t, t+1] (42 features)...")
        model_et_c2 = ExtraTreesRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
        )
        res_c2 = self.train_and_evaluate_candidate(
            "ExtraTreesRegressor", model_et_c2, "set_c2_bidirectional_context", feature_sets["set_c2_bidirectional_context"],
            train_df, val_df
        )
        results["exp_c2_bidirectional_context"] = res_c2

        # ---------------------------------------------------------
        # Determine Best Feature Set among A, B, C1, C2
        # ---------------------------------------------------------
        feature_candidates = [res_a, res_b, res_c1, res_c2]
        best_feature_candidate = min(feature_candidates, key=lambda c: (c["val_mae"], c["val_rmse"], -c["val_r2"]))
        best_feat_set_name = best_feature_candidate["feature_set_name"]
        best_features = feature_sets[best_feat_set_name]

        print(f"Best feature set by validation MAE: {best_feat_set_name} (MAE = {best_feature_candidate['val_mae']:.4f})")

        # ---------------------------------------------------------
        # Experiment D: Model Capacity Exploration on Best Feature Set
        # ---------------------------------------------------------
        print(f"Running Experiment D: Model Capacity Exploration using {best_feat_set_name}...")
        capacity_models = {
            "ExtraTreesRegressor": ExtraTreesRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
            ),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1
            ),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                max_iter=100, max_depth=6, learning_rate=0.08, min_samples_leaf=10, random_state=42
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.08, min_samples_leaf=4, random_state=42
            ),
        }

        exp_d_results = {}
        for m_name, m_inst in capacity_models.items():
            print(f"  Training {m_name}...")
            res_d = self.train_and_evaluate_candidate(
                m_name, m_inst, best_feat_set_name, best_features, train_df, val_df
            )
            exp_d_results[f"exp_d_{m_name}"] = res_d

        results.update(exp_d_results)

        # ---------------------------------------------------------
        # Overall Winning Candidate Selection (Deterministic lowest val_mae)
        # ---------------------------------------------------------
        all_candidates = list(results.values())
        winner = min(all_candidates, key=lambda c: (c["val_mae"], c["val_rmse"], -c["val_r2"]))
        print(f"Winning candidate: {winner['model_name']} with {winner['feature_set_name']} (Val MAE: {winner['val_mae']:.4f})")

        # ---------------------------------------------------------
        # Comparisons & Diagnostics
        # ---------------------------------------------------------
        base_mae = res_a["val_mae"]
        win_mae = winner["val_mae"]
        abs_mae_imp = round(win_mae - base_mae, 4)
        rel_mae_imp = round(100.0 * (win_mae - base_mae) / base_mae, 2)
        r2_imp = round(winner["val_r2"] - res_a["val_r2"], 4)

        # Paired Video Bootstrap
        bootstrap_diag = self.compute_paired_video_bootstrap(
            val_df, res_a["val_predictions"], winner["val_predictions"]
        )

        # Target Bins
        bin_df = self.compute_target_bin_metrics(val_df, res_a["val_predictions"], winner["val_predictions"])

        # Temporal Quartiles
        temp_df = self.compute_temporal_quartile_metrics(val_df, res_a["val_predictions"], winner["val_predictions"])

        # Feature Importance (if tree-based model has feature_importances_)
        feature_importance_records = []
        if hasattr(winner["model_instance"], "feature_importances_"):
            importances = winner["model_instance"].feature_importances_
            feat_names = winner["features"]
            sorted_idx = np.argsort(importances)[::-1]
            cum_share = 0.0
            for rank, idx in enumerate(sorted_idx, 1):
                imp = float(importances[idx])
                cum_share += imp
                feature_importance_records.append({
                    "rank": rank,
                    "feature_name": feat_names[idx],
                    "importance_gini": round(imp, 4),
                    "cumulative_share": round(cum_share, 4)
                })

        # Decision Rule Evaluation
        decision = self._evaluate_decision_rule(
            abs_mae_imp, rel_mae_imp, r2_imp, bootstrap_diag, bin_df, temp_df, winner
        )

        return {
            "all_results": results,
            "baseline": res_a,
            "winner": winner,
            "comparison": {
                "baseline_val_mae": base_mae,
                "winner_val_mae": win_mae,
                "absolute_mae_change": abs_mae_imp,
                "relative_mae_change_pct": rel_mae_imp,
                "r2_change": r2_imp,
            },
            "bootstrap_diagnostics": bootstrap_diag,
            "target_bin_df": bin_df,
            "temporal_df": temp_df,
            "feature_importance_records": feature_importance_records,
            "decision": decision,
            "val_df": val_df
        }

    def _evaluate_decision_rule(
        self,
        abs_mae_imp: float,
        rel_mae_imp: float,
        r2_imp: float,
        boot_diag: Dict[str, Any],
        bin_df: pd.DataFrame,
        temp_df: pd.DataFrame,
        winner: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Multi-criteria decision rule gate.
        Requires:
        1. Meaningful aggregate MAE improvement (abs <= -0.002, rel <= -3.0%).
        2. Per-video consistency: more improved than worsened, bootstrap CI upper bound <= 0.
        3. No target-tail degradation.
        4. Temporal quartile stability (delta <= 0.003 across all quartiles).
        """
        counts = boot_diag["per_video_counts"]
        ci = boot_diag["bootstrap_95_ci"]

        # Tail check: extreme bins [0.60, 0.70) and [0.90, 1.00]
        tail_degraded = False
        for _, row in bin_df.iterrows():
            if row["target_bin"] in ["0.60–0.70", "0.90–1.00"]:
                if row["delta_mae"] > 0.005:
                    tail_degraded = True

        temporal_unstable = any(row["delta_mae"] > 0.003 for _, row in temp_df.iterrows())

        is_promising = (
            abs_mae_imp <= -0.002
            and rel_mae_imp <= -3.0
            and counts["improved_count"] > counts["worsened_count"]
            and ci["ci_upper_97_5"] <= 0.0
            and not tail_degraded
            and not temporal_unstable
        )

        if is_promising:
            status = "Representation upgrade promising"
            rationale = (
                f"Winning candidate ({winner['model_name']} on {winner['feature_set_name']}) achieves a statistically "
                f"and practically meaningful improvement (Delta MAE = {abs_mae_imp:+.4f}, relative = {rel_mae_imp:+.1f}%), "
                f"improving {counts['improved_count']}% of videos with 95% bootstrap CI [{ci['ci_lower_2_5']:+.4f}, {ci['ci_upper_97_5']:+.4f}] "
                f"strictly below zero, without tail degradation or temporal instability."
            )
            recommendation = "Recommend candidate representation for separate subsequent evaluation."
        else:
            status = "Representation upgrade marginal / Phase 2 model retained"
            rationale = (
                f"The representation upgrade yields a marginal change over the Phase 2 baseline (Delta MAE = {abs_mae_imp:+.4f}, "
                f"relative = {rel_mae_imp:+.2f}%, Delta R^2 = {r2_imp:+.4f}). "
                f"The 95% bootstrap confidence interval [{ci['ci_lower_2_5']:+.4f}, {ci['ci_upper_97_5']:+.4f}] "
                f"{'contains zero, confirming the difference is not statistically distinguishable from zero' if ci['zero_in_ci'] else 'demonstrates negligible practical effect'}. "
                f"Per-video balance is {counts['improved_count']} improved vs {counts['worsened_count']} worsened."
            )
            recommendation = "Retain Phase 2 independent model (models/human_attention_experiments/dhf1k_independent/model_dhf1k.pkl) unchanged."

        return {
            "status": status,
            "rationale": rationale,
            "recommendation": recommendation,
            "decision_factors": {
                "absolute_mae_change": abs_mae_imp,
                "relative_mae_change_pct": rel_mae_imp,
                "r2_change": r2_imp,
                "bootstrap_95_ci": f"[{ci['ci_lower_2_5']:+.4f}, {ci['ci_upper_97_5']:+.4f}]",
                "videos_improved_vs_worsened": f"{counts['improved_count']} improved vs {counts['worsened_count']} worsened",
                "tail_degradation_detected": tail_degraded,
                "temporal_instability_detected": temporal_unstable
            }
        }

    def save_artifacts(self, exp_data: Dict[str, Any]):
        """Save all 9 required artifacts under models/human_attention_experiments/dhf1k_feature_upgrade/."""
        print(f"Saving artifacts to {self.output_dir}...")
        winner = exp_data["winner"]
        base = exp_data["baseline"]
        comp = exp_data["comparison"]
        boot = exp_data["bootstrap_diagnostics"]
        bin_df = exp_data["target_bin_df"]
        temp_df = exp_data["temporal_df"]
        feat_imp_records = exp_data["feature_importance_records"]
        decision = exp_data["decision"]
        val_df = exp_data["val_df"]

        # 1. candidate_metrics.csv
        candidate_rows = []
        for key, c in exp_data["all_results"].items():
            candidate_rows.append({
                "experiment_stage": key,
                "model_name": c["model_name"],
                "feature_set": c["feature_set_name"],
                "feature_count": c["feature_count"],
                "fit_time_sec": c["fit_time_sec"],
                "train_mae": c["train_mae"],
                "train_rmse": c["train_rmse"],
                "train_r2": c["train_r2"],
                "val_mae": c["val_mae"],
                "val_rmse": c["val_rmse"],
                "val_r2": c["val_r2"],
                "pearson_r": c["pearson_r"],
                "spearman_rho": c["spearman_rho"],
                "delta_mae_vs_base": round(c["val_mae"] - base["val_mae"], 4),
                "relative_mae_change_pct": round(100.0 * (c["val_mae"] - base["val_mae"]) / base["val_mae"], 2),
                "delta_r2_vs_base": round(c["val_r2"] - base["val_r2"], 4)
            })
        cand_df = pd.DataFrame(candidate_rows)
        cand_df.to_csv(os.path.join(self.output_dir, "candidate_metrics.csv"), index=False)

        # 2. validation_predictions.csv
        pred_df = val_df[["dataset_source", "video_id", "split", "window_index", "window_start", "window_end", "timestamp"]].copy()
        pred_df["target_fixation_concentration_proxy"] = val_df["target_fixation_concentration_proxy"].values
        pred_df["predicted_baseline"] = np.round(base["val_predictions"], 4)
        pred_df["predicted_upgrade_winner"] = np.round(winner["val_predictions"], 4)
        pred_df["baseline_error"] = np.round(base["val_predictions"] - pred_df["target_fixation_concentration_proxy"], 4)
        pred_df["upgrade_winner_error"] = np.round(winner["val_predictions"] - pred_df["target_fixation_concentration_proxy"], 4)
        pred_df.to_csv(os.path.join(self.output_dir, "validation_predictions.csv"), index=False)

        # 3. feature_importance.csv
        feat_df = pd.DataFrame(feat_imp_records)
        feat_df.to_csv(os.path.join(self.output_dir, "feature_importance.csv"), index=False)

        # 4. target_bin_metrics.csv
        bin_df.to_csv(os.path.join(self.output_dir, "target_bin_metrics.csv"), index=False)

        # 5. per_video_metrics.csv
        vid_df = pd.DataFrame(boot["per_video_records"])
        vid_df.to_csv(os.path.join(self.output_dir, "per_video_metrics.csv"), index=False)

        # 6. temporal_metrics.csv
        temp_df.to_csv(os.path.join(self.output_dir, "temporal_metrics.csv"), index=False)

        # 7. experiment_metadata.json
        meta_data = {
            "experiment": "DHF1K_Phase5_Feature_Representation_Upgrade",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "train_videos": "001.AVI through 600.AVI",
            "val_videos": "601.AVI through 700.AVI",
            "window_size_sec": 0.5,
            "target": "DHF1K-derived fixation concentration proxy",
            "winning_candidate": {
                "model_name": winner["model_name"],
                "feature_set": winner["feature_set_name"],
                "feature_count": winner["feature_count"]
            },
            "decision": decision
        }
        with open(os.path.join(self.output_dir, "experiment_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        # 8. feature_upgrade_report.json
        report_json_data = {
            "experiment_name": "DHF1K_Phase5_Feature_Representation_Upgrade",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": "DHF1K-derived fixation concentration proxy",
            "safety_verification": "Strict read-only verification confirmed across production and Phase 2 models.",
            "comparison_summary": comp,
            "winning_candidate": {
                "model_name": winner["model_name"],
                "feature_set_name": winner["feature_set_name"],
                "feature_count": winner["feature_count"],
                "fit_time_sec": winner["fit_time_sec"],
                "train_metrics": {
                    "mae": winner["train_mae"],
                    "rmse": winner["train_rmse"],
                    "r2": winner["train_r2"]
                },
                "val_metrics": {
                    "mae": winner["val_mae"],
                    "rmse": winner["val_rmse"],
                    "r2": winner["val_r2"],
                    "pearson_r": winner["pearson_r"],
                    "spearman_rho": winner["spearman_rho"]
                }
            },
            "candidate_metrics": candidate_rows,
            "bootstrap_diagnostics": boot,
            "target_bin_metrics": bin_df.to_dict(orient="records"),
            "temporal_metrics": temp_df.to_dict(orient="records"),
            "feature_importance_top10": feat_imp_records[:10],
            "decision": decision
        }
        with open(os.path.join(self.output_dir, "feature_upgrade_report.json"), "w", encoding="utf-8") as f:
            json.dump(report_json_data, f, indent=2)

        # 9. feature_upgrade_report.md
        self._generate_markdown_report(report_json_data, cand_df, bin_df, temp_df, boot, feat_df)

    def _generate_markdown_report(
        self,
        rep: Dict[str, Any],
        cand_df: pd.DataFrame,
        bin_df: pd.DataFrame,
        temp_df: pd.DataFrame,
        boot: Dict[str, Any],
        feat_df: pd.DataFrame
    ):
        """Generate comprehensive scientific markdown report."""
        now_utc = datetime.now(timezone.utc).strftime("%B %d, %Y")
        comp = rep.get("comparison_summary", {})
        win = rep["winning_candidate"]
        dec = rep["decision"]
        ci = boot["bootstrap_95_ci"]
        p_stats = boot["per_video_delta_stats"]
        p_cnt = boot["per_video_counts"]

        md = f"""# DHF1K Phase 5: Feature Representation Upgrade Report

**Experiment**: Feature Representation Upgrade & Model Capacity Analysis for Human Visual-Attention Signal  
**Date**: {now_utc}  
**Dataset Source**: DHF1K (`E:\\\\viral_attention_data\\\\DHF1K`)  
**Target Variable**: **DHF1K-derived fixation concentration proxy** (constructed temporal regression target)  
**Safety & Isolation**: Production NEMAR model (`models/human_attention/`) and Phase 2 independent model (`models/human_attention_experiments/dhf1k_independent/`) verified unmodified.

---

## 1. Executive Summary & Required Comparison

| Representation / Model | Feature Count | Val MAE | Val RMSE | Val R² | Pearson $r$ | Spearman $\\rho$ | Absolute $\\Delta\\text{{MAE}}$ | Relative $\\Delta\\text{{MAE}}$ (%) | $\\Delta R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for _, row in cand_df.iterrows():
            md += (
                f"| **{row['experiment_stage']}** ({row['model_name']}) | {row['feature_count']} | "
                f"**{row['val_mae']:.4f}** | {row['val_rmse']:.4f} | {row['val_r2']:+.4f} | "
                f"{row['pearson_r']:.4f} | {row['spearman_rho']:.4f} | "
                f"{row['delta_mae_vs_base']:+.4f} | {row['relative_mae_change_pct']:+.2f}% | "
                f"{row['delta_r2_vs_base']:+.4f} |\n"
            )

        md += f"""
---

## 2. Decision-Quality & Video-Level Diagnostics

### Per-Video MAE Delta Distribution ($\\Delta = \\text{{MAE}}_{{\\text{{new}}}} - \\text{{MAE}}_{{\\text{{base}}}}$):
- **Mean Delta**: **{p_stats['mean']:+.4f}**
- **Median Delta**: **{p_stats['median']:+.4f}**
- **Standard Deviation**: **{p_stats['std']:.4f}**
- **25th Percentile ($Q_1$)**: **{p_stats['p25']:+.4f}**
- **75th Percentile ($Q_3$)**: **{p_stats['p75']:+.4f}**

### Video-Level Improvement Breakdown (100 Validation Videos):
- **Improved Videos**: **{p_cnt['improved_count']} / 100** ({p_cnt['improved_pct']}%)
- **Worsened Videos**: **{p_cnt['worsened_count']} / 100** ({p_cnt['worsened_pct']}%)
- **Unchanged Videos**: **{p_cnt['unchanged_count']} / 100** ({p_cnt['unchanged_pct']}%)

### Deterministic Paired Bootstrap 95% Confidence Interval ($B = 2,000$, seed=42):
- **95% Bootstrap CI**: **[{ci['ci_lower_2_5']:+.4f}, {ci['ci_upper_97_5']:+.4f}]**
- **Bootstrap Mean ($\\pm$ SD)**: **{ci['boot_mean']:+.4f}** ($\\pm$ {ci['boot_std']:.4f})
- **Diagnostic Finding**: {ci['interpretation']}

---

## 3. Robustness Stratification

### Target Concentration Bins:
| Target Range Bin | Sample Count | Target Mean | Base MAE | Upgrade MAE | $\\Delta\\text{{MAE}}$ | Base Bias | Upgrade Bias |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for _, row in bin_df.iterrows():
            md += (
                f"| **{row['target_bin']}** | {row['sample_count']:,} | {row['target_mean']:.4f} | "
                f"{row['base_mae']:.4f} | {row['new_mae']:.4f} | **{row['delta_mae']:+.4f}** | "
                f"{row['base_bias']:+.4f} | {row['new_bias']:+.4f} |\n"
            )

        md += f"""
### Temporal Playback Quartiles:
| Quartile Interval | Sample Count | Target Mean | Base MAE | Upgrade MAE | $\\Delta\\text{{MAE}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for _, row in temp_df.iterrows():
            md += (
                f"| **{row['quartile']}** | {row['sample_count']:,} | {row['target_mean']:.4f} | "
                f"{row['base_mae']:.4f} | {row['new_mae']:.4f} | **{row['delta_mae']:+.4f}** |\n"
            )

        md += f"""
---

## 4. Feature Importance Concentration (Top 10 Features)

| Rank | Feature Name | Impurity Importance (Gini) | Cumulative Share |
| :--- | :--- | :--- | :--- |
"""
        for _, row in feat_df.head(10).iterrows():
            md += f"| {int(row['rank'])} | `{row['feature_name']}` | **{row['importance_gini']:.4f}** | {row['cumulative_share']:.4f} |\n"

        top3_share = feat_df.iloc[2]["cumulative_share"] if len(feat_df) >= 3 else 0.0
        top5_share = feat_df.iloc[4]["cumulative_share"] if len(feat_df) >= 5 else 0.0
        top10_share = feat_df.iloc[9]["cumulative_share"] if len(feat_df) >= 10 else 0.0

        md += f"""
- **Top 3 Features Share**: **{top3_share*100:.1f}%**
- **Top 5 Features Share**: **{top5_share*100:.1f}%**
- **Top 10 Features Share**: **{top10_share*100:.1f}%**

> [!NOTE]
> **Non-Causal Note**: Feature importance metrics reflect Gini impurity reductions within the decision tree ensemble and describe predictive utility in this tabular space; they do **not** represent causal visual mechanisms.

---

## 5. Decision Rule & Final Recommendation

### Final Decision: **{dec['status']}**

**Rationale**:  
{dec['rationale']}

**Technical Recommendation**:  
{dec['recommendation']}
"""

        with open(os.path.join(self.output_dir, "feature_upgrade_report.md"), "w", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    trainer = DHF1KFeatureUpgradeTrainer()
    exp_data = trainer.run_experiments()
    trainer.save_artifacts(exp_data)
    trainer.verify_model_hashes("post-execution")
    print("DHF1K Phase 5 Feature Representation Upgrade execution completed successfully!")
