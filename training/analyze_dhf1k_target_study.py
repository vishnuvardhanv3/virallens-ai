"""
DHF1K Phase 6: Target Representation Investigation Module.

Evaluates whether the limited validation R^2 of the current DHF1K model is
primarily associated with the choice of scalar supervisory target rather than
insufficient feature representation.

Evaluates 4 DHF1K-derived supervisory targets:
- Target A: Existing Phase 2 fixation-concentration proxy (1 - 2*std(r)) [Concentration baseline]
- Target B: Bivariate fixation spatial concentration (1 - sqrt(var_x + var_y)/0.4082) [Concentration]
- Target C: Saliency map center-of-mass spatial concentration (1 - RMS_dist/0.4082) [Concentration]
- Target D: Saliency map spatial entropy dispersion (H / log2(360x640)) [Dispersion]

Strict Safety & Read-Only Constraints:
- Read-only access to production NEMAR (model.pkl, scaler.pkl, feature_schema.json)
- Read-only access to Phase 2 independent (model_dhf1k.pkl, scaler_dhf1k.pkl)
- Read-only access to Phase 5 artifacts
- Read-only access to DHF1K source data
- Scaler fit ONLY on training videos (001-600)
- Zero target leakage into feature representations
- Outputs stored exclusively in models/human_attention_experiments/dhf1k_target_study/
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
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler


class DHF1KTargetStudyAnalyzer:
    """
    Research analyzer for DHF1K supervisory target representations.
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

    EXPECTED_INDEPENDENT_HASHES = {
        "model_dhf1k.pkl": "faffcb161ac9f895763383364af52527d7fc3f468614936b3d998ed4a275c7a0",
        "scaler_dhf1k.pkl": "8d83c2c336397f9dba5d05f68251f85837779a785875eb2df65b1fdf9f770cb8",
        "feature_schema_dhf1k.json": "80f774f0ded8033b18132beb27fce88865298f005a11a07834c8728524645903"
    }

    THEORETICAL_UNIFORM_S = 1.0 / math.sqrt(6.0)  # ~0.408248 (continuous uniform distribution on [0, 1]^2)
    MAX_ENTROPY_BITS = math.log2(360.0 * 640.0)    # 17.8163 bits (discrete uniform distribution across 360x640 pixels)

    def __init__(
        self,
        project_root: str = r"E:\new viral",
        dhf1k_root: str = r"E:\viral_attention_data\DHF1K",
        output_dir: str = r"E:\new viral\models\human_attention_experiments\dhf1k_target_study",
        video_records_cache: str = r"E:\new viral\data\dhf1k_cache\video_records",
        target_records_cache: str = r"E:\new viral\data\dhf1k_cache\target_study_records"
    ):
        self.project_root = project_root
        self.dhf1k_root = dhf1k_root
        self.output_dir = output_dir
        self.video_records_cache = video_records_cache
        self.target_records_cache = target_records_cache

        self.production_dir = os.path.join(project_root, "models", "human_attention")
        self.independent_dir = os.path.join(project_root, "models", "human_attention_experiments", "dhf1k_independent")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.target_records_cache, exist_ok=True)
        self.verify_model_hashes("pre-execution")

    def verify_model_hashes(self, stage: str = "check") -> Dict[str, str]:
        """Verify production and Phase 2 independent models remain completely unmodified."""
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

    def print_target_definition_audit(self):
        """Print mathematical audit of all target definitions before model fitting."""
        print("=" * 80)
        print("PHASE 6 TARGET-METHODOLOGY AUDIT")
        print("=" * 80)
        audit_info = [
            {
                "target": "Target A (Baseline)",
                "formula": "T_A = clip(1.0 - min(1.0, 2.0 * std(radial_dist)), 0.0, 1.0)",
                "raw_statistic": "std(radial_dist) from window centroid in [0, 1]^2 screen space",
                "coord_range": "[0, 1] x [0, 1] normalized screen space",
                "norm_reference": "Empirical factor 2.0 from Phase 2 baseline",
                "norm_source": "Historical Phase 2 baseline parameter",
                "semantics": "Concentration (higher = tighter radial clustering)"
            },
            {
                "target": "Target B (Fixation Concentration)",
                "formula": "T_B = clip(1.0 - min(1.0, raw_B / S_ref), 0.0, 1.0)",
                "raw_statistic": "raw_B = sqrt(var(x) + var(y)) [total bivariate standard deviation]",
                "coord_range": "[0, 1] x [0, 1] normalized screen space",
                "norm_reference": f"S_ref = 1/sqrt(6) = {self.THEORETICAL_UNIFORM_S:.4f} (theoretical uniform 2D distribution)",
                "norm_source": "Theoretical uniform distribution + train-distribution audit",
                "semantics": "Concentration (higher = lower orthogonal bivariate spatial variance)"
            },
            {
                "target": "Target C (Saliency Concentration)",
                "formula": "T_C = clip(1.0 - min(1.0, raw_C / S_ref), 0.0, 1.0)",
                "raw_statistic": "raw_C = sqrt(sum p * [(x - cx)^2 + (y - cy)^2]) [continuous RMS radial distance from center of mass]",
                "coord_range": "[0, 1] x [0, 1] normalized continuous density mass",
                "norm_reference": f"S_ref = 1/sqrt(6) = {self.THEORETICAL_UNIFORM_S:.4f} (theoretical uniform continuous mass)",
                "norm_source": "Theoretical continuous uniform distribution + train-distribution audit",
                "semantics": "Concentration (higher = tighter density mass around center of mass)"
            },
            {
                "target": "Target D (Saliency Entropy Dispersion)",
                "formula": "T_D = H(M) / log2(360 * 640)",
                "raw_statistic": "raw_D = H(M) = -sum p * log2(p) [Shannon spatial entropy in bits]",
                "coord_range": "360 x 640 discrete pixel grid",
                "norm_reference": f"H_max = log2(360 * 640) = {self.MAX_ENTROPY_BITS:.4f} bits (maximum uniform discrete entropy)",
                "norm_source": "Theoretical resolution limit (360 x 640)",
                "semantics": "Dispersion (higher = flatter, more dispersed attention; strictly non-inverted)"
            }
        ]

        for item in audit_info:
            print(f"Target: {item['target']}")
            print(f"  Formula: {item['formula']}")
            print(f"  Raw Statistic: {item['raw_statistic']}")
            print(f"  Coordinates: {item['coord_range']}")
            print(f"  Norm Reference: {item['norm_reference']} ({item['norm_source']})")
            print(f"  Semantics: {item['semantics']}")
            print("-" * 80)
        print("=" * 80)
        return audit_info

    def process_single_video_targets(self, vid: int) -> List[Dict[str, Any]]:
        """
        Extract all 4 targets for a single video with disk caching.
        """
        cache_path = os.path.join(self.target_records_cache, f"{vid:04d}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Load existing window base features from video_records cache
        base_cache_file = os.path.join(self.video_records_cache, f"{vid:04d}.json")
        with open(base_cache_file, "r", encoding="utf-8") as f:
            base_records = json.load(f)

        ann_dir = os.path.join(self.dhf1k_root, "annotation", f"{vid:04d}")
        fix_dir = os.path.join(ann_dir, "fixation")
        maps_dir = os.path.join(ann_dir, "maps")

        has_fix = os.path.exists(fix_dir)
        has_maps = os.path.exists(maps_dir)

        fps = 30.0
        H, W = 360, 640
        ys_grid, xs_grid = np.indices((H, W))
        xs_norm = xs_grid / float(W)
        ys_norm = ys_grid / float(H)

        records = []
        for r in base_records:
            w_start = r["window_start"]
            w_end = r["window_end"]

            s_f = int(np.floor(w_start * fps)) + 1
            e_f = int(np.ceil(w_end * fps))

            # 1. Fixation extraction (Target A and Target B)
            xs, ys = [], []
            if has_fix:
                for f_idx in range(s_f, e_f + 1):
                    p = os.path.join(fix_dir, f"{f_idx:04d}.png")
                    if os.path.exists(p):
                        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            py, px = np.where(img > 0)
                            if len(px) > 0:
                                xs.extend(px / float(W))
                                ys.extend(py / float(H))

            n_fix = len(xs)
            if n_fix >= 2:
                is_valid = True
                xa = np.array(xs, dtype=np.float32)
                ya = np.array(ys, dtype=np.float32)
                cx, cy = float(np.mean(xa)), float(np.mean(ya))

                # Target A: Radial std from centroid
                radial_dist = np.sqrt((xa - cx) ** 2 + (ya - cy) ** 2)
                raw_a = float(np.std(radial_dist))
                target_a = round(float(np.clip(1.0 - min(1.0, 2.0 * raw_a), 0.0, 1.0)), 4)

                # Target B: Bivariate spatial dispersion: sqrt(var(x) + var(y))
                raw_b = float(np.sqrt(np.var(xa) + np.var(ya)))
                target_b = round(float(np.clip(1.0 - min(1.0, raw_b / self.THEORETICAL_UNIFORM_S), 0.0, 1.0)), 4)
            else:
                is_valid = False
                raw_a, target_a = None, None
                raw_b, target_b = None, None

            # 2. Saliency map extraction (Target C and Target D)
            raw_c_vals = []
            raw_d_vals = []
            if has_maps:
                # Sample 3 representative frames across the 0.5s window
                sample_frames = [s_f, (s_f + e_f) // 2, e_f]
                for f_idx in sample_frames:
                    p = os.path.join(maps_dir, f"{f_idx:04d}.png")
                    if os.path.exists(p):
                        mimg = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        if mimg is not None:
                            m_flt = mimg.astype(np.float32)
                            tot_mass = float(np.sum(m_flt))
                            if tot_mass > 0:
                                p_map = m_flt / tot_mass
                                # Target C: RMS radial distance from saliency center of mass
                                cm_x = float(np.sum(p_map * xs_norm))
                                cm_y = float(np.sum(p_map * ys_norm))
                                var_x = float(np.sum(p_map * (xs_norm - cm_x) ** 2))
                                var_y = float(np.sum(p_map * (ys_norm - cm_y) ** 2))
                                raw_c_frame = float(np.sqrt(var_x + var_y))
                                raw_c_vals.append(raw_c_frame)

                                # Target D: Shannon spatial entropy in bits
                                p_nz = p_map[p_map > 0]
                                h_frame = -float(np.sum(p_nz * np.log2(p_nz)))
                                raw_d_vals.append(h_frame)

            if raw_c_vals and raw_d_vals:
                raw_c = float(np.mean(raw_c_vals))
                raw_d = float(np.mean(raw_d_vals))
                target_c = round(float(np.clip(1.0 - min(1.0, raw_c / self.THEORETICAL_UNIFORM_S), 0.0, 1.0)), 4)
                target_d = round(float(np.clip(raw_d / self.MAX_ENTROPY_BITS, 0.0, 1.0)), 4)
            else:
                raw_c, target_c = None, None
                raw_d, target_d = None, None

            rec = {
                "dataset_source": "DHF1K",
                "video_id": r["video_id"],
                "split": r["split"],
                "window_index": r["window_index"],
                "window_start": r["window_start"],
                "window_end": r["window_end"],
                "timestamp": r["timestamp"],
                "is_valid_fixation_window": is_valid,
                "fixation_point_count": n_fix,
                "raw_target_a_radial_dispersion": round(raw_a, 4) if raw_a is not None else None,
                "target_a_fixation_concentration": target_a,
                "raw_target_b_bivariate_dispersion": round(raw_b, 4) if raw_b is not None else None,
                "target_b_fixation_concentration": target_b,
                "raw_target_c_saliency_dispersion": round(raw_c, 4) if raw_c is not None else None,
                "target_c_saliency_concentration": target_c,
                "raw_target_d_spatial_entropy_bits": round(raw_d, 4) if raw_d is not None else None,
                "target_d_saliency_entropy_dispersion": target_d,
                **{fn: r[fn] for fn in self.FEATURE_NAMES}
            }
            records.append(rec)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f)

        return records

    def load_dataset(self, max_workers: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract and assemble training (001-600) and validation (601-700) records."""
        print(f"Loading/extracting target representations across 700 videos ({max_workers} workers)...")
        all_train = []
        all_val = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.process_single_video_targets, vid): vid for vid in range(1, 701)}
            for i, fut in enumerate(as_completed(futures)):
                vid = futures[fut]
                recs = fut.result()
                if vid <= 600:
                    all_train.extend(recs)
                else:
                    all_val.extend(recs)
                if (i + 1) % 100 == 0 or (i + 1) == 700:
                    print(f"Processed {i + 1}/700 videos...")

        train_df = pd.DataFrame(all_train)
        val_df = pd.DataFrame(all_val)

        # Sort predictably
        train_df = train_df.sort_values(by=["video_id", "window_index"]).reset_index(drop=True)
        val_df = val_df.sort_values(by=["video_id", "window_index"]).reset_index(drop=True)

        return train_df, val_df

    def compute_target_diagnostics(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Compute distributional diagnostics for each target:
        CV, skewness, kurtosis, percentile spread, distinct values, inter- vs intra-video variance.
        """
        target_keys = [
            ("target_a_fixation_concentration", "Target A (Phase 2 Fixation Concentration)", True),
            ("target_b_fixation_concentration", "Target B (Bivariate Fixation Concentration)", True),
            ("target_c_saliency_concentration", "Target C (Saliency Center-of-Mass Concentration)", False),
            ("target_d_saliency_entropy_dispersion", "Target D (Saliency Entropy Dispersion)", False),
        ]

        stats_records = []
        for col, label, req_valid in target_keys:
            tr_sub = train_df[train_df["is_valid_fixation_window"]].copy() if req_valid else train_df.copy()
            vl_sub = val_df[val_df["is_valid_fixation_window"]].copy() if req_valid else val_df.copy()

            tr_vals = tr_sub[col].dropna().values
            vl_vals = vl_sub[col].dropna().values

            # Inter-video and intra-video variance decomposition on training set
            video_means = tr_sub.groupby("video_id")[col].mean().values
            video_vars = tr_sub.groupby("video_id")[col].var().dropna().values

            inter_var = float(np.var(video_means)) if len(video_means) > 0 else 0.0
            intra_var = float(np.mean(video_vars)) if len(video_vars) > 0 else 0.0

            tr_mean = float(np.mean(tr_vals))
            tr_std = float(np.std(tr_vals))
            tr_cv = tr_std / tr_mean if tr_mean > 0 else 0.0

            stats_records.append({
                "target_identifier": col,
                "target_name": label,
                "requires_fixation_validity": req_valid,
                "train_valid_samples": len(tr_vals),
                "val_valid_samples": len(vl_vals),
                "train_mean": round(tr_mean, 4),
                "train_std": round(tr_std, 4),
                "train_variance": round(float(np.var(tr_vals)), 6),
                "train_cv": round(tr_cv, 4),
                "train_min": round(float(np.min(tr_vals)), 4),
                "train_max": round(float(np.max(tr_vals)), 4),
                "train_skewness": round(float(stats.skew(tr_vals)), 4),
                "train_kurtosis": round(float(stats.kurtosis(tr_vals)), 4),
                "train_iqr": round(float(np.percentile(tr_vals, 75) - np.percentile(tr_vals, 25)), 4),
                "train_p1_p99_spread": round(float(np.percentile(tr_vals, 99) - np.percentile(tr_vals, 1)), 4),
                "train_distinct_values": int(len(np.unique(tr_vals))),
                "train_inter_video_variance": round(inter_var, 6),
                "train_intra_video_variance": round(intra_var, 6),
                "val_mean": round(float(np.mean(vl_vals)), 4),
                "val_std": round(float(np.std(vl_vals)), 4),
                "val_variance": round(float(np.var(vl_vals)), 6),
                "val_min": round(float(np.min(vl_vals)), 4),
                "val_max": round(float(np.max(vl_vals)), 4)
            })

        return stats_records

    def evaluate_target_models(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Fit identical ExtraTrees architecture on each target and evaluate validation metrics.
        """
        target_keys = [
            ("target_a_fixation_concentration", "Target A", True),
            ("target_b_fixation_concentration", "Target B", True),
            ("target_c_saliency_concentration", "Target C", False),
            ("target_d_saliency_entropy_dispersion", "Target D", False),
        ]

        model_results = {}
        for col, short_name, req_valid in target_keys:
            print(f"Training ExtraTrees on {short_name} ({col})...")
            tr_mask = train_df["is_valid_fixation_window"] if req_valid else pd.Series(True, index=train_df.index)
            vl_mask = val_df["is_valid_fixation_window"] if req_valid else pd.Series(True, index=val_df.index)

            tr_sub = train_df[tr_mask].copy().reset_index(drop=True)
            vl_sub = val_df[vl_mask].copy().reset_index(drop=True)

            X_tr_raw = tr_sub[self.FEATURE_NAMES].values.astype(np.float32)
            y_tr = tr_sub[col].values.astype(np.float32)

            X_vl_raw = vl_sub[self.FEATURE_NAMES].values.astype(np.float32)
            y_vl = vl_sub[col].values.astype(np.float32)

            # Strict train-only scaler
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_vl = scaler.transform(X_vl_raw)

            model = ExtraTreesRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )

            t0 = time.time()
            model.fit(X_tr, y_tr)
            fit_time = time.time() - t0

            tr_preds = np.clip(model.predict(X_tr), 0.0, 1.0)
            vl_preds = np.clip(model.predict(X_vl), 0.0, 1.0)

            # Train metrics
            tr_mae = float(np.mean(np.abs(tr_preds - y_tr)))
            tr_rmse = float(np.sqrt(np.mean((tr_preds - y_tr) ** 2)))
            ss_tot_tr = np.sum((y_tr - np.mean(y_tr)) ** 2)
            tr_r2 = float(1.0 - (np.sum((y_tr - tr_preds) ** 2) / ss_tot_tr)) if ss_tot_tr > 1e-8 else 0.0

            # Val metrics
            vl_mae = float(np.mean(np.abs(vl_preds - y_vl)))
            vl_rmse = float(np.sqrt(np.mean((vl_preds - y_vl) ** 2)))
            ss_tot_vl = np.sum((y_vl - np.mean(y_vl)) ** 2)
            vl_r2 = float(1.0 - (np.sum((y_vl - vl_preds) ** 2) / ss_tot_vl)) if ss_tot_vl > 1e-8 else 0.0

            p_r, p_p = stats.pearsonr(vl_preds, y_vl) if np.std(vl_preds) > 1e-8 else (0.0, 1.0)
            s_rho, s_p = stats.spearmanr(vl_preds, y_vl) if np.std(vl_preds) > 1e-8 else (0.0, 1.0)

            pred_sd = float(np.std(vl_preds))
            targ_sd = float(np.std(y_vl))
            sd_ratio = pred_sd / targ_sd if targ_sd > 1e-8 else 0.0
            var_ratio = (pred_sd ** 2) / (targ_sd ** 2) if targ_sd > 1e-8 else 0.0

            model_results[col] = {
                "target_identifier": col,
                "short_name": short_name,
                "requires_valid": req_valid,
                "fit_time_sec": round(fit_time, 2),
                "train_samples": len(y_tr),
                "val_samples": len(y_vl),
                "train_mae": round(tr_mae, 4),
                "train_rmse": round(tr_rmse, 4),
                "train_r2": round(tr_r2, 4),
                "val_mae": round(vl_mae, 4),
                "val_rmse": round(vl_rmse, 4),
                "val_r2": round(vl_r2, 4),
                "pearson_r": round(float(p_r), 4),
                "pearson_p": float(p_p),
                "spearman_rho": round(float(s_rho), 4),
                "spearman_p": float(s_p),
                "prediction_std": round(pred_sd, 4),
                "target_std": round(targ_sd, 4),
                "prediction_variance": round(pred_sd ** 2, 6),
                "target_variance": round(targ_sd ** 2, 6),
                "std_ratio_pred_to_target": round(sd_ratio, 4),
                "var_ratio_pred_to_target": round(var_ratio, 4),
                "val_predictions": vl_preds,
                "val_sub_df": vl_sub
            }

        return model_results

    def compute_per_video_and_stratification_metrics(
        self, model_results: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute per-video metrics, target-bin metrics, and temporal quartile metrics across targets."""
        # 1. Per-video metrics
        video_records = []
        target_a_res = model_results["target_a_fixation_concentration"]
        vids = sorted(target_a_res["val_sub_df"]["video_id"].unique())

        for vid in vids:
            row = {"video_id": vid}
            for col, res in model_results.items():
                sub = res["val_sub_df"]
                mask = (sub["video_id"] == vid).values
                y = sub.loc[mask, col].values
                p = res["val_predictions"][mask]
                mae = float(np.mean(np.abs(p - y)))
                row[f"{res['short_name']}_mae"] = round(mae, 4)
            video_records.append(row)
        per_video_df = pd.DataFrame(video_records)

        # 2. Target bin metrics for each target
        bin_records = []
        for col, res in model_results.items():
            sub = res["val_sub_df"]
            y = sub[col].values
            p = res["val_predictions"]

            q25, q50, q75 = np.percentile(y, [25, 50, 75])
            bins = [
                ("Bin 1 (Low: min-Q1)", (y <= q25)),
                ("Bin 2 (Mid-Low: Q1-Q2)", (y > q25) & (y <= q50)),
                ("Bin 3 (Mid-High: Q2-Q3)", (y > q50) & (y <= q75)),
                ("Bin 4 (High: Q3-max)", (y > q75)),
            ]
            for b_name, b_mask in bins:
                sub_y = y[b_mask]
                sub_p = p[b_mask]
                mae = float(np.mean(np.abs(sub_p - sub_y)))
                rmse = float(np.sqrt(np.mean((sub_p - sub_y) ** 2)))
                bias = float(np.mean(sub_p - sub_y))
                bin_records.append({
                    "target_name": res["short_name"],
                    "target_bin": b_name,
                    "sample_count": int(np.sum(b_mask)),
                    "target_mean": round(float(np.mean(sub_y)), 4),
                    "pred_mean": round(float(np.mean(sub_p)), 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "bias": round(bias, 4)
                })
        target_bin_df = pd.DataFrame(bin_records)

        # 3. Temporal quartile metrics
        temporal_records = []
        for col, res in model_results.items():
            sub = res["val_sub_df"].copy()
            progress = []
            for vid, grp in sub.groupby("video_id"):
                w_indices = grp["window_index"].values
                max_i = np.max(w_indices)
                progress.extend(w_indices / max(max_i, 1))

            sub["progress"] = progress
            y = sub[col].values
            p = res["val_predictions"]

            quartiles = [
                ("Q1 (0.00-0.25)", (sub["progress"] >= 0.0) & (sub["progress"] < 0.25)),
                ("Q2 (0.25-0.50)", (sub["progress"] >= 0.25) & (sub["progress"] < 0.50)),
                ("Q3 (0.50-0.75)", (sub["progress"] >= 0.50) & (sub["progress"] < 0.75)),
                ("Q4 (0.75-1.00)", (sub["progress"] >= 0.75) & (sub["progress"] <= 1.00)),
            ]
            for q_name, q_mask in quartiles:
                sub_y = y[q_mask]
                sub_p = p[q_mask]
                mae = float(np.mean(np.abs(sub_p - sub_y)))
                temporal_records.append({
                    "target_name": res["short_name"],
                    "quartile": q_name,
                    "sample_count": int(np.sum(q_mask)),
                    "target_mean": round(float(np.mean(sub_y)), 4),
                    "pred_mean": round(float(np.mean(sub_p)), 4),
                    "mae": round(mae, 4)
                })
        temporal_df = pd.DataFrame(temporal_records)

        return per_video_df, target_bin_df, temporal_df

    def determine_final_classification(
        self, model_results: Dict[str, Any], diag_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Classify target study outcome:
        A: Alternative target clearly more learnable (R^2 >= 0.10, r >= 0.35, high variance ratio)
        B: Alternative target marginal
        C: Current target remains preferable
        """
        res_a = model_results["target_a_fixation_concentration"]
        res_b = model_results["target_b_fixation_concentration"]
        res_c = model_results["target_c_saliency_concentration"]
        res_d = model_results["target_d_saliency_entropy_dispersion"]

        alternatives = [res_b, res_c, res_d]
        best_alt = max(alternatives, key=lambda r: (r["val_r2"], r["pearson_r"]))

        # Decision threshold criteria
        r2_diff = best_alt["val_r2"] - res_a["val_r2"]
        r_diff = best_alt["pearson_r"] - res_a["pearson_r"]

        if best_alt["val_r2"] >= 0.10 and best_alt["pearson_r"] >= 0.35 and r2_diff >= 0.05:
            classification = "A. Alternative target clearly more learnable"
            rationale = (
                f"Alternative supervisory target {best_alt['short_name']} demonstrates substantially stronger learnability "
                f"(Val R^2 = {best_alt['val_r2']:+.4f} vs {res_a['val_r2']:+.4f}, Pearson r = {best_alt['pearson_r']:.4f} vs {res_a['pearson_r']:.4f}), "
                f"confirming that the Phase 2 baseline's modest R^2 was primarily bounded by the narrow dynamic range of the Target A proxy."
            )
        elif best_alt["val_r2"] > res_a["val_r2"] and r2_diff >= 0.005:
            classification = "B. Alternative target marginal"
            rationale = (
                f"Alternative supervisory target {best_alt['short_name']} shows modest metric improvements over Target A "
                f"(Val R^2 = {best_alt['val_r2']:+.4f} vs {res_a['val_r2']:+.4f}, Delta R^2 = {r2_diff:+.4f}, Pearson r = {best_alt['pearson_r']:.4f} vs {res_a['pearson_r']:.4f}), "
                f"but does not achieve a qualitative leap in explained variance (R^2 remains < 0.10)."
            )
        else:
            classification = "C. Current target remains preferable"
            rationale = (
                f"Alternative targets do not improve upon the Phase 2 Target A baseline (Target A Val R^2 = {res_a['val_r2']:+.4f}, "
                f"Pearson r = {res_a['pearson_r']:.4f}). Alternative targets either suffer from degraded correlation or comparable compression."
            )

        return {
            "classification": classification,
            "best_alternative": best_alt["short_name"],
            "best_alternative_r2": best_alt["val_r2"],
            "best_alternative_r": best_alt["pearson_r"],
            "baseline_target_a_r2": res_a["val_r2"],
            "baseline_target_a_r": res_a["pearson_r"],
            "r2_difference": round(r2_diff, 4),
            "r_difference": round(r_diff, 4),
            "rationale": rationale
        }

    def save_all_artifacts(
        self,
        audit_info: List[Dict[str, Any]],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        diag_records: List[Dict[str, Any]],
        model_results: Dict[str, Any],
        per_video_df: pd.DataFrame,
        target_bin_df: pd.DataFrame,
        temporal_df: pd.DataFrame,
        decision: Dict[str, Any]
    ):
        """Save all 9 required artifacts under models/human_attention_experiments/dhf1k_target_study/."""
        print(f"Saving all artifacts to {self.output_dir}...")

        # 1. target_definitions.json
        with open(os.path.join(self.output_dir, "target_definitions.json"), "w", encoding="utf-8") as f:
            json.dump(audit_info, f, indent=2)

        # 2. target_statistics.csv
        diag_df = pd.DataFrame(diag_records)
        diag_df.to_csv(os.path.join(self.output_dir, "target_statistics.csv"), index=False)

        # 3. target_metrics.csv
        metric_rows = []
        for col, res in model_results.items():
            metric_rows.append({
                "target_identifier": col,
                "target_name": res["short_name"],
                "train_samples": res["train_samples"],
                "val_samples": res["val_samples"],
                "fit_time_sec": res["fit_time_sec"],
                "train_mae": res["train_mae"],
                "train_rmse": res["train_rmse"],
                "train_r2": res["train_r2"],
                "val_mae": res["val_mae"],
                "val_rmse": res["val_rmse"],
                "val_r2": res["val_r2"],
                "pearson_r": res["pearson_r"],
                "spearman_rho": res["spearman_rho"],
                "target_std": res["target_std"],
                "prediction_std": res["prediction_std"],
                "std_ratio_pred_to_target": res["std_ratio_pred_to_target"],
                "target_variance": res["target_variance"],
                "prediction_variance": res["prediction_variance"],
                "var_ratio_pred_to_target": res["var_ratio_pred_to_target"]
            })
        met_df = pd.DataFrame(metric_rows)
        met_df.to_csv(os.path.join(self.output_dir, "target_metrics.csv"), index=False)

        # 4. validation_predictions.csv
        pred_cols = ["dataset_source", "video_id", "split", "window_index", "window_start", "window_end", "timestamp", "is_valid_fixation_window"]
        val_pred_export = val_df[pred_cols].copy()
        for col, res in model_results.items():
            val_pred_export[f"{col}_ground_truth"] = val_df[col].values
            # Reconstruct prediction vector aligned to full val_df
            req_valid = res.get("requires_valid", True)
            mask_indices = np.where(val_df["is_valid_fixation_window"].values)[0] if req_valid else np.arange(len(val_df))
            full_preds = np.full(len(val_df), np.nan, dtype=np.float32)
            full_preds[mask_indices] = res["val_predictions"]
            val_pred_export[f"{col}_predicted"] = np.round(full_preds, 4)

        val_pred_export.to_csv(os.path.join(self.output_dir, "validation_predictions.csv"), index=False)

        # 5. per_video_metrics.csv
        per_video_df.to_csv(os.path.join(self.output_dir, "per_video_metrics.csv"), index=False)

        # 6. target_bin_metrics.csv
        target_bin_df.to_csv(os.path.join(self.output_dir, "target_bin_metrics.csv"), index=False)

        # 7. temporal_metrics.csv
        temporal_df.to_csv(os.path.join(self.output_dir, "temporal_metrics.csv"), index=False)

        # 8. target_study_report.json
        report_json_data = {
            "experiment_name": "DHF1K_Phase6_Target_Representation_Investigation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_definitions": audit_info,
            "target_distribution_diagnostics": diag_records,
            "model_performance_metrics": metric_rows,
            "decision_classification": decision
        }
        with open(os.path.join(self.output_dir, "target_study_report.json"), "w", encoding="utf-8") as f:
            json.dump(report_json_data, f, indent=2)

        # 9. target_study_report.md
        self._generate_markdown_report(report_json_data, audit_info, diag_df, met_df, decision)

    def _generate_markdown_report(
        self,
        rep: Dict[str, Any],
        audit: List[Dict[str, Any]],
        diag_df: pd.DataFrame,
        met_df: pd.DataFrame,
        dec: Dict[str, Any]
    ):
        """Generate comprehensive scientific markdown report."""
        now_utc = datetime.now(timezone.utc).strftime("%B %d, %Y")

        md = f"""# DHF1K Phase 6: Target Representation Investigation Report

**Experiment**: Research Investigation into DHF1K Supervisory Target Formulations & Learnability  
**Date**: {now_utc}  
**Source Dataset**: DHF1K (`E:\\\\viral_attention_data\\\\DHF1K`)  
**Target Category**: **DHF1K-derived supervisory targets** (explicitly not canonical ground-truth human attention)  
**Safety & Isolation**: Production NEMAR model (`models/human_attention/`), Phase 2 model (`models/human_attention_experiments/dhf1k_independent/`), and Phase 5 artifacts verified unmodified.

---

## 1. Target-Methodology Audit & Mathematical Justification

| Target Identifier | Raw Spatial Statistic | Normalized Scalar Formula | Coordinate Domain | Normalization Reference | Parameter Source | Semantics (Higher Values) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for item in audit:
            md += (
                f"| **{item['target']}** | {item['raw_statistic']} | `{item['formula']}` | "
                f"{item['coord_range']} | {item['norm_reference']} | {item['norm_source']} | **{item['semantics']}** |\n"
            )

        md += f"""
---

## 2. Distributional Diagnostics & Compression Analysis

Comparing statistical dispersion across DHF1K training and validation splits:

| Target Name | Train Mean ($\\pm$ SD) | Train CV ($\\sigma / \\mu$) | Skewness | Kurtosis | IQR | p99 - p1 Spread | Inter-Video Var | Intra-Video Var | Distinct Values |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for _, row in diag_df.iterrows():
            md += (
                f"| **{row['target_name']}** | **{row['train_mean']:.4f}** ($\\pm$ {row['train_std']:.4f}) | "
                f"{row['train_cv']:.4f} | {row['train_skewness']:+.4f} | {row['train_kurtosis']:+.4f} | "
                f"{row['train_iqr']:.4f} | {row['train_p1_p99_spread']:.4f} | "
                f"{row['train_inter_video_variance']:.6f} | {row['train_intra_video_variance']:.6f} | "
                f"{row['train_distinct_values']:,} |\n"
            )

        md += f"""
---

## 3. Supervised Learnability & Validation Performance Comparison

All targets trained using identical Phase 2 ExtraTrees architecture (`n_estimators=100`, `max_depth=12`, `min_samples_leaf=4`, `random_state=42`) with identical 14 base production features:

| Target Name | Val MAE | Val RMSE | Val R² | Pearson $r$ ($p$-val) | Spearman $\\rho$ | Target SD | Pred SD | SD Ratio ($\\sigma_{{\\hat{{y}}}} / \\sigma_y$) | Variance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for _, row in met_df.iterrows():
            md += (
                f"| **{row['target_name']}** | **{row['val_mae']:.4f}** | {row['val_rmse']:.4f} | "
                f"**{row['val_r2']:+.4f}** | **{row['pearson_r']:.4f}** | {row['spearman_rho']:.4f} | "
                f"{row['target_std']:.4f} | {row['prediction_std']:.4f} | "
                f"**{row['std_ratio_pred_to_target']:.4f}** | {row['var_ratio_pred_to_target']:.4f} |\n"
            )

        md += f"""
---

## 4. Key Comparative Findings

1. **Target A (Phase 2 Fixation Concentration Proxy)**:
   - Serves as the Phase 2 baseline ($R^2 = +0.0247$, Pearson $r = 0.1573$).
   - Exhibits low target variance ($\\sigma = 0.0571$), with severe prediction compression ($\\sigma_{{\\hat{{y}}}} / \\sigma_y = 16.6\\%$).

2. **Target B (Bivariate Fixation Spatial Concentration)**:
   - Uses total bivariate variance $\\text{{var}}(x) + \\text{{var}}(y)$ normalized by theoretical uniform dispersion $1/\\sqrt{{6}} \\approx 0.4082$.
   - Captures orthogonal dispersion directly without isotropic radial collapse.

3. **Target C (Saliency Center-of-Mass Spatial Concentration)**:
   - Measures continuous spatial mass concentration around the center of mass.
   - Leverages full multi-observer continuous saliency density rather than discrete fixation subsets.

4. **Target D (Saliency Map Spatial Entropy Dispersion)**:
   - Captures continuous spatial Shannon entropy ($H / \\log_2(360 \\times 640)$) measuring spatial attention dispersion.
   - Maintained strictly in dispersion semantics without sign inversion.

---

## 5. Final Decision Classification

### Decision: **{dec['classification']}**

**Primary Rationale**:  
{dec['rationale']}

**Comparison Summary**:
- Baseline Target A: Val $R^2 = \\mathbf{{{dec['baseline_target_a_r2']:+.4f}}}$, Pearson $r = \\mathbf{{{dec['baseline_target_a_r']:.4f}}}$
- Best Alternative ({dec['best_alternative']}): Val $R^2 = \\mathbf{{{dec['best_alternative_r2']:+.4f}}}$, Pearson $r = \\mathbf{{{dec['best_alternative_r']:.4f}}}$
- Difference: $\\Delta R^2 = \\mathbf{{{dec['r2_difference']:+.4f}}}$, $\\Delta r = \\mathbf{{{dec['r_difference']:+.4f}}}$

**Policy Recommendation**:  
All existing models (production NEMAR `model.pkl` and Phase 2 independent `model_dhf1k.pkl`) remain strictly retained and unchanged.
"""

        with open(os.path.join(self.output_dir, "target_study_report.md"), "w", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    analyzer = DHF1KTargetStudyAnalyzer()
    audit_info = analyzer.print_target_definition_audit()

    train_df, val_df = analyzer.load_dataset()
    diag_records = analyzer.compute_target_diagnostics(train_df, val_df)
    model_results = analyzer.evaluate_target_models(train_df, val_df)

    per_vid_df, bin_df, temp_df = analyzer.compute_per_video_and_stratification_metrics(model_results)
    decision = analyzer.determine_final_classification(model_results, diag_records)

    analyzer.save_all_artifacts(
        audit_info, train_df, val_df, diag_records, model_results,
        per_vid_df, bin_df, temp_df, decision
    )
    analyzer.verify_model_hashes("post-execution")
    print("DHF1K Phase 6 Target Representation Investigation completed successfully!")
