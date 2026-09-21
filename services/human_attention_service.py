"""Human Visual Attention Inference Service.

Loads the trained Human Visual Attention model from NEMAR BBBD Experiment 1
and generates temporal visual-attention potential predictions for MP4 videos.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
import joblib
import numpy as np

from training.feature_extraction import VideoFeatureExtractor


EXPECTED_FEATURE_ORDER = [
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
    "audio_speech_presence",
]


class SchemaValidationError(ValueError):
    """Raised when model feature schema does not match the canonical specification."""
    pass


_ATTENTION_CACHE: Dict[str, List[Dict[str, Any]]] = {}


class HumanAttentionService:
    """Inference engine for the trained Human Visual Attention Regressor."""

    def __init__(
        self,
        model_dir: str | Path = r"E:\new viral\models\human_attention",
        window_sec: float = 0.5,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.window_sec = float(window_sec)

        self.model_path = self.model_dir / "model.pkl"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.schema_path = self.model_dir / "feature_schema.json"
        self.metadata_path = self.model_dir / "training_metadata.json"
        self.eval_path = self.model_dir / "evaluation.json"
        self.cog_model_path = self.model_dir / "cognitive_classifier.pkl"
        self.cog_scaler_path = self.model_dir / "cognitive_scaler.pkl"

        self._load_artifacts()
        self._validate_schema()
        self.extractor = VideoFeatureExtractor(window_sec=self.window_sec)

    def _load_artifacts(self) -> None:
        """Load trained regression and classification models, scalers, and metadata."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Trained scaler not found at {self.scaler_path}")
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at {self.schema_path}")

        # Load primary regressor & scaler
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        # Load schema, metadata, and evaluation logs
        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.evaluation: Dict[str, Any] = {}
        if self.eval_path.exists():
            with open(self.eval_path, "r", encoding="utf-8") as f:
                self.evaluation = json.load(f)

        # Auxiliary cognitive state classifier
        self.cognitive_model = None
        self.cognitive_scaler = None
        if self.cog_model_path.exists() and self.cog_scaler_path.exists():
            try:
                self.cognitive_model = joblib.load(self.cog_model_path)
                self.cognitive_scaler = joblib.load(self.cog_scaler_path)
            except Exception:
                self.cognitive_model = None
                self.cognitive_scaler = None

    def _validate_schema(self) -> None:
        """Strictly validate schema dimensionality, names, and ordering."""
        schema_names = self.schema.get("feature_names", [])
        if len(schema_names) != len(EXPECTED_FEATURE_ORDER):
            raise SchemaValidationError(
                f"Schema dimension mismatch: expected {len(EXPECTED_FEATURE_ORDER)} features, "
                f"found {len(schema_names)}"
            )

        if schema_names != EXPECTED_FEATURE_ORDER:
            raise SchemaValidationError(
                f"Feature ordering mismatch!\nExpected: {EXPECTED_FEATURE_ORDER}\nFound: {schema_names}"
            )

        self.feature_names = schema_names

    def predict_attention(
        self,
        video_path: str | Path,
        max_duration_sec: Optional[float] = None,
        audio_wav_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run temporal attention inference on an MP4 video."""
        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            st = v_path.stat()
            cache_key = f"{v_path.resolve()}_{st.st_size}_{st.st_mtime}_{max_duration_sec}"
            if cache_key in _ATTENTION_CACHE:
                return list(_ATTENTION_CACHE[cache_key])
        except Exception:
            cache_key = None

        # Extract features using exact training implementation (optionally reusing pre-extracted audio WAV)
        window_features = self.extractor.extract_features(
            str(v_path), max_duration_sec=max_duration_sec, audio_wav_path=audio_wav_path
        )
        if not window_features:
            return []

        # Validate feature matrix order and missing values
        rows = []
        for wf in window_features:
            row = []
            for fn in self.feature_names:
                val = wf.get(fn)
                if val is None or np.isnan(val):
                    # Deterministic safe fallback for input features
                    val = 0.0
                row.append(float(val))
            rows.append(row)

        X = np.array(rows, dtype=np.float32)
        if X.shape[1] != len(self.feature_names):
            raise SchemaValidationError(
                f"Extracted feature matrix columns ({X.shape[1]}) != schema ({len(self.feature_names)})"
            )

        # Scale features using saved scaler (never fit on inference data)
        X_scaled = self.scaler.transform(X)

        # Predict raw attention
        raw_preds = self.model.predict(X_scaled)
        # Strictly clip to [0.0, 1.0]
        norm_preds = np.clip(raw_preds, 0.0, 1.0)

        # Uncertainty estimation via tree ensemble dispersion
        confidences: List[Optional[float]] = []
        if hasattr(self.model, "estimators_"):
            try:
                tree_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                tree_stds = np.std(tree_preds, axis=0)
                # Formula: confidence = clip(1.0 - 2.0 * std, 0.0, 1.0)
                conf_arr = np.clip(1.0 - (2.0 * tree_stds), 0.0, 1.0)
                confidences = [round(float(c), 4) for c in conf_arr]
            except Exception:
                confidences = [None] * len(norm_preds)
        else:
            confidences = [None] * len(norm_preds)

        # Baseline attention for relative calculations (opening 2.0s or first min(4, len) windows)
        baseline_slice = norm_preds[: min(4, len(norm_preds))]
        baseline_score = float(np.mean(baseline_slice)) if len(baseline_slice) > 0 else 0.5
        body_slice = norm_preds[min(4, len(norm_preds)) :] if len(norm_preds) > 4 else norm_preds
        body_score = float(np.mean(body_slice)) if len(body_slice) > 0 else baseline_score
        opening_vs_body = round(baseline_score - body_score, 4)

        results: List[Dict[str, Any]] = []
        n_windows = len(window_features)

        for i, wf in enumerate(window_features):
            start_t = round(float(wf["window_start"]), 3)
            end_t = round(float(wf["window_end"]), 3)
            score = round(float(norm_preds[i]), 4)
            conf = confidences[i]

            # Relative measures
            att_delta = round(score - baseline_score, 4)
            norm_delta = round((score - baseline_score) / baseline_score, 4) if baseline_score > 0 else 0.0

            # Local peak & trough check
            prev_s = float(norm_preds[i - 1]) if i > 0 else score
            next_s = float(norm_preds[i + 1]) if i < n_windows - 1 else score
            is_local_peak = bool(score > prev_s and score >= next_s)
            is_local_trough = bool(score < prev_s and score <= next_s)

            # Peak prominence (height above higher neighboring trough)
            local_min = min(prev_s, next_s)
            prominence = round(max(0.0, score - local_min), 4) if is_local_peak else 0.0

            # Percentile within video
            pct_rank = round(float(np.mean(norm_preds <= score)), 4)
            is_top_percentile = bool(pct_rank >= 0.80)
            is_relative_peak = bool(is_local_peak and (att_delta > 0.03 or is_top_percentile))

            feat_vals = {fn: round(float(wf.get(fn, 0.0)), 4) for fn in self.feature_names}

            results.append({
                "start_time": start_t,
                "end_time": end_t,
                "start": start_t,
                "end": end_t,
                "attention_score": score,
                "confidence": conf,
                "baseline_attention": round(baseline_score, 4),
                "attention_delta": att_delta,
                "normalized_attention_delta": norm_delta,
                "percentile_rank": pct_rank,
                "is_local_peak": is_local_peak,
                "is_local_trough": is_local_trough,
                "relative_peak": is_relative_peak,
                "top_percentile_event": is_top_percentile,
                "peak_prominence": prominence,
                "opening_vs_body_diff": opening_vs_body,
                "feature_values": feat_vals,
            })

        # CRITICAL: Always enforce strictly chronological ordering
        results.sort(key=lambda x: (x["start_time"], x["end_time"]))
        if cache_key:
            _ATTENTION_CACHE[cache_key] = list(results)
        return results

    def check_generalization(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect out-of-distribution or weak generalization signals without altering predictions."""
        if not predictions:
            return {
                "generalization_warning": False,
                "reason": "No predictions to evaluate.",
                "severity": "none",
                "variance": 0.0,
                "std": 0.0,
                "out_of_range_features": [],
            }

        scores = [float(p["attention_score"]) for p in predictions]
        var = float(np.var(scores))
        std = float(np.std(scores))

        # Check out-of-distribution feature values against schema ranges
        schema_ranges = self.schema.get("feature_ranges", {})
        out_of_range = []
        for p in predictions:
            fvals = p.get("feature_values", {})
            for fn, val in fvals.items():
                if fn in schema_ranges:
                    rng = schema_ranges[fn]
                    fmin = float(rng.get("min", -9999.0))
                    fmax = float(rng.get("max", 9999.0))
                    fmean = float(rng.get("mean", 0.0))
                    fstd = float(rng.get("std", 1.0))
                    # Check if beyond 3 standard deviations or hard training bounds
                    if val < fmin - fstd or val > fmax + fstd:
                        if fn not in out_of_range:
                            out_of_range.append(fn)

        reasons = []
        severity = "none"
        is_warning = False

        if std < 0.015:
            is_warning = True
            severity = "medium"
            reasons.append(
                "Model-derived attention variation is low for this video. Interpret temporal differences cautiously "
                "because the model was trained on a limited laboratory stimulus set."
            )
        elif std < 0.03:
            severity = "low"
            reasons.append("Low dynamic attention range observed across temporal segments.")

        if out_of_range:
            is_warning = True
            if severity == "none":
                severity = "low"
            reasons.append(
                f"Feature values for [{', '.join(out_of_range[:3])}] are outside typical laboratory training ranges."
            )

        return {
            "generalization_warning": is_warning,
            "reason": "; ".join(reasons) if reasons else "Prediction distribution aligns with expected laboratory parameters.",
            "severity": severity,
            "variance": round(var, 6),
            "std": round(std, 4),
            "out_of_range_features": out_of_range,
        }


def predict_attention(
    video_path: str | Path,
    model_dir: str | Path = r"E:\new viral\models\human_attention",
) -> List[Dict[str, Any]]:
    """Standalone public interface for attention prediction."""
    service = HumanAttentionService(model_dir=model_dir)
    return service.predict_attention(video_path)

