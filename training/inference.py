"""
Inference API for Human Visual Attention Model.

Exposes:
    predict_attention(video_path: str) -> List[Dict[str, Any]]

Returns temporal predictions:
[
    {
        "start": 0.0,
        "end": 0.5,
        "attention_score": 0.742,
        "confidence": 0.885
    }
]
Scores are normalized to 0.0–1.0.
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional
import joblib
import numpy as np

sys.path.append(r"E:\new viral\training")
from feature_extraction import VideoFeatureExtractor
from dataset_builder import AttentionDatasetBuilder


class AttentionPredictor:
    """
    Inference engine for human visual attention potential.
    """

    def __init__(
        self,
        model_dir: str = r"E:\new viral\models\human_attention",
        window_sec: float = 0.5
    ):
        self.model_dir = model_dir
        self.window_sec = window_sec
        self.model_path = os.path.join(model_dir, "model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.schema_path = os.path.join(model_dir, "feature_schema.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.feature_names = self.schema["feature_names"]
        self.extractor = VideoFeatureExtractor(window_sec=window_sec)

    def predict(
        self,
        video_path: str,
        max_duration_sec: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Run inference on an MP4 video and return temporal attention predictions.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Extract features for all temporal windows
        window_features = self.extractor.extract_features(video_path, max_duration_sec=max_duration_sec)
        if not window_features:
            return []

        # Build feature matrix in schema order
        rows = []
        for wf in window_features:
            rows.append([wf.get(fn, 0.0) for fn in self.feature_names])

        X = np.array(rows, dtype=np.float32)
        X_scaled = self.scaler.transform(X)

        # Regressor predictions
        raw_preds = self.model.predict(X_scaled)
        # Normalize strictly to [0.0, 1.0]
        norm_preds = np.clip(raw_preds, 0.0, 1.0)

        # Uncertainty estimation via tree ensemble variance
        has_estimators = hasattr(self.model, "estimators_")
        if has_estimators:
            tree_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            tree_stds = np.std(tree_preds, axis=0)
            confidences = np.clip(1.0 - (2.0 * tree_stds), 0.0, 1.0)
        else:
            confidences = np.full(len(norm_preds), 0.85, dtype=np.float32)

        results = []
        for i, wf in enumerate(window_features):
            results.append({
                "start": round(wf["window_start"], 2),
                "end": round(wf["window_end"], 2),
                "attention_score": round(float(norm_preds[i]), 4),
                "confidence": round(float(confidences[i]), 4)
            })

        return results


def predict_attention(
    video_path: str,
    model_dir: str = r"E:\new viral\models\human_attention"
) -> List[Dict[str, Any]]:
    """
    Public inference API function.
    """
    predictor = AttentionPredictor(model_dir=model_dir)
    return predictor.predict(video_path)


if __name__ == "__main__":
    test_video = r"E:\new viral\training\stimuli\task-stim01.mp4"
    if os.path.exists(test_video):
        pred = predict_attention(test_video)
        print(f"Predicted {len(pred)} windows for {os.path.basename(test_video)}:")
        print(json.dumps(pred[:5], indent=2))
