"""
Dataset Builder for Human Visual Attention Model and Cognitive State Classifier.

Integrates:
- Video and audio multi-modal features per 0.5s temporal window
- Synchronized eye-tracking targets aggregated across participants (ses-01 attentive condition)
- Participant-cohort targets (train subs, val subs, test subs) for zero-leakage evaluation
- Auxiliary cognitive viewing state dataset (ses-01 attentive vs ses-02 distracted)
"""

import json
import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

sys.path.append(r"E:\new viral\training")
from synchronization import VideoGazeSynchronizer
from gaze_processing import GazeProcessor
from feature_extraction import VideoFeatureExtractor


class AttentionDatasetBuilder:
    """
    Builds structured training, validation, and test datasets.
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

    TEMPORAL_EYE_FEATURES = [
        "fixation_rate",
        "saccade_rate",
        "blink_rate",
        "mean_pupil_z",
        "gaze_dispersion",
        "valid_gaze_ratio"
    ]

    def __init__(
        self,
        project_root: str = r"E:\new viral",
        raw_dataset_root: str = r"E:\v1.0.0",
        window_sec: float = 0.5
    ):
        self.project_root = project_root
        self.raw_dataset_root = raw_dataset_root
        self.window_sec = window_sec

        self.sync = VideoGazeSynchronizer()
        self.gaze_proc = GazeProcessor(os.path.join(raw_dataset_root, "derivatives"))
        self.extractor = VideoFeatureExtractor(window_sec=window_sec)

        self.splits_path = os.path.join(project_root, "training", "splits.json")
        self.manifest_path = os.path.join(project_root, "training", "stimuli", "stimulus_manifest.json")
        self.cache_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(self.cache_dir, exist_ok=True)

        with open(self.splits_path, "r", encoding="utf-8") as f:
            self.splits = json.load(f)

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def get_or_extract_video_features(self, stimulus_id: str, video_path: str, effective_dur: float) -> List[Dict[str, Any]]:
        """
        Extract video features with caching to avoid recomputation.
        """
        cache_file = os.path.join(self.cache_dir, f"{stimulus_id}_features.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                features = json.load(f)
            # Filter up to effective duration
            return [f for f in features if f["window_end"] <= effective_dur + 1e-3]

        print(f"Extracting video features for {stimulus_id} ({effective_dur:.1f}s)...")
        features = self.extractor.extract_features(video_path, max_duration_sec=effective_dur)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2)

        return features

    def build_primary_attention_dataset(self) -> Dict[str, Any]:
        """
        Build the primary human visual attention dataset across all 5 stimuli.
        Features = video multimodal metrics.
        Targets = human_visual_attention aggregated across train, val, and test participant cohorts.
        """
        train_pids = set(self.splits["train"]["participants"])
        val_pids = set(self.splits["validation"]["participants"])
        test_pids = set(self.splits["test"]["participants"])

        dataset_records = []

        for stim_item in self.manifest:
            stim_id = stim_item["stimulus_id"]
            video_path = stim_item["local_path"]

            # Load attentive session (ses-01) for all participants
            train_recs, val_recs, test_recs, all_recs = [], [], [], []

            for sub in sorted(os.listdir(self.gaze_proc.derivatives_root)):
                if not sub.startswith("sub-"):
                    continue
                try:
                    rec = self.gaze_proc.load_participant_recording(sub, "ses-01", stim_id)
                    all_recs.append(rec)
                    if sub in train_pids:
                        train_recs.append(rec)
                    elif sub in val_pids:
                        val_recs.append(rec)
                    elif sub in test_pids:
                        test_recs.append(rec)
                except Exception:
                    continue

            if not all_recs:
                continue

            effective_dur = min(stim_item["duration"], all_recs[0]["duration_sec"])
            video_feats = self.get_or_extract_video_features(stim_id, video_path, effective_dur)

            for feat in video_feats:
                w_start = feat["window_start"]
                w_end = feat["window_end"]

                if w_end > effective_dur:
                    break

                target_all = self.gaze_proc.compute_window_attention_target(all_recs, w_start, w_end)
                target_train = self.gaze_proc.compute_window_attention_target(train_recs, w_start, w_end) if train_recs else target_all
                target_val = self.gaze_proc.compute_window_attention_target(val_recs, w_start, w_end) if val_recs else target_all
                target_test = self.gaze_proc.compute_window_attention_target(test_recs, w_start, w_end) if test_recs else target_all

                record = {
                    "stimulus_id": stim_id,
                    "window_start": w_start,
                    "window_end": w_end,
                    # Multi-modal Video Features (X)
                    **{fn: feat[fn] for fn in self.FEATURE_NAMES},
                    # Targets across cohorts
                    "human_visual_attention_all": target_all["human_visual_attention"],
                    "human_visual_attention_train": target_train["human_visual_attention"],
                    "human_visual_attention_val": target_val["human_visual_attention"],
                    "human_visual_attention_test": target_test["human_visual_attention"],
                    # Auxiliary attention telemetry
                    "gaze_dispersion": target_all["gaze_dispersion"],
                    "fixation_rate": target_all["fixation_rate"],
                    "saccade_rate": target_all["saccade_rate"],
                    "blink_rate": target_all["blink_rate"],
                    "mean_pupil_z": target_all["mean_pupil_z"]
                }
                dataset_records.append(record)

        df = pd.DataFrame(dataset_records)
        save_path = os.path.join(self.cache_dir, "primary_attention_dataset.csv")
        df.to_csv(save_path, index=False)
        print(f"Primary attention dataset built: {len(df)} samples -> {save_path}")

        return {
            "total_samples": len(df),
            "columns": df.columns.tolist(),
            "save_path": save_path,
            "data": df
        }

    def build_cognitive_state_dataset(self, max_windows_per_rec: int = 150) -> Dict[str, Any]:
        """
        Build cognitive viewing state dataset: ses-01 (attentive, 1) vs ses-02 (distracted, 0).
        Evaluates temporal eye features with participant-level splitting.
        """
        train_pids = set(self.splits["train"]["participants"])
        val_pids = set(self.splits["validation"]["participants"])
        test_pids = set(self.splits["test"]["participants"])

        rows = []
        for stim_item in self.manifest:
            stim_id = stim_item["stimulus_id"]
            for sub in sorted(os.listdir(self.gaze_proc.derivatives_root)):
                if not sub.startswith("sub-"):
                    continue

                split_label = "train" if sub in train_pids else ("val" if sub in val_pids else "test")

                for ses_id, label in [("ses-01", 1), ("ses-02", 0)]:
                    try:
                        rec = self.gaze_proc.load_participant_recording(sub, ses_id, stim_id)
                    except Exception:
                        continue

                    dur = rec["duration_sec"]
                    num_windows = min(max_windows_per_rec, int(dur / self.window_sec))

                    # Sample every other window to maintain independence and manage dataset size
                    for w_idx in range(0, num_windows, 2):
                        w_start = w_idx * self.window_sec
                        w_end = (w_idx + 1) * self.window_sec

                        target_sub = self.gaze_proc.compute_window_attention_target([rec], w_start, w_end)
                        rows.append({
                            "participant_id": sub,
                            "session_id": ses_id,
                            "stimulus_id": stim_id,
                            "split": split_label,
                            "window_start": w_start,
                            "window_end": w_end,
                            "is_attentive": label,
                            **{fn: target_sub[fn] for fn in self.TEMPORAL_EYE_FEATURES}
                        })

        df = pd.DataFrame(rows)
        save_path = os.path.join(self.cache_dir, "cognitive_state_dataset.csv")
        df.to_csv(save_path, index=False)
        print(f"Cognitive state dataset built: {len(df)} samples -> {save_path}")

        return {
            "total_samples": len(df),
            "save_path": save_path,
            "data": df
        }


if __name__ == "__main__":
    builder = AttentionDatasetBuilder()
    prim = builder.build_primary_attention_dataset()
    print("Primary dataset shape:", prim["data"].shape)
