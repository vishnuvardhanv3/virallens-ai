"""
Comprehensive Test Suite for Human Visual Attention Model and Pipeline.

Validates all 19 requirements from Section 17:
1. dataset metadata parsing
2. stimulus manifest
3. video accessibility
4. FPS extraction
5. duration validation
6. 128 Hz synchronization
7. gaze loading
8. coordinate normalization
9. interpolation-mask handling
10. fixation density
11. temporal aggregation
12. pupil normalization
13. participant split isolation
14. feature extraction
15. target generation
16. model training
17. model serialization
18. inference
19. prediction range
"""

import json
import math
import os
import sys
import numpy as np
import pytest

sys.path.append(r"E:\new viral\training")
from synchronization import VideoGazeSynchronizer
from gaze_processing import GazeProcessor
from feature_extraction import VideoFeatureExtractor


# Fixtures
@pytest.fixture(scope="session")
def project_root():
    return r"E:\new viral"


@pytest.fixture(scope="session")
def dataset_root():
    return r"E:\v1.0.0"


@pytest.fixture(scope="session")
def manifest_data(project_root):
    manifest_path = os.path.join(project_root, "training", "stimuli", "stimulus_manifest.json")
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def splits_data(project_root):
    splits_path = os.path.join(project_root, "training", "splits.json")
    assert os.path.exists(splits_path), f"Splits not found: {splits_path}"
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Test 1: Dataset metadata parsing
def test_dataset_metadata_parsing(dataset_root):
    readme_path = os.path.join(dataset_root, "README.md")
    desc_path = os.path.join(dataset_root, "dataset_description.json")
    assert os.path.exists(readme_path)
    assert os.path.exists(desc_path)
    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    assert "BBBD" in content or "Brain, Body" in content
    assert "Experiment 1" in content
    assert "EyeTrack" in content or "eyetrack" in content
    with open(desc_path, "r", encoding="utf-8") as f:
        desc = json.load(f)
    assert "Brain, Body" in desc["Name"]


# Test 2: Stimulus manifest
def test_stimulus_manifest(manifest_data):
    assert len(manifest_data) == 5
    stim_ids = [m["stimulus_id"] for m in manifest_data]
    expected_ids = ["task-stim01", "task-stim02", "task-stim03", "task-stim04", "task-stim05"]
    assert sorted(stim_ids) == sorted(expected_ids)

    for item in manifest_data:
        assert "stimulus_id" in item
        assert "source_id" in item
        assert "title" in item
        assert "duration" in item
        assert "local_path" in item
        assert "fps" in item
        assert "width" in item
        assert "height" in item
        assert item["source_status"] == "available_verified"


# Test 3 & 4 & 5: Video accessibility, FPS extraction, duration validation
def test_video_properties_and_fps(manifest_data):
    sync = VideoGazeSynchronizer()
    for item in manifest_data:
        v_path = item["local_path"]
        assert os.path.exists(v_path), f"Video missing: {v_path}"

        props = sync.get_video_properties(v_path)
        assert props["fps"] > 0, "FPS must be positive"
        assert abs(props["fps"] - item["fps"]) < 0.01, f"FPS mismatch: {props['fps']} vs {item['fps']}"
        assert props["width"] == 1280
        assert props["height"] == 720
        assert props["duration"] > 0
        assert abs(props["duration"] - item["duration"]) < 0.5


# Test 6: 128 Hz synchronization formula and alignment
def test_128hz_synchronization():
    sync = VideoGazeSynchronizer(sampling_rate_hz=128.0)

    # Known timestamps test: sample_index = floor(t * 128)
    assert sync.time_to_sample_index(0.0) == 0
    assert sync.time_to_sample_index(0.0078125) == 1
    assert sync.time_to_sample_index(0.5) == 64
    assert sync.time_to_sample_index(1.0) == 128
    assert sync.time_to_sample_index(184.859375) == 23662

    # Frame to sample mapping with 23.976 fps
    fps = 23.976023976
    # Frame 0 -> t=0 -> sample 0
    assert sync.frame_to_sample_index(0, fps) == 0
    # Frame 24 -> t ~ 1.001s -> sample 128
    assert sync.frame_to_sample_index(24, fps) == 128

    # Window slices
    s_idx, e_idx = sync.time_window_to_sample_slice(0.0, 0.5)
    assert s_idx == 0
    assert e_idx == 64


# Test 7: Gaze loading from primary derivative
def test_gaze_loading(dataset_root):
    gp = GazeProcessor(os.path.join(dataset_root, "derivatives"))
    rec = gp.load_participant_recording("sub-02", "ses-01", "task-stim01")

    assert rec["sample_count"] == 23662
    assert len(rec["x_screen"]) == 23662
    assert len(rec["y_screen"]) == 23662
    assert len(rec["vdx"]) == 23662
    assert len(rec["vdy"]) == 23662


# Test 8: Coordinate normalization into [0, 1]x[0, 1]
def test_coordinate_normalization(dataset_root):
    gp = GazeProcessor(os.path.join(dataset_root, "derivatives"))
    rec = gp.load_participant_recording("sub-02", "ses-01", "task-stim01")

    x_norm = rec["x_norm"]
    y_norm = rec["y_norm"]

    assert np.all(x_norm >= 0.0) and np.all(x_norm <= 1.0)
    assert np.all(y_norm >= 0.0) and np.all(y_norm <= 1.0)
    # Average gaze should be centered in the video
    assert 0.35 < np.mean(x_norm) < 0.65
    assert 0.35 < np.mean(y_norm) < 0.65


# Test 9: Interpolation mask handling
def test_interpolation_mask(dataset_root):
    gp = GazeProcessor(os.path.join(dataset_root, "derivatives"))
    rec = gp.load_participant_recording("sub-02", "ses-01", "task-stim01")

    mask = rec["interp_mask"]
    assert len(mask) == rec["sample_count"]
    # There should be both genuine and interpolated samples
    assert np.any(mask), "Should detect interpolated samples"
    assert np.any(~mask), "Should retain genuine measured gaze samples"
    # Interpolation fraction is typically 5-20% (blinks/artifacts)
    interp_pct = np.mean(mask)
    assert 0.01 <= interp_pct <= 0.35, f"Unexpected interpolation ratio: {interp_pct}"


# Test 10: Fixation density map generation
def test_fixation_density():
    x = np.array([0.5, 0.51, 0.49, 0.52])
    y = np.array([0.5, 0.49, 0.51, 0.50])
    heatmap = GazeProcessor.generate_spatial_density_map(x, y, grid_w=32, grid_h=18)

    assert heatmap.shape == (18, 32)
    # Density map must be a normalized probability distribution summing to 1.0
    assert abs(np.sum(heatmap) - 1.0) < 1e-4
    # Peak must be centered around (9, 16)
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    assert abs(peak_y - 9) <= 2
    assert abs(peak_x - 16) <= 2


# Test 11: Temporal window aggregation
def test_temporal_aggregation(dataset_root):
    gp = GazeProcessor(os.path.join(dataset_root, "derivatives"))
    rec = gp.load_participant_recording("sub-02", "ses-01", "task-stim01")
    target = gp.compute_window_attention_target([rec], 0.0, 1.0)

    assert "human_visual_attention" in target
    assert "gaze_dispersion" in target
    assert "fixation_rate" in target
    assert "saccade_rate" in target
    assert "blink_rate" in target
    assert target["valid_gaze_ratio"] > 0.0


# Test 12: Pupil normalization
def test_pupil_normalization(dataset_root):
    gp = GazeProcessor(os.path.join(dataset_root, "derivatives"))
    rec = gp.load_participant_recording("sub-02", "ses-01", "task-stim01")

    pupil_raw = rec["pupil_raw"]
    pupil_z = rec["pupil_z"]

    assert len(pupil_raw) == len(pupil_z)
    # Raw values are in camera sensor pixels (> 1000)
    assert np.nanmean(pupil_raw) > 500.0
    # Normalized values must have mean ~ 0 and std ~ 1
    assert abs(np.nanmean(pupil_z)) < 0.1
    assert abs(np.nanstd(pupil_z) - 1.0) < 0.15


# Test 13: Participant split isolation (Zero leakage)
def test_participant_split_isolation(splits_data):
    train_subs = set(splits_data["train"]["participants"])
    val_subs = set(splits_data["validation"]["participants"])
    test_subs = set(splits_data["test"]["participants"])

    assert len(train_subs) == 19
    assert len(val_subs) == 4
    assert len(test_subs) == 4

    # Mutual exclusivity checks
    assert len(train_subs.intersection(val_subs)) == 0, "Leakage between train and validation!"
    assert len(train_subs.intersection(test_subs)) == 0, "Leakage between train and test!"
    assert len(val_subs.intersection(test_subs)) == 0, "Leakage between validation and test!"


# Test 14: Feature extraction output
def test_feature_extraction(manifest_data):
    extractor = VideoFeatureExtractor(window_sec=0.5)
    first_video = manifest_data[0]["local_path"]
    feats = extractor.extract_features(first_video, max_duration_sec=2.0)

    assert len(feats) == 4
    expected_keys = [
        "motion_magnitude", "motion_std", "scene_cut_rate", "time_since_cut",
        "brightness_mean", "contrast", "visual_complexity",
        "audio_rms", "audio_silence_ratio", "audio_spectral_centroid", "audio_speech_presence"
    ]
    for row in feats:
        for k in expected_keys:
            assert k in row, f"Missing feature {k}"
            assert not math.isnan(row[k])


# Test 15: Target generation
def test_target_generation(dataset_root):
    gp = GazeProcessor(os.path.join(dataset_root, "derivatives"))
    rec1 = gp.load_participant_recording("sub-02", "ses-01", "task-stim01")
    rec2 = gp.load_participant_recording("sub-03", "ses-01", "task-stim01")

    res = gp.compute_window_attention_target([rec1, rec2], 5.0, 6.0)
    assert 0.0 <= res["human_visual_attention"] <= 1.0
    assert res["density_peak"] > 0.0


# Test 16 & 17: Model training & serialization
def test_model_training_and_serialization(project_root):
    model_dir = os.path.join(project_root, "models", "human_attention")
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    schema_path = os.path.join(model_dir, "feature_schema.json")
    meta_path = os.path.join(model_dir, "training_metadata.json")
    eval_path = os.path.join(model_dir, "evaluation.json")

    assert os.path.exists(model_path), f"Model not found: {model_path}"
    assert os.path.exists(scaler_path), f"Scaler not found: {scaler_path}"
    assert os.path.exists(schema_path), f"Schema not found: {schema_path}"
    assert os.path.exists(meta_path), f"Metadata not found: {meta_path}"
    assert os.path.exists(eval_path), f"Evaluation not found: {eval_path}"

    with open(schema_path, "r") as f:
        schema = json.load(f)
    assert schema["feature_count"] == 14

    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["target"] == "human_visual_attention"
    assert "limitations" in meta


# Test 18 & 19: Inference API and prediction range
def test_inference_api_and_range(project_root, manifest_data):
    from inference import predict_attention

    first_video = manifest_data[0]["local_path"]
    model_dir = os.path.join(project_root, "models", "human_attention")

    preds = predict_attention(first_video, model_dir=model_dir)
    assert len(preds) > 0

    for item in preds[:10]:
        assert "start" in item
        assert "end" in item
        assert "attention_score" in item
        assert "confidence" in item
        # Prediction score must be strictly normalized in [0.0, 1.0]
        score = item["attention_score"]
        assert 0.0 <= score <= 1.0, f"Attention score {score} out of [0.0, 1.0] range!"
        assert 0.0 <= item["confidence"] <= 1.0
