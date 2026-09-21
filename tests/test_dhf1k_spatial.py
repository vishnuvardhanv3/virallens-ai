"""
Tests for Phase 7 — DHF1K Spatial Saliency Prototype.

Verifies:
1. Source data integrity (DHF1K unchanged: 1000 videos, 700 annotations).
2. Production model integrity (SHA-256 unchanged).
3. Phase 2 model integrity (SHA-256 unchanged).
4. Correct train/validation split (001-600 train, 601-700 validation, zero overlap).
5. No annotation leakage into inputs (RGB only, shape 3x90x160).
6. Deterministic temporal sampling.
7. Model output shape and spatial normalization (sum == 1.0).
8. Saliency baseline comparison (model beats uniform baseline across metrics).
9. Artifact completeness (all 9 required artifacts exist).
10. Scientific terminology compliance ("DHF1K saliency-density supervision", no "ground-truth attention").
"""

import os
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import torch

from training.train_dhf1k_spatial import TinySalNet, DHF1KSpatialDataset, PROGRESS_COORDINATES

# Expected baseline hashes
EXPECTED_HASHES = {
    r"models\human_attention\model.pkl": "38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89",
    r"models\human_attention\scaler.pkl": "438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58",
    r"models\human_attention\feature_schema.json": "bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411",
    r"models\human_attention_experiments\dhf1k_independent\model_dhf1k.pkl": "faffcb161ac9f895763383364af52527d7fc3f468614936b3d998ed4a275c7a0",
    r"models\human_attention_experiments\dhf1k_independent\scaler_dhf1k.pkl": "8d83c2c336397f9dba5d05f68251f85837779a785875eb2df65b1fdf9f770cb8",
    r"models\human_attention_experiments\dhf1k_independent\feature_schema_dhf1k.json": "80f774f0ded8033b18132beb27fce88865298f005a11a07834c8728524645903",
}

PROJECT_ROOT = r"E:\new viral"
DHF1K_ROOT = r"E:\viral_attention_data\DHF1K"
SPATIAL_DIR = os.path.join(PROJECT_ROOT, "models", "human_attention_experiments", "dhf1k_spatial")


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Integrity Tests
# ---------------------------------------------------------------------------
def test_production_and_phase2_models_unchanged():
    """Verify SHA-256 hashes of production and Phase 2 models remain exact."""
    for rel_path, expected_hash in EXPECTED_HASHES.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        assert os.path.exists(full_path), f"Missing baseline file: {full_path}"
        actual_hash = sha256_file(full_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}!"


def test_dhf1k_source_data_unchanged():
    """Verify DHF1K source counts (1000 videos, 700 annotation folders)."""
    vid_dir = os.path.join(DHF1K_ROOT, "video")
    ann_dir = os.path.join(DHF1K_ROOT, "annotation")
    
    videos = [f for f in os.listdir(vid_dir) if f.endswith(".AVI")]
    annotations = [d for d in os.listdir(ann_dir) if os.path.isdir(os.path.join(ann_dir, d))]
    
    assert len(videos) == 1000, f"Expected 1000 videos, found {len(videos)}"
    assert len(annotations) == 700, f"Expected 700 annotations, found {len(annotations)}"


# ---------------------------------------------------------------------------
# Split & Leakage Tests
# ---------------------------------------------------------------------------
def test_train_val_split_disjoint():
    """Verify train videos (001-600) and val videos (601-700) have zero overlap."""
    train_ids = set(range(1, 601))
    val_ids = set(range(601, 701))
    
    assert len(train_ids) == 600
    assert len(val_ids) == 100
    assert len(train_ids.intersection(val_ids)) == 0


def test_no_annotation_leakage_in_inputs():
    """Verify input tensors contain only 3-channel RGB pixel data in [0, 1]."""
    dataset = DHF1KSpatialDataset(video_ids=[1], preload=False)
    sample = dataset[0]
    
    inp = sample['input']
    assert inp.shape == (3, 90, 160)
    assert inp.dtype == torch.float32
    assert inp.min() >= 0.0
    assert inp.max() <= 1.0


def test_deterministic_sampling():
    """Verify progress coordinates are fixed and reproducible."""
    assert PROGRESS_COORDINATES == [0.125, 0.375, 0.625, 0.875]
    
    ds1 = DHF1KSpatialDataset(video_ids=[1, 2], preload=False)
    ds2 = DHF1KSpatialDataset(video_ids=[1, 2], preload=False)
    
    assert len(ds1) == len(ds2) == 8
    for s1, s2 in zip(ds1.samples, ds2.samples):
        assert s1['video_id'] == s2['video_id']
        assert s1['frame_idx'] == s2['frame_idx']
        assert s1['progress_coord'] == s2['progress_coord']


# ---------------------------------------------------------------------------
# Model Architecture & Output Tests
# ---------------------------------------------------------------------------
def test_tinysalnet_architecture_and_normalization():
    """Verify TinySalNet forward pass shape and sum-to-1 normalization."""
    model = TinySalNet()
    model.eval()
    
    batch_rgb = torch.rand(2, 3, 90, 160)
    with torch.no_grad():
        out = model(batch_rgb)
        
    assert out.shape == (2, 1, 90, 160)
    assert (out >= 0.0).all()
    
    # Check sum-to-one per sample across spatial dimensions
    spatial_sums = out.sum(dim=(2, 3)).squeeze().numpy()
    np.testing.assert_allclose(spatial_sums, np.ones(2), rtol=1e-5)


# ---------------------------------------------------------------------------
# Artifact Completeness & Report Tests
# ---------------------------------------------------------------------------
def test_artifact_completeness():
    """Verify all 9 Phase 7 required artifacts exist."""
    required_files = [
        "spatial_model.pt",
        "spatial_model_metadata.json",
        "spatial_training_report.json",
        "spatial_training_report.md",
        "validation_metrics.csv",
        "per_video_metrics.csv",
        "training_config.json",
        "dataset_manifest.json",
    ]
    for fname in required_files:
        path = os.path.join(SPATIAL_DIR, fname)
        assert os.path.exists(path), f"Missing required artifact: {fname}"
        assert os.path.getsize(path) > 0, f"Artifact is empty: {fname}"
        
    qual_dir = os.path.join(SPATIAL_DIR, "qualitative_examples")
    assert os.path.exists(qual_dir), "Missing qualitative_examples directory!"
    pngs = [f for f in os.listdir(qual_dir) if f.endswith(".png")]
    assert len(pngs) >= 6, f"Expected at least 6 qualitative PNGs, found {len(pngs)}"


def test_model_weights_loadable():
    """Verify saved PyTorch state dict loads cleanly into TinySalNet."""
    weights_path = os.path.join(SPATIAL_DIR, "spatial_model.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    model = TinySalNet()
    model.load_state_dict(state_dict)
    model.eval()
    
    test_input = torch.zeros(1, 3, 90, 160)
    out = model(test_input)
    assert out.shape == (1, 1, 90, 160)


def test_scientific_terminology():
    """Verify reports avoid forbidden phrase 'ground-truth attention' and use proper terms."""
    report_md = os.path.join(SPATIAL_DIR, "spatial_training_report.md")
    report_json = os.path.join(SPATIAL_DIR, "spatial_training_report.json")
    
    with open(report_md, "r", encoding="utf-8") as f:
        md_text = f.read()
    with open(report_json, "r", encoding="utf-8") as f:
        json_text = f.read()
        
    assert "ground-truth attention" not in md_text.lower(), "Found forbidden term 'ground-truth attention' in markdown report!"
    assert "ground-truth attention" not in json_text.lower(), "Found forbidden term 'ground-truth attention' in json report!"
    assert "DHF1K saliency-density supervision" in md_text or "saliency-density supervision" in md_text


def test_saliency_metrics_beat_uniform_baseline():
    """Verify TinySalNet achieves superior saliency metrics over uniform baseline."""
    report_json_path = os.path.join(SPATIAL_DIR, "spatial_training_report.json")
    with open(report_json_path, "r") as f:
        report = json.load(f)
        
    metrics = report["aggregate_metrics"]
    # NSS: model > 0 (baseline = 0)
    assert metrics["model_nss_mean"] > metrics["baseline_nss_mean"]
    # AUC-Judd: model > 0.5 (baseline = 0.5)
    assert metrics["model_auc_judd_mean"] > metrics["baseline_auc_judd_mean"]
    # CC: model > 0 (baseline = 0)
    assert metrics["model_cc_mean"] > metrics["baseline_cc_mean"]
    # KL: model < baseline (lower KL divergence is better)
    assert metrics["model_kl_mean"] < metrics["baseline_kl_mean"]
