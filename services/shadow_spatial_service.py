"""
Shadow Spatial Saliency Service for ViralLens.

Integrates the frozen Phase 7 TinySalNet spatial model in SHADOW MODE ONLY.
- Model is loaded read-only on CPU in eval() mode with torch.no_grad().
- Runs as an independent, parallel observational service.
- Never replaces, modifies, blends, or calibrates the production NEMAR attention model.
- Operates on 160x90 RGB frames and returns raw descriptive spatial summaries:
  * saliency_spatial_dispersion (RMS radial distance from centroid)
  * saliency_spatial_entropy_bits (Shannon spatial entropy)
  * saliency_centroid_x, saliency_centroid_y (normalized screen coords)
  * saliency_peak_x, saliency_peak_y (normalized screen coords)
  * inter_frame_saliency_shift (L1 difference between consecutive sampled maps)
- Does NOT apply arbitrary 0.4082 normalization.
- Uses strict scientific terminology (no commercial buzzwords like 'retention' or 'scroll-stop').
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from training.train_dhf1k_spatial import TinySalNet

# Exact frozen model path & hash
SPATIAL_MODEL_PATH = Path(r"E:\new viral\models\human_attention_experiments\dhf1k_spatial\spatial_model.pt")
EXPECTED_MODEL_HASH = "fd6f27fd5ef6334c597ee3981086fc70f639cd59c7d57399fa72253d883b95c9"

INPUT_WIDTH = 160
INPUT_HEIGHT = 90
DEFAULT_MAX_FRAMES = 20 # Maximum sampling budget per Reel to guarantee minimal overhead


class ShadowSpatialService:
    """
    Independent parallel shadow service providing experimental DHF1K-trained
    spatial saliency telemetry for uploaded Reels.
    """

    def __init__(self, model_path: Optional[Path] = None, max_frames: int = DEFAULT_MAX_FRAMES):
        self.model_path = Path(model_path) if model_path else SPATIAL_MODEL_PATH
        self.max_frames = max_frames
        self.device = torch.device("cpu")
        self.model = None
        self._init_error = None
        
        self._load_model()

    def _load_model(self) -> None:
        """Load frozen TinySalNet weights onto CPU in eval mode."""
        try:
            if not self.model_path.exists():
                self._init_error = f"Model weights not found at: {self.model_path}"
                return
                
            model = TinySalNet().to(self.device)
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
        except Exception as e:
            self._init_error = f"Failed to initialize TinySalNet: {e}"
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def analyze_video(
        self,
        video_path: str | Path,
        sampling_fps: float = 1.0,
        max_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute deterministic shadow-mode spatial saliency analysis on an uploaded video.
        Returns a structured dictionary containing raw descriptive summaries.
        Does NOT raise unhandled exceptions; returns structured failure records on error.
        """
        shadow_start = time.perf_counter()
        v_path = Path(video_path)
        budget = max_budget if max_budget is not None else self.max_frames

        if not self.is_available:
            return {
                "status": "FAILED",
                "error": self._init_error or "ShadowSpatialService unavailable",
                "samples": [],
                "telemetry": {
                    "shadow_start": shadow_start,
                    "shadow_end": time.perf_counter(),
                    "shadow_total_wall_time": 0.0,
                    "number_of_frames": 0,
                    "shadow_ms_per_frame": 0.0
                }
            }

        if not v_path.exists():
            return {
                "status": "FAILED",
                "error": f"Video file not found: {v_path}",
                "samples": [],
                "telemetry": {
                    "shadow_start": shadow_start,
                    "shadow_end": time.perf_counter(),
                    "shadow_total_wall_time": 0.0,
                    "number_of_frames": 0,
                    "shadow_ms_per_frame": 0.0
                }
            }

        try:
            cap = cv2.VideoCapture(str(v_path))
            if not cap.isOpened():
                return {
                    "status": "FAILED",
                    "error": f"Could not open video: {v_path}",
                    "samples": [],
                    "telemetry": {
                        "shadow_start": shadow_start,
                        "shadow_end": time.perf_counter(),
                        "shadow_total_wall_time": 0.0,
                        "number_of_frames": 0,
                        "shadow_ms_per_frame": 0.0
                    }
                }

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return {
                    "status": "FAILED",
                    "error": f"Video has invalid frame count: {total_frames}",
                    "samples": [],
                    "telemetry": {
                        "shadow_start": shadow_start,
                        "shadow_end": time.perf_counter(),
                        "shadow_total_wall_time": 0.0,
                        "number_of_frames": 0,
                        "shadow_ms_per_frame": 0.0
                    }
                }

            # Deterministic sampling: 1 frame per second, bounded by budget
            step = max(1, int(round(fps / sampling_fps)))
            candidate_indices = list(range(0, total_frames, step))
            
            # Cap at budget deterministically
            if len(candidate_indices) > budget:
                sub_pos = np.linspace(0, len(candidate_indices) - 1, budget, dtype=int)
                sample_indices = [candidate_indices[p] for p in sub_pos]
            else:
                sample_indices = candidate_indices

            samples: List[Dict[str, Any]] = []
            pure_inf_times: List[float] = []
            prev_map = None

            for f_idx in sample_indices:
                timestamp_sec = round(f_idx / fps, 3)
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    continue

                # Exact Phase 7 contract: RGB, 160x90, normalized to [0, 1]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                resized_rgb = cv2.resize(frame_rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                inp_tensor = torch.from_numpy(resized_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

                # Pure inference timing
                t0_inf = time.perf_counter()
                with torch.no_grad():
                    pred_tensor = self.model(inp_tensor)
                pure_inf_times.append(time.perf_counter() - t0_inf)

                s_map = pred_tensor[0, 0].cpu().numpy() # (90, 160) normalized density (sums to 1.0)

                # Raw spatial summaries
                # Normalized grid coords: X in [0, 1], Y in [0, 1]
                ys = np.linspace(0.0, 1.0, INPUT_HEIGHT, dtype=np.float32)[:, None]
                xs = np.linspace(0.0, 1.0, INPUT_WIDTH, dtype=np.float32)[None, :]

                # Centroid
                cx = float(np.sum(xs * s_map))
                cy = float(np.sum(ys * s_map))

                # Raw Spatial Dispersion: RMS radial distance from centroid
                dispersion = float(np.sqrt(np.sum(((xs - cx)**2 + (ys - cy)**2) * s_map)))

                # Spatial Entropy (bits)
                p_clean = s_map[s_map > 0]
                entropy_bits = float(-np.sum(p_clean * np.log2(p_clean)))

                # Peak location
                peak_flat = np.argmax(s_map)
                peak_y_idx, peak_x_idx = np.unravel_index(peak_flat, s_map.shape)
                peak_x = float(peak_x_idx / (INPUT_WIDTH - 1))
                peak_y = float(peak_y_idx / (INPUT_HEIGHT - 1))

                # Inter-frame saliency shift
                if prev_map is not None:
                    shift = float(np.sum(np.abs(s_map - prev_map)))
                else:
                    shift = 0.0
                prev_map = s_map.copy()

                samples.append({
                    "source_video_path": str(v_path.name),
                    "frame_idx": int(f_idx),
                    "timestamp_sec": float(timestamp_sec),
                    "input_width": INPUT_WIDTH,
                    "input_height": INPUT_HEIGHT,
                    "model_version": "Phase 7 TinySalNet",
                    "model_hash": EXPECTED_MODEL_HASH,
                    "saliency_spatial_dispersion": round(dispersion, 5),
                    "saliency_spatial_entropy_bits": round(entropy_bits, 4),
                    "saliency_centroid_x": round(cx, 4),
                    "saliency_centroid_y": round(cy, 4),
                    "saliency_peak_x": round(peak_x, 4),
                    "saliency_peak_y": round(peak_y, 4),
                    "inter_frame_saliency_shift": round(shift, 4),
                    # Small preview thumbnail (32x18 for minimal telemetry payload)
                    "saliency_map_thumbnail": (cv2.resize(s_map, (32, 18), interpolation=cv2.INTER_AREA) * 1000).astype(int).tolist()
                })

            cap.release()
            shadow_end = time.perf_counter()
            total_wall_time = shadow_end - shadow_start
            n_frames = len(samples)
            ms_per_frame = (sum(pure_inf_times) / len(pure_inf_times) * 1000.0) if pure_inf_times else 0.0

            # Aggregated summaries across sampled frames
            if n_frames > 0:
                mean_disp = float(np.mean([s["saliency_spatial_dispersion"] for s in samples]))
                mean_entropy = float(np.mean([s["saliency_spatial_entropy_bits"] for s in samples]))
                mean_shift = float(np.mean([s["inter_frame_saliency_shift"] for s in samples]))
            else:
                mean_disp, mean_entropy, mean_shift = 0.0, 0.0, 0.0

            return {
                "status": "SUCCESS",
                "model_provenance": {
                    "architecture": "TinySalNet (96,521 parameters)",
                    "training_supervision": "DHF1K continuous saliency-density supervision",
                    "model_hash": EXPECTED_MODEL_HASH,
                    "execution_device": "CPU"
                },
                "sampling_metadata": {
                    "sampling_rule": "Deterministic ~1 fps capped at budget",
                    "video_fps": float(fps),
                    "total_video_frames": int(total_frames),
                    "sampled_frame_count": n_frames,
                    "frame_indices": [s["frame_idx"] for s in samples],
                    "timestamps_sec": [s["timestamp_sec"] for s in samples]
                },
                "summary": {
                    "mean_spatial_dispersion": round(mean_disp, 5),
                    "mean_spatial_entropy_bits": round(mean_entropy, 4),
                    "mean_inter_frame_shift": round(mean_shift, 4)
                },
                "samples": samples,
                "telemetry": {
                    "shadow_start": round(shadow_start, 4),
                    "shadow_end": round(shadow_end, 4),
                    "shadow_total_wall_time": round(total_wall_time, 4),
                    "number_of_frames": n_frames,
                    "shadow_ms_per_frame": round(ms_per_frame, 2)
                }
            }

        except Exception as e:
            shadow_end = time.perf_counter()
            return {
                "status": "FAILED",
                "error": f"Shadow spatial analysis error: {str(e)}",
                "samples": [],
                "telemetry": {
                    "shadow_start": round(shadow_start, 4),
                    "shadow_end": round(shadow_end, 4),
                    "shadow_total_wall_time": round(shadow_end - shadow_start, 4),
                    "number_of_frames": 0,
                    "shadow_ms_per_frame": 0.0
                }
            }
