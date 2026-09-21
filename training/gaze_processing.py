"""
Gaze Processing, Pupil Normalization, and Attention Target Generation.

Handles:
- Loading cleaned derivative gaze (desc-gaze_visualangle_eyetrack)
- Tracking interpolation timestamps (desc-gaze_interpolation_timestamps)
- Coordinate normalization: Screen (1280x1024) to Video (1280x720 letterboxed)
- Pupil z-score normalization per participant/session (desc-pupil_eyetrack)
- Spatial fixation density mapping and dispersion metrics
- Temporal attention target calculation (human_visual_attention)
"""

import gzip
import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class GazeProcessor:
    """
    Processes eye-tracking signals from NEMAR BBBD Experiment 1 derivatives.
    """
    SCREEN_WIDTH: float = 1280.0
    SCREEN_HEIGHT: float = 1024.0
    VIDEO_WIDTH: float = 1280.0
    VIDEO_HEIGHT: float = 720.0
    # Center offset for 16:9 video on 5:4 monitor
    TOP_OFFSET: float = (1024.0 - 720.0) / 2.0  # 152.0 px
    LEFT_OFFSET: float = 0.0
    SAMPLING_RATE_HZ: float = 128.0

    def __init__(self, derivatives_root: str = r"E:\v1.0.0\derivatives"):
        self.derivatives_root = derivatives_root

    def load_participant_recording(
        self,
        participant_id: str,
        session_id: str,
        stimulus_id: str
    ) -> Dict[str, Any]:
        """
        Load gaze, pupil, fixations, saccades, blinks, and interpolation masks.
        """
        sub_dir = os.path.join(self.derivatives_root, participant_id, session_id, "eyetrack")
        prefix = f"{participant_id}_{session_id}_{stimulus_id}"

        # 1. Gaze coordinates (x, y, vdx, vdy) at 128 Hz
        gaze_file = os.path.join(sub_dir, f"{prefix}_desc-gaze_visualangle_eyetrack.tsv.gz")
        if not os.path.exists(gaze_file):
            raise FileNotFoundError(f"Gaze file not found: {gaze_file}")
        df_gaze = pd.read_csv(gaze_file, sep="\t", header=None, names=["x", "y", "vdx", "vdy"])

        # 2. Pupil size at 128 Hz
        pupil_file = os.path.join(sub_dir, f"{prefix}_desc-pupil_eyetrack.tsv.gz")
        df_pupil = pd.read_csv(pupil_file, sep="\t", header=None, names=["pupil_raw"])

        # 3. Interpolation timestamps
        interp_file = os.path.join(sub_dir, f"{prefix}_desc-gaze_interpolation_timestamps.tsv.gz")
        df_interp = pd.read_csv(interp_file, sep="\t") if os.path.exists(interp_file) else pd.DataFrame()

        # 4. Fixations
        fix_file = os.path.join(sub_dir, f"{prefix}_desc-fixations.tsv.gz")
        df_fix = pd.read_csv(fix_file, sep="\t") if os.path.exists(fix_file) else pd.DataFrame()

        # 5. Saccades
        sacc_file = os.path.join(sub_dir, f"{prefix}_desc-saccades.tsv.gz")
        df_sacc = pd.read_csv(sacc_file, sep="\t") if os.path.exists(sacc_file) else pd.DataFrame()

        # 6. Blinks
        blink_file = os.path.join(sub_dir, f"{prefix}_desc-blinks.tsv.gz")
        df_blink = pd.read_csv(blink_file, sep="\t") if os.path.exists(blink_file) else pd.DataFrame()

        # Build interpolation mask (1 = interpolated/blink, 0 = true measured gaze)
        total_samples = len(df_gaze)
        interp_mask = np.zeros(total_samples, dtype=bool)
        if not df_interp.empty:
            for _, row in df_interp.iterrows():
                start_idx = max(0, int(np.floor(row["start_time"] * self.SAMPLING_RATE_HZ)))
                end_idx = min(total_samples, int(np.ceil(row["end_time"] * self.SAMPLING_RATE_HZ)))
                interp_mask[start_idx:end_idx] = True

        # Coordinate transformation into normalized video coordinates [0, 1]
        x_vid = df_gaze["x"].values - self.LEFT_OFFSET
        y_vid = df_gaze["y"].values - self.TOP_OFFSET

        x_norm = np.clip(x_vid / self.VIDEO_WIDTH, 0.0, 1.0)
        y_norm = np.clip(y_vid / self.VIDEO_HEIGHT, 0.0, 1.0)

        # Pupil normalization (z-score per participant/session)
        pupil_raw = df_pupil["pupil_raw"].values
        valid_pupil = pupil_raw[~interp_mask] if np.any(~interp_mask) else pupil_raw
        p_mean = np.nanmean(valid_pupil) if len(valid_pupil) > 0 else 0.0
        p_std = np.nanstd(valid_pupil) if len(valid_pupil) > 0 else 1.0
        if p_std < 1e-4:
            p_std = 1.0
        pupil_z = (pupil_raw - p_mean) / p_std

        return {
            "participant_id": participant_id,
            "session_id": session_id,
            "stimulus_id": stimulus_id,
            "sample_count": total_samples,
            "duration_sec": total_samples / self.SAMPLING_RATE_HZ,
            "x_screen": df_gaze["x"].values,
            "y_screen": df_gaze["y"].values,
            "vdx": df_gaze["vdx"].values,
            "vdy": df_gaze["vdy"].values,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "interp_mask": interp_mask,
            "pupil_raw": pupil_raw,
            "pupil_z": pupil_z,
            "fixations": df_fix,
            "saccades": df_sacc,
            "blinks": df_blink
        }

    @staticmethod
    def generate_spatial_density_map(
        x_points: np.ndarray,
        y_points: np.ndarray,
        grid_w: int = 32,
        grid_h: int = 18,
        sigma: float = 1.5
    ) -> np.ndarray:
        """
        Generate a normalized 2D spatial fixation density map on a [0, 1]x[0, 1] grid.
        Returns a 2D probability distribution that sums to 1.0 (or uniform if no points).
        """
        if len(x_points) == 0:
            return np.ones((grid_h, grid_w), dtype=np.float32) / (grid_h * grid_w)

        # 2D Histogram binning
        x_bins = np.linspace(0.0, 1.0, grid_w + 1)
        y_bins = np.linspace(0.0, 1.0, grid_h + 1)

        heatmap, _, _ = np.histogram2d(y_points, x_points, bins=[y_bins, x_bins])
        heatmap = heatmap.astype(np.float32)

        # Gaussian kernel smoothing
        smoothed = gaussian_filter(heatmap, sigma=sigma)
        total = np.sum(smoothed)
        if total > 0:
            smoothed /= total
        else:
            smoothed = np.ones((grid_h, grid_w), dtype=np.float32) / (grid_h * grid_w)

        return smoothed

    @staticmethod
    def compute_window_attention_target(
        gaze_records: List[Dict[str, Any]],
        start_sec: float,
        end_sec: float
    ) -> Dict[str, float]:
        """
        Aggregate gaze observations across attentive participants in [start_sec, end_sec).
        Computes spatial fixation density and human_visual_attention score.
        """
        all_x, all_y = [], []
        total_samples = 0
        interpolated_samples = 0
        fixation_counts = 0
        saccade_counts = 0
        blink_counts = 0
        pupil_z_vals = []

        fs = 128.0
        start_idx = int(np.floor(start_sec * fs))
        end_idx = int(np.ceil(end_sec * fs))

        for rec in gaze_records:
            n = rec["sample_count"]
            s_idx = min(start_idx, n)
            e_idx = min(end_idx, n)
            if s_idx >= e_idx:
                continue

            sub_mask = rec["interp_mask"][s_idx:e_idx]
            sub_x = rec["x_norm"][s_idx:e_idx]
            sub_y = rec["y_norm"][s_idx:e_idx]
            sub_pz = rec["pupil_z"][s_idx:e_idx]

            total_samples += len(sub_mask)
            interpolated_samples += int(np.sum(sub_mask))

            # Only consider genuine, non-interpolated gaze for spatial distribution
            valid = ~sub_mask
            if np.any(valid):
                all_x.extend(sub_x[valid])
                all_y.extend(sub_y[valid])
                pupil_z_vals.extend(sub_pz[valid])

            # Fixation count overlapping this window
            if not rec["fixations"].empty:
                f_df = rec["fixations"]
                f_overlap = f_df[(f_df["end_time"] >= start_sec) & (f_df["start_time"] < end_sec)]
                fixation_counts += len(f_overlap)

            # Saccade count overlapping this window
            if not rec["saccades"].empty:
                s_df = rec["saccades"]
                s_overlap = s_df[(s_df["end_time"] >= start_sec) & (s_df["start_time"] < end_sec)]
                saccade_counts += len(s_overlap)

            # Blink count overlapping this window
            if not rec["blinks"].empty:
                b_df = rec["blinks"]
                b_overlap = b_df[(b_df["end_time"] >= start_sec) & (b_df["start_time"] < end_sec)]
                blink_counts += len(b_overlap)

        num_participants = len(gaze_records) if len(gaze_records) > 0 else 1
        window_dur = max(1e-4, end_sec - start_sec)

        valid_ratio = 1.0 - (interpolated_samples / total_samples) if total_samples > 0 else 0.0

        if len(all_x) > 1:
            x_arr = np.array(all_x)
            y_arr = np.array(all_y)
            # Spatial dispersion: radial standard deviation of normalized coordinates
            center_x = np.mean(x_arr)
            center_y = np.mean(y_arr)
            radial_dist = np.sqrt((x_arr - center_x) ** 2 + (y_arr - center_y) ** 2)
            dispersion = float(np.std(radial_dist))

            # Spatial density map peak
            density_map = GazeProcessor.generate_spatial_density_map(x_arr, y_arr)
            density_peak = float(np.max(density_map))
        else:
            center_x, center_y = 0.5, 0.5
            dispersion = 0.5
            density_peak = 1.0 / (32 * 18)

        # Scientific target: human_visual_attention
        # Higher when participants are steadily fixating on a consistent focus of interest,
        # with low gaze dispersion across participants and high genuine fixation ratio.
        attention_concentration = max(0.0, 1.0 - min(1.0, 2.5 * dispersion))
        human_visual_attention = float(np.clip(valid_ratio * (0.5 * attention_concentration + 0.5 * min(1.0, density_peak * 15.0)), 0.0, 1.0))

        return {
            "window_start": start_sec,
            "window_end": end_sec,
            "human_visual_attention": round(human_visual_attention, 4),
            "valid_gaze_ratio": round(valid_ratio, 4),
            "gaze_dispersion": round(dispersion, 4),
            "density_peak": round(density_peak, 5),
            "mean_gaze_x": round(float(center_x), 4),
            "mean_gaze_y": round(float(center_y), 4),
            "fixation_rate": round((fixation_counts / num_participants) / window_dur, 3),
            "saccade_rate": round((saccade_counts / num_participants) / window_dur, 3),
            "blink_rate": round((blink_counts / num_participants) / window_dur, 3),
            "mean_pupil_z": round(float(np.mean(pupil_z_vals)), 4) if len(pupil_z_vals) > 0 else 0.0
        }
