"""
Video and Eye-Tracking Synchronization Module for NEMAR BBBD Experiment 1.

The eye-tracking data was recorded at 128 Hz starting at t = 0.
Synchronization Formula:
    video_time = frame_index / fps
    sample_index = floor(video_time * 128)
"""

import math
import os
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


class VideoGazeSynchronizer:
    """
    Synchronizes stimulus video frames with 128 Hz eye-tracking recordings.
    """
    SAMPLING_RATE_HZ: float = 128.0
    SAMPLE_DURATION_SEC: float = 1.0 / 128.0

    def __init__(self, sampling_rate_hz: float = 128.0):
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.sample_duration_sec = 1.0 / self.sampling_rate_hz

    @staticmethod
    def get_video_properties(video_path: str) -> Dict[str, Any]:
        """
        Extract exact FPS, frame count, width, height, and duration using OpenCV.
        Does NOT assume FPS.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps} for {video_path}")

        duration = frame_count / fps

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "video_path": os.path.abspath(video_path)
        }

    def frame_to_time(self, frame_index: int, fps: float) -> float:
        """
        Calculate video timestamp in seconds for a given 0-indexed frame.
        """
        if frame_index < 0:
            raise ValueError(f"frame_index cannot be negative: {frame_index}")
        return frame_index / fps

    def time_to_sample_index(self, video_time: float) -> int:
        """
        Convert video timestamp in seconds to eye-tracking sample index at 128 Hz.
        sample_index = floor(video_time * 128)
        """
        if video_time < 0:
            raise ValueError(f"video_time cannot be negative: {video_time}")
        return int(math.floor(video_time * self.sampling_rate_hz))

    def frame_to_sample_index(self, frame_index: int, fps: float) -> int:
        """
        Direct conversion from frame index to eye-tracking sample index.
        """
        t = self.frame_to_time(frame_index, fps)
        return self.time_to_sample_index(t)

    def sample_index_to_time(self, sample_index: int) -> float:
        """
        Convert eye-tracking sample index to start timestamp in seconds.
        """
        if sample_index < 0:
            raise ValueError(f"sample_index cannot be negative: {sample_index}")
        return sample_index / self.sampling_rate_hz

    def time_window_to_sample_slice(
        self,
        start_time: float,
        end_time: float,
        max_samples: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Get sample slice indices [start_idx, end_idx) for a temporal window [start_time, end_time).
        """
        if start_time < 0 or end_time < start_time:
            raise ValueError(f"Invalid time window: [{start_time}, {end_time})")

        start_idx = self.time_to_sample_index(start_time)
        end_idx = int(math.ceil(end_time * self.sampling_rate_hz))

        if max_samples is not None:
            start_idx = min(start_idx, max_samples)
            end_idx = min(end_idx, max_samples)

        return start_idx, end_idx

    def validate_recording_alignment(
        self,
        stimulus_id: str,
        video_path: str,
        total_eye_samples: int
    ) -> Dict[str, Any]:
        """
        Validate synchronization between stimulus video and eye-tracking recording.
        """
        props = self.get_video_properties(video_path)
        fps = props["fps"]
        frame_count = props["frame_count"]
        video_duration = props["duration"]

        eye_duration = total_eye_samples / self.sampling_rate_hz

        effective_duration = min(video_duration, eye_duration)
        effective_frames = int(math.floor(effective_duration * fps))
        effective_samples = int(math.floor(effective_duration * self.sampling_rate_hz))

        t_zero = self.frame_to_time(0, fps)
        sample_zero = self.time_to_sample_index(t_zero)
        assert sample_zero == 0, f"Sample at t=0 must be 0, got {sample_zero}"

        t_last_frame = self.frame_to_time(effective_frames - 1, fps)
        sample_last_frame = self.time_to_sample_index(t_last_frame)
        assert sample_last_frame < total_eye_samples, (
            f"Last frame sample {sample_last_frame} exceeds total samples {total_eye_samples}"
        )

        return {
            "stimulus_id": stimulus_id,
            "fps": fps,
            "total_video_frames": frame_count,
            "video_duration_sec": round(video_duration, 4),
            "total_eye_samples": total_eye_samples,
            "eye_duration_sec": round(eye_duration, 4),
            "effective_duration_sec": round(effective_duration, 4),
            "effective_video_frames": effective_frames,
            "effective_eye_samples": effective_samples,
            "sampling_rate_hz": self.sampling_rate_hz,
            "synchronized": True
        }


if __name__ == "__main__":
    import json
    sync = VideoGazeSynchronizer()
    stim_manifest_path = r"E:\new viral\training\stimuli\stimulus_manifest.json"
    with open(stim_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    known_samples = {
        "task-stim01": 23662,
        "task-stim02": 18196,
        "task-stim03": 49770,
        "task-stim04": 47371,
        "task-stim05": 19232
    }

    print("--- Synchronization Validation ---")
    for item in manifest:
        stim_id = item["stimulus_id"]
        res = sync.validate_recording_alignment(
            stimulus_id=stim_id,
            video_path=item["local_path"],
            total_eye_samples=known_samples[stim_id]
        )
        print(f"[{stim_id}] FPS={res['fps']:.4f}, VideoDur={res['video_duration_sec']}s, "
              f"EyeDur={res['eye_duration_sec']}s, Effective={res['effective_duration_sec']}s -> OK")
