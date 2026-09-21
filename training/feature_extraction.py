"""
Video and Audio Feature Extraction Module for Human Visual Attention Model.

Extracts temporally aligned, interpretable features across:
- Visual: motion magnitude, frame difference, scene change rate, brightness, contrast, visual complexity (Laplacian variance), face presence.
- Editing: cut rate, mean shot duration, time since previous cut, transition rate.
- Audio: RMS energy, silence ratio, spectral centroid, speech presence.
- Text: text presence heuristic based on gradient density in caption/title regions.
"""

import math
import os
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from scipy.io import wavfile


class VideoFeatureExtractor:
    """
    Extracts multi-modal features per temporal window (default 0.5s).
    Uses fast sequential frame decoding and vectorized audio analysis.
    """

    def __init__(self, window_sec: float = 0.5):
        self.window_sec = float(window_sec)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_audio_wav(self, video_path: str, temp_wav_path: str) -> bool:
        """
        Extract mono 16kHz audio from MP4 using FFmpeg.
        """
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", temp_wav_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def load_audio_signal(self, video_path: str, audio_wav_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Extract and load audio samples and sampling rate.
        If audio_wav_path is provided and exists, reads directly without re-extracting.
        """
        if audio_wav_path and os.path.exists(audio_wav_path):
            try:
                sr, data = wavfile.read(audio_wav_path)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype != np.float32:
                    data = data.astype(np.float32)
                return data, sr
            except Exception:
                pass

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        success = self.extract_audio_wav(video_path, tmp_path)
        if not success or not os.path.exists(tmp_path):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return np.zeros(0, dtype=np.float32), 16000

        try:
            sr, data = wavfile.read(tmp_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype != np.float32:
                data = data.astype(np.float32)
            os.remove(tmp_path)
            return data, sr
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return np.zeros(0, dtype=np.float32), 16000

    def compute_audio_window_features(
        self,
        audio: np.ndarray,
        sr: int,
        start_sec: float,
        end_sec: float
    ) -> Dict[str, float]:
        """
        Calculate audio RMS, silence ratio, spectral centroid, and speech presence.
        """
        if len(audio) == 0 or sr <= 0:
            return {
                "audio_rms": 0.0,
                "audio_silence_ratio": 1.0,
                "audio_spectral_centroid": 0.0,
                "audio_speech_presence": 0.0
            }

        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        chunk = audio[start_idx:end_idx]

        if len(chunk) == 0:
            return {
                "audio_rms": 0.0,
                "audio_silence_ratio": 1.0,
                "audio_spectral_centroid": 0.0,
                "audio_speech_presence": 0.0
            }

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        silence_threshold = 0.015
        silence_ratio = float(np.mean(np.abs(chunk) < silence_threshold))

        # FFT for spectral centroid and speech band (300-3400 Hz)
        fft_vals = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)

        fft_sum = np.sum(fft_vals)
        if fft_sum > 1e-6:
            centroid = float(np.sum(freqs * fft_vals) / fft_sum)
            speech_mask = (freqs >= 300) & (freqs <= 3400)
            speech_energy = float(np.sum(fft_vals[speech_mask]) / fft_sum)
        else:
            centroid = 0.0
            speech_energy = 0.0

        return {
            "audio_rms": round(rms, 5),
            "audio_silence_ratio": round(silence_ratio, 4),
            "audio_spectral_centroid": round(centroid, 2),
            "audio_speech_presence": round(speech_energy, 4)
        }

    def extract_features(
        self,
        video_path: str,
        max_duration_sec: Optional[float] = None,
        audio_wav_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract multi-modal window features using fast sequential streaming.
        Can optionally use a pre-extracted audio WAV path to skip redundant audio demuxing.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        if max_duration_sec is not None:
            duration = min(duration, max_duration_sec)

        num_windows = int(math.floor(duration / self.window_sec))

        # Pre-load audio (reusing pre-extracted WAV if provided)
        audio, sr = self.load_audio_signal(video_path, audio_wav_path=audio_wav_path)

        # Buckets for window features
        window_buckets: List[Dict[str, list]] = [
            {
                "diffs": [],
                "brightness": [],
                "contrast": [],
                "complexity": [],
                "faces": [],
                "text_scores": [],
                "cuts": 0
            }
            for _ in range(num_windows)
        ]

        prev_gray = None
        last_cut_time = 0.0
        shot_start_time = 0.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t = frame_idx / fps
            if t >= duration:
                break

            w_idx = int(math.floor(t / self.window_sec))
            if w_idx >= num_windows:
                break

            # Downsample to 320x180 for efficient feature processing
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Brightness & Contrast
            b_val = float(np.mean(gray)) / 255.0
            c_val = float(np.std(gray)) / 255.0
            window_buckets[w_idx]["brightness"].append(b_val)
            window_buckets[w_idx]["contrast"].append(c_val)

            # Visual Complexity (Laplacian variance)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0
            window_buckets[w_idx]["complexity"].append(lap_var)

            # Motion & Cut Detection
            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, prev_gray))) / 255.0
                window_buckets[w_idx]["diffs"].append(diff)
                if diff > 0.32:  # Shot cut threshold
                    window_buckets[w_idx]["cuts"] += 1
                    last_cut_time = t
                    shot_start_time = t
            else:
                window_buckets[w_idx]["diffs"].append(0.0)

            # Face Detection (every 4th frame to remain real-time)
            if frame_idx % 4 == 0:
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=4, minSize=(28, 28)
                )
                window_buckets[w_idx]["faces"].append(1.0 if len(faces) > 0 else 0.0)

            # Text / Subtitle presence heuristic (lower third + center horizontal edge energy)
            lower_third = gray[115:170, :]
            sobel_x = cv2.Sobel(lower_third, cv2.CV_64F, 1, 0, ksize=3)
            edge_score = float(np.mean(np.abs(sobel_x))) / 255.0
            window_buckets[w_idx]["text_scores"].append(edge_score)

            prev_gray = gray
            frame_idx += 1

        cap.release()

        # Aggregate window metrics into feature dicts
        extracted = []
        current_cut_time = 0.0

        for w_idx in range(num_windows):
            w_start = w_idx * self.window_sec
            w_end = (w_idx + 1) * self.window_sec
            w_mid = (w_start + w_end) / 2.0
            b = window_buckets[w_idx]

            # Visual aggregations
            diffs = b["diffs"] if len(b["diffs"]) > 0 else [0.0]
            bright = b["brightness"] if len(b["brightness"]) > 0 else [0.5]
            cont = b["contrast"] if len(b["contrast"]) > 0 else [0.2]
            comp = b["complexity"] if len(b["complexity"]) > 0 else [0.1]
            faces = b["faces"] if len(b["faces"]) > 0 else [0.0]
            texts = b["text_scores"] if len(b["text_scores"]) > 0 else [0.0]

            if b["cuts"] > 0:
                current_cut_time = w_mid

            time_since_cut = max(0.0, w_mid - current_cut_time)

            # Audio features
            audio_feats = self.compute_audio_window_features(audio, sr, w_start, w_end)

            row = {
                "window_start": round(w_start, 3),
                "window_end": round(w_end, 3),
                "motion_magnitude": round(float(np.mean(diffs)), 5),
                "motion_std": round(float(np.std(diffs)), 5),
                "scene_cut_rate": round(b["cuts"] / self.window_sec, 2),
                "time_since_cut": round(time_since_cut, 2),
                "brightness_mean": round(float(np.mean(bright)), 4),
                "brightness_std": round(float(np.std(bright)), 4),
                "contrast": round(float(np.mean(cont)), 4),
                "visual_complexity": round(float(np.mean(comp)), 4),
                "face_presence": round(float(np.mean(faces)), 3),
                "text_presence": round(float(np.mean(texts)), 4),
                **audio_feats
            }
            extracted.append(row)

        return extracted
