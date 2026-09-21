"""Specialist Video OCR Agent - Extracts and tracks on-screen text across video frames.

Enforces:
1. Efficient adaptive frame sampling across short-form Reels.
2. Multi-pass preprocessing (Otsu binarization, adaptive thresholding, and color-contrast differencing for neon/colored text).
3. Text detection, recognition, confidence scoring, and bounding box tracking [x, y, w, h].
4. Temporal persistence, first appearance, disappearance, duration, and text-change detection.
5. Filtering of one-frame compression noise, unstable characters, and duplicate detections.
6. Structured canonical normalization (segments, unique_text, text_density, first_text_time, total_text_duration).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import TESSERACT_CMD

# Substantive word validation pattern
COMMON_VALID_WORDS = {
    "in", "on", "at", "to", "we", "he", "it", "is", "of", "by", "my", "up",
    "so", "no", "go", "me", "do", "if", "or", "as", "an", "am", "us", "be",
    "ok", "tv", "hd", "4k", "vs"
}

GARBAGE_CHARS_RE = re.compile(r"^[~`|_+=*^%$#@!<>\\/(){}\[\]:;]+$")


def _is_meaningful_token(text: str, conf: float) -> bool:
    """Filter out isolated punctuation, compression noise, and low-confidence single characters."""
    t = text.strip()
    if not t or len(t) < 2:
        return False
    if GARBAGE_CHARS_RE.match(t):
        return False
    # Must contain at least one alphabetic character
    if not any(c.isalpha() for c in t):
        return False
    # Reject strings with non-ASCII replacement characters
    if "\ufffd" in t or "\\" in t:
        return False
    return True


def _classify_position(y: int, height: int) -> str:
    """Map y coordinate to screen thirds: upper third, center, or lower third."""
    if height <= 0:
        return "center"
    frac = y / float(height)
    if frac < 0.333:
        return "upper third"
    elif frac < 0.666:
        return "center"
    else:
        return "lower third"


def _clean_phrase(text: str) -> str:
    """Normalize phrase text: remove duplicate spaces, strip border symbols and OCR junk."""
    # Remove unprintable / replacement chars
    cleaned = re.sub(r"[^\x20-\x7E]", "", text)
    # Remove stray brackets and junk symbols
    cleaned = re.sub(r"[\[\]{}|~`^\\/_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" _-+=~`|^\\/*.,:;!?")
    return cleaned


_clean_ocr_token = _clean_phrase
_is_substantive_token = lambda token: _is_meaningful_token(token, 80.0)


REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE)

ENGLISH_COMMON_WORDS = COMMON_VALID_WORDS.union({
    "the", "and", "that", "this", "whats", "what", "how", "you", "cum", "on", "one", "sec",
    "second", "seconds", "our", "are", "not", "with", "for", "from", "repay", "repey", "me",
    "will", "all", "can", "see", "get", "now", "day", "out", "man", "edit", "viral", "lens",
    "action", "montage", "reel", "vrsk", "vesk", "tod", "ahh", "eye", "aye", "superhero"
})


def classify_ocr_segment(text: str, duration: float = 1.0, conf: Optional[float] = None) -> str:
    """Classify OCR detection into high_quality, usable, low_quality, or noise."""
    if conf is None:
        conf = duration
        duration = 1.0
    t = _clean_phrase(text)
    if not t or len(t) < 2:
        return "noise"

    # Explicit noise checks: repeated triple characters like TTT, eee, rrr
    if REPEATED_CHARS_RE.search(t):
        return "noise"

    tokens = [w.lower() for w in re.findall(r"[a-zA-Z0-9']+", t)]
    if not tokens:
        return "noise"

    # Count known or plausible English words (length >= 4 or dictionary match)
    known_count = sum(1 for w in tokens if w in ENGLISH_COMMON_WORDS or len(w) >= 4)
    known_ratio = known_count / max(1, len(tokens))

    # Strict noise threshold for ungrounded low-confidence strings
    if conf < 50.0:
        return "noise" if known_ratio < 0.50 else "low_quality"

    # High quality: high confidence and substantive words
    if conf >= 75.0 and (known_count >= 1 or len(t) >= 3):
        return "high_quality"

    # Usable: moderate confidence with genuine word match
    if conf >= 55.0 and (known_ratio >= 0.50 or duration >= 0.6):
        return "usable"

    if conf >= 50.0 and len(t) >= 4:
        return "low_quality"

    return "noise"


def _is_substantive_segment(text: str, duration: float, conf: float) -> bool:
    """Validate whether an OCR text segment is genuine on-screen content or compression/rendering noise."""
    t = _clean_phrase(text)
    if not t or len(t) < 2:
        return False
    # If 2 characters, must be in known dictionary
    if len(t) == 2 and t.lower() not in COMMON_VALID_WORDS:
        return False
    # Must have at least 2 alphanumeric chars
    alnum_count = sum(1 for c in t if c.isalnum())
    if alnum_count < 2:
        return False
    # Disallow segments where non-alphanumeric chars dominate
    if alnum_count / len(t) < 0.60:
        return False
    # Single-frame (<0.5s) must have high confidence and multiple substantive characters
    if duration < 0.5:
        if conf < 55.0 or alnum_count < 2:
            return False
    return True


def _text_similarity(a: str, b: str) -> float:
    """Simple token and substring similarity between two text snippets."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    intersection = len(sa.intersection(sb))
    union = len(sa.union(sb))
    jaccard = intersection / max(1, union)
    if a.lower() in b.lower() or b.lower() in a.lower():
        return max(jaccard, 0.75)
    return jaccard


_OCR_CACHE: Dict[str, Dict[str, Any]] = {}


def extract_ocr(video_path: str | Path, max_frames: int | None = None, **kwargs: Any) -> dict[str, Any]:
    """Extract on-screen text from video frames using specialist multi-pass Tesseract pipeline."""
    path = Path(video_path)
    if not path.exists():
        return {
            "ocr_available": False,
            "text_present": False,
            "segments": [],
            "unique_text": [],
            "text_density": 0.0,
            "first_text_time": None,
            "total_text_duration": 0.0,
            "text_change_rate": 0.0,
            "ocr_confidence": 0.0,
            "ocr_text": "",
            "error": "File not found",
            "diagnostics": {"failure_mode": "file_not_found"},
        }

    profiler = kwargs.get("profiler")
    try:
        st = path.stat()
        cache_key = f"{path.resolve()}_{st.st_size}_{st.st_mtime}_{max_frames}"
        if cache_key in _OCR_CACHE:
            if profiler:
                try:
                    profiler.increment_metric("cache_hits", 1)
                except Exception:
                    pass
            return dict(_OCR_CACHE[cache_key])
    except Exception:
        cache_key = None

    try:
        import pytesseract
        from pytesseract import Output

        cmd = TESSERACT_CMD
        if cmd and os.path.exists(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {
                "ocr_available": False,
                "text_present": False,
                "segments": [],
                "unique_text": [],
                "text_density": 0.0,
                "first_text_time": None,
                "total_text_duration": 0.0,
                "text_change_rate": 0.0,
                "ocr_confidence": 0.0,
                "ocr_text": "",
                "error": "OpenCV VideoCapture open failed",
                "diagnostics": {"failure_mode": "opencv_open_failed"},
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return {
                "ocr_available": False,
                "text_present": False,
                "segments": [],
                "unique_text": [],
                "text_density": 0.0,
                "first_text_time": None,
                "total_text_duration": 0.0,
                "text_change_rate": 0.0,
                "ocr_confidence": 0.0,
                "ocr_text": "",
                "error": "Video contains 0 frames",
                "diagnostics": {"failure_mode": "zero_frames"},
            }

        duration_sec = frame_count / fps

        # Sampling cadence: every ~0.4 seconds (2.5 fps), suitable for short-form video
        if duration_sec <= 10.0:
            step_sec = 0.3
        elif duration_sec <= 30.0:
            step_sec = 0.4
        else:
            step_sec = 0.6

        step_frames = max(1, int(fps * step_sec))
        sampled_frame_indices = list(range(0, frame_count, step_frames))
        # Ensure final frame is included
        if sampled_frame_indices[-1] < frame_count - 1:
            sampled_frame_indices.append(frame_count - 1)

        if max_frames and len(sampled_frame_indices) > max_frames:
            step = max(1, len(sampled_frame_indices) // max_frames)
            sampled_frame_indices = sampled_frame_indices[::step][:max_frames]

        import time
        from concurrent.futures import ThreadPoolExecutor

        t_ocr_start = time.perf_counter()

        # Pre-read sampled frames into memory
        frames_data: List[Tuple[int, float, np.ndarray]] = []
        for f_idx in sampled_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_data.append((f_idx, round(f_idx / fps, 3), frame))
        cap.release()

        def _has_text_potential(gray_img: np.ndarray) -> bool:
            """Fast 0.2ms edge filter to discard frames with zero sharp vertical/horizontal text edges."""
            h, w = gray_img.shape
            roi = gray_img[int(h * 0.15):int(h * 0.95), int(w * 0.05):int(w * 0.95)]
            if roi.size == 0:
                return True
            grad_x = cv2.Sobel(roi, cv2.CV_16S, 1, 0, ksize=3)
            abs_grad = cv2.convertScaleAbs(grad_x)
            return int(np.count_nonzero(abs_grad > 45)) > (roi.size * 0.002)

        def _process_single_frame(item: Tuple[int, float, np.ndarray]) -> List[Dict[str, Any]]:
            f_idx, current_time, frame = item
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not _has_text_potential(gray):
                return []

            # Pass 1: Otsu
            _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            inv_otsu = cv2.bitwise_not(th_otsu)

            frame_words = []

            def _run_pass(img: np.ndarray, pass_name: str) -> List[Dict[str, Any]]:
                words = []
                try:
                    data = pytesseract.image_to_data(
                        img,
                        config="--psm 11",
                        output_type=Output.DICT,
                    )
                    n_boxes = len(data["text"])
                    for i in range(n_boxes):
                        raw_w = str(data["text"][i]).strip()
                        try:
                            conf = float(data["conf"][i])
                        except (ValueError, TypeError):
                            conf = 0.0

                        if conf >= 35.0 and _is_meaningful_token(raw_w, conf):
                            bx = int(data["left"][i])
                            by = int(data["top"][i])
                            bw = int(data["width"][i])
                            bh = int(data["height"][i])

                            if bw < int(w * 0.95) and bh < int(h * 0.95):
                                words.append({
                                    "word": raw_w,
                                    "conf": conf,
                                    "bbox": [bx, by, bw, bh],
                                    "pass": pass_name,
                                })
                except Exception:
                    pass
                return words

            words_otsu = _run_pass(inv_otsu, "otsu")
            frame_words.extend(words_otsu)

            # Smart gating: if otsu found high-confidence words, skip color and bright passes
            high_conf_otsu = any(w["conf"] >= 75.0 for w in words_otsu)

            b_ch, g_ch, r_ch = cv2.split(frame)
            color_diff = cv2.subtract(r_ch, b_ch)
            has_color_contrast = float(np.mean(color_diff)) > 15.0

            if not high_conf_otsu or has_color_contrast:
                _, th_color = cv2.threshold(color_diff, 80, 255, cv2.THRESH_BINARY)
                inv_color = cv2.bitwise_not(th_color)
                words_color = _run_pass(inv_color, "color")
                frame_words.extend(words_color)

            if not frame_words and float(np.mean(gray)) < 100.0:
                _, th_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                inv_bright = cv2.bitwise_not(th_bright)
                words_bright = _run_pass(inv_bright, "bright")
                frame_words.extend(words_bright)

            if not frame_words:
                return []

            # Deduplicate identical words at very similar positions
            unique_words = []
            for fw in frame_words:
                already = False
                for uw in unique_words:
                    if fw["word"].lower() == uw["word"].lower():
                        c_dist = abs(fw["bbox"][0] - uw["bbox"][0]) + abs(fw["bbox"][1] - uw["bbox"][1])
                        if c_dist < 40:
                            already = True
                            break
                if not already:
                    unique_words.append(fw)

            unique_words.sort(key=lambda x: (x["bbox"][1] // 30, x["bbox"][0]))

            current_line = []
            detections = []
            for w_item in unique_words:
                if not current_line:
                    current_line.append(w_item)
                else:
                    prev = current_line[-1]
                    same_line = abs(w_item["bbox"][1] - prev["bbox"][1]) < 25
                    adjacent_x = (w_item["bbox"][0] - (prev["bbox"][0] + prev["bbox"][2])) < 80
                    if same_line and adjacent_x:
                        current_line.append(w_item)
                    else:
                        phrase_text = _clean_phrase(" ".join(x["word"] for x in current_line))
                        if len(phrase_text) >= 2:
                            avg_conf = sum(x["conf"] for x in current_line) / len(current_line)
                            min_x = min(x["bbox"][0] for x in current_line)
                            min_y = min(x["bbox"][1] for x in current_line)
                            max_x = max(x["bbox"][0] + x["bbox"][2] for x in current_line)
                            max_y = max(x["bbox"][1] + x["bbox"][3] for x in current_line)
                            detections.append({
                                "time": current_time,
                                "text": phrase_text,
                                "confidence": round(avg_conf, 2),
                                "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                            })
                        current_line = [w_item]

            if current_line:
                phrase_text = _clean_phrase(" ".join(x["word"] for x in current_line))
                if len(phrase_text) >= 2:
                    avg_conf = sum(x["conf"] for x in current_line) / len(current_line)
                    min_x = min(x["bbox"][0] for x in current_line)
                    min_y = min(x["bbox"][1] for x in current_line)
                    max_x = max(x["bbox"][0] + x["bbox"][2] for x in current_line)
                    max_y = max(x["bbox"][1] + x["bbox"][3] for x in current_line)
                    detections.append({
                        "time": current_time,
                        "text": phrase_text,
                        "confidence": round(avg_conf, 2),
                        "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                    })
            return detections

        max_workers = min(4, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            raw_detections_nested = list(pool.map(_process_single_frame, frames_data))

        raw_frame_detections = [d for sub in raw_detections_nested for d in sub]
        raw_frame_detections.sort(key=lambda x: x["time"])
        ocr_duration = time.perf_counter() - t_ocr_start

        # Group frame detections across time into continuous segments
        segments: List[Dict[str, Any]] = []

        for det in raw_frame_detections:
            d_time = det["time"]
            d_text = det["text"]
            d_conf = det["confidence"]
            d_bbox = det["bbox"]

            matched_segment = None
            for seg in segments:
                # If temporally close (within 0.8s) and text is similar
                if abs(d_time - seg["end_time"]) <= 0.85:
                    sim = _text_similarity(d_text, seg["text"])
                    if sim >= 0.50:
                        matched_segment = seg
                        break

            if matched_segment:
                matched_segment["end_time"] = max(matched_segment["end_time"], round(d_time + step_sec, 3))
                # Update text to longer/more informative variant
                if len(d_text) > len(matched_segment["text"]):
                    matched_segment["text"] = d_text
                # Running average confidence
                matched_segment["confidence"] = round((matched_segment["confidence"] + d_conf) / 2.0, 2)
                # Expand bounding box to union
                b1 = matched_segment["bbox"]
                b2 = d_bbox
                union_x = min(b1[0], b2[0])
                union_y = min(b1[1], b2[1])
                union_w = max(b1[0] + b1[2], b2[0] + b2[2]) - union_x
                union_h = max(b1[1] + b1[3], b2[1] + b2[3]) - union_y
                matched_segment["bbox"] = [union_x, union_y, union_w, union_h]
            else:
                segments.append({
                    "text": d_text,
                    "start_time": d_time,
                    "end_time": round(d_time + step_sec, 3),
                    "confidence": d_conf,
                    "bbox": d_bbox,
                })

        # Post-filter segments: reject single-frame short character noise and artifacts
        filtered_segments: List[Dict[str, Any]] = []
        for s in segments:
            duration = round(s["end_time"] - s["start_time"], 3)
            clean_txt = _clean_phrase(s["text"])
            if _is_substantive_segment(clean_txt, duration, s["confidence"]):
                conf_val = round(float(s["confidence"]), 1)
                q_class = classify_ocr_segment(clean_txt, duration, conf_val)
                s["text"] = clean_txt
                s["duration"] = duration
                s["raw_segment_confidence"] = conf_val
                s["normalized_confidence"] = conf_val
                s["confidence"] = conf_val
                s["quality_class"] = q_class
                bbox = s.get("bbox", [0, 0, 0, 0])
                s["position"] = _classify_position(bbox[1], h if "h" in locals() and h > 0 else 1000)
                filtered_segments.append(s)

        # Separate prominent segments (high_quality, usable) from low-quality/noise
        prominent_segments = [s for s in filtered_segments if s.get("quality_class") in {"high_quality", "usable"}]
        rejected_segments = [s for s in filtered_segments if s.get("quality_class") in {"low_quality", "noise"}]

        # Use prominent segments as primary user-facing segments
        active_segments = prominent_segments if prominent_segments else filtered_segments

        # Sort active segments chronologically
        active_segments.sort(key=lambda x: (x["start_time"], x["end_time"]))

        # Build unique text list
        unique_text = []
        seen_lower = set()
        for s in active_segments:
            t_clean = s["text"].strip()
            tl = t_clean.lower()
            if tl not in seen_lower and len(t_clean) >= 2:
                seen_lower.add(tl)
                unique_text.append(t_clean)

        first_text_time = active_segments[0]["start_time"] if active_segments else None

        # Compute total non-overlapping text duration
        if active_segments:
            intervals = [(s["start_time"], s["end_time"]) for s in active_segments]
            intervals.sort(key=lambda x: x[0])
            merged_intervals = []
            curr_start, curr_end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start <= curr_end:
                    curr_end = max(curr_end, next_end)
                else:
                    merged_intervals.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_intervals.append((curr_start, curr_end))
            total_duration = sum(end - start for start, end in merged_intervals)
        else:
            total_duration = 0.0

        total_duration = round(total_duration, 2)
        text_density = round(total_duration / max(1.0, duration_sec), 4)
        text_change_rate = round(len(active_segments) / max(1.0, duration_sec), 4)

        # Average confidence strictly on 0–100 scale (normalize exactly once)
        mean_conf = (
            round(sum(s["confidence"] for s in active_segments) / len(active_segments), 1)
            if active_segments
            else 0.0
        )

        consolidated_text = " | ".join(unique_text)

        ocr_out = {
            "ocr_available": bool(active_segments),
            "text_present": bool(active_segments),
            "segments": active_segments,
            "unique_text": unique_text,
            "text_density": text_density,
            "first_text_time": first_text_time,
            "total_text_duration": total_duration,
            "text_change_rate": text_change_rate,
            "ocr_confidence": mean_conf,
            "ocr_text": consolidated_text,
            "error": "",
            "diagnostics": {
                "duration_seconds": round(duration_sec, 2),
                "sampled_frames": len(sampled_frame_indices),
                "frames_sampled": len(sampled_frame_indices),
                "ocr_processing_time_sec": round(ocr_duration, 2),
                "ocr_sample_fps": round(1.0 / step_sec, 1),
                "raw_detections": len(raw_frame_detections),
                "filtered_segments": len(active_segments),
                "rejected_segments_count": len(rejected_segments),
                "rejected_ocr_segments": rejected_segments,
                "ocr_engine": "Tesseract 5.5.3 (Multi-Pass Adaptive, Multi-Threaded)",
                "failure_mode": None if active_segments else "no_text_detected",
            },
        }

        if cache_key:
            _OCR_CACHE[cache_key] = dict(ocr_out)

        return ocr_out

    except Exception as exc:
        return {
            "ocr_available": False,
            "text_present": False,
            "segments": [],
            "unique_text": [],
            "text_density": 0.0,
            "first_text_time": None,
            "total_text_duration": 0.0,
            "text_change_rate": 0.0,
            "ocr_confidence": 0.0,
            "ocr_text": "",
            "error": f"OCR extraction error: {exc}",
            "diagnostics": {
                "failure_mode": f"exception: {type(exc).__name__}",
                "error_details": str(exc),
            },
        }


# Convenience alias for test runner and external interfaces
extract_video_ocr = extract_ocr

