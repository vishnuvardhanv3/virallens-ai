"""Editing Agent.

Specialist agent for video editing syntax: cut rate, shot duration, transitions,
pacing, and audio-visual synchronization.
Compares editing events with predicted visual-attention trajectory
without asserting causality.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class EditingAgent:
    """Analyzes editing tempo and temporal alignment with attention potential."""

    @staticmethod
    def analyze(
        technical_signals: Optional[Dict[str, Any]] = None,
        attention_predictions: Optional[List[Dict[str, Any]]] = None,
        twelvelabs_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Examine shot cuts, duration, and alignment with attention dynamics."""
        technical_signals = technical_signals or {}
        attention_predictions = attention_predictions or []
        twelvelabs_analysis = twelvelabs_analysis or {}

        video_tech = technical_signals.get("video", {})
        duration_sec = float(video_tech.get("duration_sec") or 5.0)
        scene_change_count = int(video_tech.get("scene_change_count") or 0)

        # Shot cuts from window features
        cut_windows = []
        for w in attention_predictions:
            if w.get("feature_values", {}).get("scene_cut_rate", 0) > 0.4:
                cut_windows.append(w)

        total_cuts = max(scene_change_count, len(cut_windows))
        cut_rate = round(total_cuts / max(1.0, duration_sec), 2)
        mean_shot_duration = round(duration_sec / max(1, total_cuts + 1), 2)

        # Categorize pacing
        if cut_rate > 1.2:
            pacing = "kinetic_rapid_fire"
        elif cut_rate >= 0.5:
            pacing = "dynamic_rhythmic"
        elif cut_rate >= 0.2:
            pacing = "moderate_narrative"
        else:
            pacing = "deliberate_continuous"

        # Compare editing events with attention peaks (non-causal alignment)
        aligned_events: List[Dict[str, Any]] = []
        if attention_predictions:
            scores = [float(w["attention_score"]) for w in attention_predictions]
            peak_idx = int(np.argmax(scores))
            peak_time = float(attention_predictions[peak_idx].get("start_time", 0.0))
            peak_score = scores[peak_idx]

            # Find nearest cut
            nearest_cut_time = None
            min_dist = 999.0
            for cw in cut_windows:
                ct = float(cw.get("start_time", 0.0))
                dist = abs(ct - peak_time)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cut_time = ct

            if nearest_cut_time is not None and min_dist <= 1.0:
                aligned_events.append({
                    "attention_peak_time": peak_time,
                    "attention_peak_score": peak_score,
                    "nearest_cut_time": nearest_cut_time,
                    "temporal_offset_sec": round(min_dist, 2),
                    "relationship": "Visual cut temporally coincided with attention peak (non-causal correlation).",
                })
            else:
                aligned_events.append({
                    "attention_peak_time": peak_time,
                    "attention_peak_score": peak_score,
                    "nearest_cut_time": nearest_cut_time,
                    "temporal_offset_sec": round(min_dist, 2) if nearest_cut_time else None,
                    "relationship": "Attention peak emerged during internal shot motion rather than at cut boundary.",
                })

        findings = [
            f"Video demonstrates {total_cuts} detected cuts across {duration_sec:.1f}s (cut rate: {cut_rate:.2f} cuts/sec, mean shot duration: {mean_shot_duration:.1f}s).",
            f"Overall visual editing rhythm categorized as '{pacing}'.",
        ]

        evidence = [
            f"Observed cut rate of {cut_rate:.2f} cuts/sec across {duration_sec:.1f} seconds.",
            f"Pacing classification '{pacing}' with mean shot duration of {mean_shot_duration:.1f}s.",
        ]
        if aligned_events:
            evidence.append(aligned_events[0]["relationship"])

        recommendations = []
        if mean_shot_duration > 4.0:
            recommendations.append(
                "Mean shot duration exceeds 4.0s; test introducing b-roll cutaways or camera angle changes "
                "to modulate visual pacing."
            )
        elif cut_rate > 1.5:
            recommendations.append(
                "Cut rate is very rapid (>1.5 cuts/sec); ensure visual focal points remain legible during key information delivery."
            )
        else:
            recommendations.append(
                "Editing rhythm is well-balanced with attention trajectory; maintain current shot change frequency."
            )

        return {
            "agent": "editing",
            "findings": findings,
            "evidence": evidence,
            "confidence": 0.86,
            "uncertainties": [
                "Editing metrics track frame-level optical differences and cut boundaries; "
                "temporal alignments with attention peaks denote empirical co-occurrence rather than direct causation."
            ],
            "recommendations": recommendations,
            "cut_rate": cut_rate,
            "total_cuts": total_cuts,
            "mean_shot_duration": mean_shot_duration,
            "pacing": pacing,
            "aligned_events": aligned_events,
        }
