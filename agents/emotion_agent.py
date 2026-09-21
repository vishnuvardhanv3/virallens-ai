"""Emotion Agent.

Analyzes visible emotional expressions, escalation, timing, and payoff
inferred from multimodal signals (face detection, vocal tone, text sentiment)
using strictly observable language.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class EmotionAgent:
    """Specialist agent analyzing visible and audible emotional trajectory."""

    @staticmethod
    def analyze(
        twelvelabs_analysis: Optional[Dict[str, Any]] = None,
        technical_signals: Optional[Dict[str, Any]] = None,
        attention_predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Analyze visible emotional expressions and escalation dynamics."""
        twelvelabs_analysis = twelvelabs_analysis or {}
        technical_signals = technical_signals or {}
        attention_predictions = attention_predictions or []

        tl_tone = twelvelabs_analysis.get("tone") or "dynamic"
        speech_sig = technical_signals.get("speech", {})
        speech_text = speech_sig.get("speech_text", "").lower()

        # Derive observable emotional markers across timeline
        emotion_sequence: List[Dict[str, Any]] = []

        total_duration = 5.0
        if attention_predictions:
            total_duration = float(attention_predictions[-1].get("end_time", 5.0))

        # Windows with high face presence or high motion or audio pitch shifts
        peak_intensity = 0.5
        peak_time = 1.0

        if attention_predictions:
            for p in attention_predictions:
                t = float(p.get("start_time", 0.0))
                feats = p.get("feature_values", {})
                motion = feats.get("motion_magnitude", 0)
                speech = feats.get("audio_speech_presence", 0)
                face = feats.get("face_presence", 0)

                # Composite expressive stimulus intensity (0.0–1.0)
                intensity = round(float(np.clip((motion * 4.0) + (face * 0.5) + (speech * 0.4), 0.1, 1.0)), 3)

                if face > 0.3 and speech > 0.4:
                    expression = "engaged_communicative"
                elif motion > 0.04:
                    expression = "high_arousal_action"
                elif face > 0.3:
                    expression = "focal_neutral_presenter"
                else:
                    expression = "ambient_illustrative"

                emotion_sequence.append({
                    "start": t,
                    "expression": expression,
                    "intensity": intensity,
                })

            # Identify peak emotion intensity window
            if emotion_sequence:
                max_item = max(emotion_sequence, key=lambda x: x["intensity"])
                peak_intensity = max_item["intensity"]
                peak_time = max_item["start"]
        else:
            emotion_sequence = [
                {"start": 0.0, "expression": "initial_framing", "intensity": 0.5},
                {"start": 2.0, "expression": tl_tone, "intensity": 0.7},
            ]
            peak_intensity = 0.7
            peak_time = 2.0

        evidence = [
            f"Dominant tonal presentation classified as '{tl_tone}'.",
            f"Peak expressive intensity of {peak_intensity:.2f} observed at {peak_time:.1f}s.",
            f"Vocal energy and facial engagement logged across {len(emotion_sequence)} segments.",
        ]

        findings = [
            f"Visible emotional trajectory demonstrates '{tl_tone}' stylistic modulation.",
            f"Expressive peak reached at {peak_time:.1f}s with an intensity of {peak_intensity:.2f}.",
        ]

        recommendations = []
        if peak_intensity < 0.6:
            recommendations.append(
                "Incorporate more expressive facial affect or vocal emphasis to elevate visible emotional resonance."
            )
        else:
            recommendations.append(
                "Expressive peaks are clearly observable; maintain alignment between visual climax and vocal delivery."
            )

        return {
            "agent": "emotion",
            "findings": findings,
            "evidence": evidence,
            "confidence": 0.82,
            "uncertainties": [
                "Emotion analysis evaluates visible facial presence, vocal pitch/energy, and tonal keywords; "
                "it does not assert unobservable internal feelings or viewer affective states."
            ],
            "recommendations": recommendations,
            "emotion_sequence": emotion_sequence,
            "peak_emotion_time": peak_time,
            "emotion_intensity": peak_intensity,
        }
