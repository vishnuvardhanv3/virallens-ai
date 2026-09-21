"""Hook Agent.

Specialist agent for the opening 1–3 seconds of an Instagram Reel.
Integrates the Human Visual Attention Model output, Twelve Labs multimodal signals,
and local visual/auditory evidence using relative temporal measures.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class HookAgent:
    """Evaluates opening hook efficacy without conflating attention with commercial retention."""

    @staticmethod
    def analyze(
        attention_predictions: List[Dict[str, Any]],
        twelvelabs_analysis: Optional[Dict[str, Any]] = None,
        technical_signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze opening hook (0.0–3.0s) combining attention model and multimodal signals."""
        twelvelabs_analysis = twelvelabs_analysis or {}
        technical_signals = technical_signals or {}

        # Isolate windows in opening 3 seconds (up to t <= 3.0s) strictly chronologically
        hook_windows = [
            p for p in sorted(
                attention_predictions,
                key=lambda x: (float(x.get("start_time", x.get("start", 0))), float(x.get("end_time", x.get("end", 0))))
            )
            if float(p.get("start_time", p.get("start", 0))) < 3.0
        ]

        if not hook_windows and attention_predictions:
            hook_windows = attention_predictions[: min(6, len(attention_predictions))]

        hook_scores = [float(w["attention_score"]) for w in hook_windows] if hook_windows else [0.5]
        mean_hook_attention = round(float(np.mean(hook_scores)), 4)
        peak_hook_score = round(float(np.max(hook_scores)), 4)

        # Baseline attention for opening
        baseline_score = float(hook_windows[0]["attention_score"]) if hook_windows else 0.5

        # Detect first attention event timing using RELATIVE measures rather than universal 0.70
        first_event_time: Optional[float] = None
        for idx, w in enumerate(hook_windows):
            score = float(w["attention_score"])
            feats = w.get("feature_values", {})
            has_stimulus = any([
                feats.get("face_presence", 0) > 0.2,
                feats.get("text_presence", 0) > 0.1,
                feats.get("scene_cut_rate", 0) > 0.5,
                feats.get("motion_magnitude", 0) > 0.03,
                feats.get("audio_speech_presence", 0) > 0.3,
            ])
            delta = score - baseline_score
            step_delta = score - hook_scores[idx - 1] if idx > 0 else 0.0
            is_meaningful = (
                delta >= 0.04
                or w.get("is_local_peak", False)
                or (score >= 0.60 and (idx == 0 or step_delta >= 0.035))
            )
            # Must require BOTH meaningful attention potential AND observable stimulus
            if is_meaningful and has_stimulus:
                first_event_time = round(float(w.get("start_time", w.get("start", 0))), 2)
                break

        # Gather multimodal evidence
        tl_hook = twelvelabs_analysis.get("hook", {})
        tl_hook_type = tl_hook.get("type") or "visual_dynamic"

        speech_sig = technical_signals.get("speech", {})
        has_speech = speech_sig.get("has_speech", False)
        speech_text = speech_sig.get("speech_text", "")

        ocr_sig = technical_signals.get("ocr", {})
        has_ocr = ocr_sig.get("ocr_available", False) and bool(ocr_sig.get("ocr_text"))
        ocr_text = ocr_sig.get("ocr_text", "")

        # Check feature presence in opening windows
        face_detected = any(w.get("feature_values", {}).get("face_presence", 0) > 0.2 for w in hook_windows)
        high_motion = any(w.get("feature_values", {}).get("motion_magnitude", 0) > 0.03 for w in hook_windows)
        rapid_cut = any(w.get("feature_values", {}).get("scene_cut_rate", 0) > 0.5 for w in hook_windows)

        # Determine refined hook category
        if face_detected and has_speech:
            hook_type = "direct_address_personal"
        elif high_motion or rapid_cut:
            hook_type = "rapid_visual_action"
        elif has_ocr and not has_speech:
            hook_type = "text_led_curiosity"
        elif face_detected:
            hook_type = "facial_engagement"
        else:
            hook_type = tl_hook_type

        # Build evidence list with provenance
        evidence = [
            f"[MODEL_DERIVED] Predicted visual-attention potential across first 3.0s averaged {mean_hook_attention:.2f} (peak: {peak_hook_score:.2f}).",
        ]
        if first_event_time is not None:
            evidence.append(f"[MODEL_DERIVED] First notable attention event identified at {first_event_time:.2f}s.")
        else:
            evidence.append("[MODEL_DERIVED] No distinct major attention event detected in the first 3.0s window.")

        if has_speech:
            evidence.append(f"[OBSERVED] Opening vocalization detected: '{speech_text[:60]}...'" if len(speech_text) > 60 else f"[OBSERVED] Opening vocalization: '{speech_text}'")
        if has_ocr:
            evidence.append(f"[OBSERVED] On-screen graphic text present in opening: '{ocr_text[:50]}'")
        if face_detected:
            evidence.append("[OBSERVED] Human face presence detected within opening 2.0s.")

        # Identify weaknesses
        weaknesses = []
        if mean_hook_attention < 0.55:
            weaknesses.append("Initial visual contrast and stimulus intensity remain low in opening 1.5 seconds.")
        if first_event_time is not None and first_event_time > 1.2:
            weaknesses.append(f"First significant attention event is delayed ({first_event_time:.2f}s into video).")
        elif first_event_time is None:
            weaknesses.append("No prominent visual stimulus onset observed in the opening 3.0s.")
        if not face_detected and not has_speech and not has_ocr:
            weaknesses.append("Absence of clear human presence, vocal hook, or prominent on-screen typography.")

        # Recommendation
        if weaknesses:
            rec = f"[RECOMMENDATION] Strengthen the opening 1.0s: {weaknesses[0]} Introduce immediate visual novelty or textual framing."
        else:
            rec = "[RECOMMENDATION] Opening visual pace and stimulus onset effectively align with early visual attention potential."

        return {
            "agent": "hook",
            "hook_type": hook_type,
            "attention_score": mean_hook_attention,
            "peak_attention_score": peak_hook_score,
            "first_attention_event": first_event_time,
            "first_attention_event_display": f"{first_event_time:.2f}s" if first_event_time is not None else "None detected",
            "findings": [
                f"[MODEL_DERIVED] Opening hook classified as '{hook_type}' with mean predicted visual-attention potential of {mean_hook_attention:.2f}.",
                f"[OBSERVED] Multimodal features: face={face_detected}, motion={high_motion}, cuts={rapid_cut}, speech={has_speech}, text={has_ocr}."
            ],
            "evidence": evidence,
            "confidence": 0.88,
            "uncertainties": [
                "Hook attention scores reflect laboratory visual saliency dynamics, not commercial mobile retention."
            ],
            "weaknesses": weaknesses,
            "recommendation": rec,
            "recommendations": [rec],
        }
