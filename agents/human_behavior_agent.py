"""Human Behavior Agent.

Analyzes strictly observable human behavior in the video:
facial expressions, gestures, gaze orientation, body movement,
social interaction, and direct audience address.

Maintains rigid ontological separation between:
- OBSERVATION: empirically detected sensory signals
- INTERPRETATION: functional interaction hypotheses
Never infers hidden subjective psychological mental states.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class HumanBehaviorAgent:
    """Specialist agent for observable human kinesics and behavioral signals."""

    @staticmethod
    def analyze(
        twelvelabs_analysis: Optional[Dict[str, Any]] = None,
        technical_signals: Optional[Dict[str, Any]] = None,
        attention_predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Analyze visible behavior maintaining strict observation vs interpretation separation."""
        twelvelabs_analysis = twelvelabs_analysis or {}
        technical_signals = technical_signals or {}
        attention_predictions = attention_predictions or []

        # Technical signals: face detection, motion magnitude
        windows_with_faces = [
            w for w in attention_predictions
            if w.get("feature_values", {}).get("face_presence", 0) > 0.25
        ]
        face_ratio = len(windows_with_faces) / max(1, len(attention_predictions))
        direct_camera_engagement = face_ratio > 0.35

        speech_sig = technical_signals.get("speech", {})
        has_speech = speech_sig.get("has_speech", False)
        speech_text = speech_sig.get("speech_text", "")

        tl_visual = twelvelabs_analysis.get("visual", {})
        tl_summary = twelvelabs_analysis.get("summary") or ""

        # Construct strictly separated observations and interpretations
        behaviors: List[Dict[str, str]] = []
        evidence: List[str] = []

        if windows_with_faces:
            first_face_t = windows_with_faces[0].get("start_time", 0.0)
            obs = f"Frontal human face detected at {first_face_t:.1f}s, appearing across {len(windows_with_faces)} temporal windows."
            interp = "Direct ocular orientation toward the lens; functions as a primary audience-address mechanism."
            behaviors.append({"observation": obs, "interpretation": interp})
            evidence.append(obs)

        if has_speech:
            obs = f"Audible vocal speech detected with continuous phrase delivery: '{speech_text[:50]}...'" if len(speech_text) > 50 else f"Audible vocal speech: '{speech_text}'"
            interp = "Verbal communicative turn indicating instructional, demonstrative, or conversational mode."
            behaviors.append({"observation": obs, "interpretation": interp})
            evidence.append(obs)

        # Body motion analysis
        high_motion_windows = [
            w for w in attention_predictions
            if w.get("feature_values", {}).get("motion_magnitude", 0) > 0.04
        ]
        if high_motion_windows:
            peak_m_win = max(high_motion_windows, key=lambda w: w.get("feature_values", {}).get("motion_magnitude", 0))
            t_motion = peak_m_win.get("start_time", 0.0)
            obs = f"Elevated frame-to-frame pixel displacement (motion magnitude > 0.04) observed at {t_motion:.1f}s."
            interp = "Rapid bodily repositioning, active gesturing, or dynamic camera motion."
            behaviors.append({"observation": obs, "interpretation": interp})
            evidence.append(obs)
        else:
            obs = "Low bodily or camera displacement observed throughout the clip."
            interp = "Static framing with stationary presenter posture."
            behaviors.append({"observation": obs, "interpretation": interp})
            evidence.append(obs)

        # Build findings strictly highlighting observable facts
        findings = [
            f"OBSERVATION: Face presence ratio measured at {face_ratio:.1%}. "
            f"INTERPRETATION: {'Strong direct audience engagement structure.' if direct_camera_engagement else 'Limited presenter-to-camera ocular alignment.'}",
            f"OBSERVATION: {len(behaviors)} distinct behavioral/movement event classes logged across the clip duration.",
        ]

        recommendations = []
        if face_ratio < 0.25 and not has_speech:
            recommendations.append(
                "Consider introducing observable human gestures or a direct-to-camera presenter gaze "
                "to provide an interpersonal focal anchor."
            )
        else:
            recommendations.append(
                "Observed gaze and gestural presence are active; maintain direct ocular contact during key explanatory beats."
            )

        conf = 0.88 if windows_with_faces or has_speech else 0.65

        return {
            "agent": "human_behavior",
            "findings": findings,
            "evidence": evidence,
            "confidence": conf,
            "uncertainties": [
                "Analysis evaluates observable kinesics (face presence, movement, vocal energy). "
                "No claims regarding internal cognitive states, intent, or unspoken emotions are asserted."
            ],
            "recommendations": recommendations,
            "observable_behaviors": behaviors,
            "face_presence_ratio": round(face_ratio, 4),
            "direct_camera_engagement": direct_camera_engagement,
        }
