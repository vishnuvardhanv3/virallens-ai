"""Human Visual Attention Agent.

Specialist agent for laboratory-derived visual attention potential analysis.
Analyzes temporal attention predictions from the NEMAR BBBD trained model
and correlates them with observable multimodal video events.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class HumanAttentionAgent:
    """Specialist agent analyzing predicted human visual-attention potential."""

    ATTENTION_EVENT_DELTA = 0.05
    MIN_EVENT_SCORE = 0.50

    @classmethod
    def analyze(
        cls,
        attention_predictions: List[Dict[str, Any]],
        twelvelabs_analysis: Optional[Dict[str, Any]] = None,
        technical_signals: Optional[Dict[str, Any]] = None,
        event_delta_threshold: float = 0.05,
        min_event_score: float = 0.50,
    ) -> Dict[str, Any]:
        """Analyze temporal attention predictions and connect to observable video features."""
        twelvelabs_analysis = twelvelabs_analysis or {}
        technical_signals = technical_signals or {}

        if not attention_predictions:
            return {
                "agent": "human_attention",
                "findings": ["[MODEL_DERIVED] No temporal attention predictions available for evaluation."],
                "evidence": [],
                "confidence": 0.0,
                "uncertainties": ["Inference did not yield valid temporal windows."],
                "recommendations": ["[RECOMMENDATION] Ensure video input contains valid decodable video and audio frames."],
                "first_attention_event": None,
                "first_attention_event_time": None,
                "first_attention_event_score": None,
                "first_attention_event_display": "No major attention event detected in the analyzed window",
                "opening_attention": None,
                "peak_attention": None,
                "peak_attention_time": None,
                "lowest_attention": None,
                "lowest_attention_time": None,
                "attention_stability": None,
                "temporal_consistency": None,
                "attention_curve": [],
                "timeline_events": [],
                "provenance_records": [],
                "generalization_warning": False,
            }

        # Step 1: Ensure strictly chronological order: start_time asc, then end_time asc
        sorted_predictions = sorted(
            attention_predictions,
            key=lambda x: (float(x.get("start_time", x.get("start", 0))), float(x.get("end_time", x.get("end", 0))))
        )

        scores = [float(p["attention_score"]) for p in sorted_predictions]
        confidences = [p.get("confidence") for p in sorted_predictions if p.get("confidence") is not None]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.75

        # Metric 1: Peak attention & relative peak prominence
        max_idx = int(np.argmax(scores))
        peak_score = round(scores[max_idx], 4)
        peak_time = round(float(sorted_predictions[max_idx]["start_time"]), 3)

        # Metric 2: Lowest attention
        min_idx = int(np.argmin(scores))
        lowest_score = round(scores[min_idx], 4)
        lowest_time = round(float(sorted_predictions[min_idx]["start_time"]), 3)

        # Metric 3: Early-video baseline attention (opening 2.0s or first 4 windows)
        baseline_slice = scores[: min(4, len(scores))]
        baseline_score = float(np.mean(baseline_slice)) if baseline_slice else 0.5
        opening_slice = scores[: min(2, len(scores))]
        opening_attention = round(float(np.mean(opening_slice)), 4) if opening_slice else round(scores[0], 4)

        # Metric 4: Stability & temporal consistency
        score_std = float(np.std(scores))
        score_mean = float(np.mean(scores)) if scores else 0.5
        # Formula: temporal consistency = max(0.0, 1.0 - 2.0 * std)
        stability = round(max(0.0, min(1.0, 1.0 - score_std * 2.0)), 4)
        cv = round(score_std / max(1e-4, score_mean), 4)

        # Low variance / generalization check
        low_variation_warning = bool(score_std < 0.015)

        # Connect temporal windows to observable stimuli events & inspect for first major attention event
        timeline_events = []
        provenance_records = []
        first_event_idx: Optional[int] = None
        first_event_stimulus: Optional[str] = None

        for idx, p in enumerate(sorted_predictions):
            start_t = round(float(p["start_time"]), 3)
            end_t = round(float(p["end_time"]), 3)
            score = round(float(p["attention_score"]), 4)
            feats = p.get("feature_values", {})

            # Identify observable stimulus features at this window
            mechanisms = []
            if feats.get("face_presence", 0) > 0.2:
                mechanisms.append("human face presence")
            if feats.get("text_presence", 0) > 0.1:
                mechanisms.append("on-screen textual graphics")
            if feats.get("scene_cut_rate", 0) > 0.5:
                mechanisms.append("rapid scene transition")
            if feats.get("motion_magnitude", 0) > 0.03:
                mechanisms.append("high visual motion onset")
            if feats.get("audio_speech_presence", 0) > 0.3:
                mechanisms.append("vocal speech energy")

            has_observable_stimulus = len(mechanisms) > 0
            mech_str = ", ".join(mechanisms) if mechanisms else "ambient visual continuity"

            # Check criteria for first major attention event:
            # Must require BOTH: meaningful attention change (delta >= threshold) AND observable stimulus
            delta_from_baseline = score - baseline_score
            prev_score = scores[idx - 1] if idx > 0 else baseline_score
            step_delta = score - prev_score

            if first_event_idx is None:
                is_meaningful_change = (
                    delta_from_baseline >= event_delta_threshold
                    or (score >= min_event_score and (step_delta >= 0.035 or (idx == 0 and score >= 0.60)))
                    or p.get("is_local_peak", False)
                )
                if is_meaningful_change and has_observable_stimulus:
                    first_event_idx = idx
                    first_event_stimulus = mech_str

            timeline_events.append({
                "timestamp": f"{start_t:.1f}s–{end_t:.1f}s",
                "start": start_t,
                "end": end_t,
                "start_time": start_t,
                "end_time": end_t,
                "attention_score": score,
                "confidence": p.get("confidence"),
                "major_observed_event": mech_str,
                "attention_mechanism": f"Temporal window characterized by {mech_str}.",
                "baseline_delta": round(delta_from_baseline, 4),
                "is_local_peak": p.get("is_local_peak", False),
                "peak_prominence": p.get("peak_prominence", 0.0),
            })

            if has_observable_stimulus:
                provenance_records.append({
                    "evidence_type": "OBSERVED",
                    "claim": f"Observable stimulus present at {start_t:.1f}s–{end_t:.1f}s: {mech_str}",
                    "source": "VideoFeatureExtractor",
                    "timestamp": start_t,
                })

        # Process first major event outcome
        if first_event_idx is not None:
            first_event_time = round(float(sorted_predictions[first_event_idx]["start_time"]), 3)
            first_event_score = round(scores[first_event_idx], 4)
            first_event_display = f"{first_event_time:.2f}s (Score: {first_event_score:.2f})"
        else:
            first_event_time = None
            first_event_score = None
            first_event_display = "No major attention event detected in the analyzed window"

        # Trajectory categorization
        if len(scores) >= 4:
            first_half = float(np.mean(scores[: len(scores) // 2]))
            second_half = float(np.mean(scores[len(scores) // 2 :]))
            if low_variation_warning:
                trajectory = "Uniform attention curve with low temporal dynamic range"
            elif first_half - second_half > 0.12:
                trajectory = "Front-loaded attention with gradual subsequent tapering"
            elif second_half - first_half > 0.12:
                trajectory = "Progressive visual attention escalation leading to late peak"
            elif score_std < 0.08:
                trajectory = "Low temporal variation in model output"
            else:
                trajectory = "Oscillating visual attention dynamics responding to scene edits"
        else:
            trajectory = "Short duration continuous attention segment"

        findings = []
        evidence = []

        # Finding 1: Peak attention
        findings.append(
            f"[MODEL_DERIVED] Laboratory-derived attention signal indicates peak predicted visual-attention potential of "
            f"{peak_score:.2f} at {peak_time:.2f}s."
        )
        provenance_records.append({
            "evidence_type": "MODEL_DERIVED",
            "claim": f"Peak visual-attention potential of {peak_score:.2f} at {peak_time:.2f}s",
            "source": "HumanAttentionService",
            "timestamp": peak_time,
        })

        # Finding 2: First major attention event
        if first_event_time is not None:
            findings.append(
                f"[MODEL_DERIVED] First major attention event emerged at {first_event_time:.2f}s "
                f"(score: {first_event_score:.2f}, stimulus: {first_event_stimulus})."
            )
            provenance_records.append({
                "evidence_type": "MODEL_DERIVED",
                "claim": f"First major attention event at {first_event_time:.2f}s aligned with {first_event_stimulus}",
                "source": "HumanAttentionAgent",
                "timestamp": first_event_time,
            })
        else:
            findings.append(
                "[MODEL_DERIVED] No major attention event detected in the analyzed window (attention delta did not exceed baseline threshold)."
            )

        # Finding 3: Stability & temporal consistency (Non-causal phrasing)
        if score_std < 0.08:
            findings.append(
                f"[MODEL_DERIVED] Low temporal variation in the model output (std: {score_std:.4f}); "
                "temporal variation is low, and temporal differences should be interpreted cautiously."
            )
        else:
            findings.append(
                f"[MODEL_DERIVED] Normalized curve stability measured at {stability:.2f} (std={score_std:.2f}), "
                f"indicating {trajectory.lower()}."
            )

        # Ensure timeline_events are strictly sorted by numeric start_time ascending
        timeline_events.sort(key=lambda x: (float(x.get("start_time", 0.0)), float(x.get("end_time", 0.0))))

        evidence.append(
            f"[OBSERVED] Peak score of {peak_score:.2f} at {peak_time:.2f}s coincides with "
            f"{timeline_events[max_idx]['major_observed_event']}."
        )
        evidence.append(
            f"[OBSERVED] Opening interval (0.0–1.0s) exhibited mean predicted visual-attention potential of {opening_attention:.2f}."
        )

        recommendations = []
        if first_event_time is not None and first_event_time > 1.5:
            rec = "[RECOMMENDATION] Introduce dynamic visual stimuli or face engagement earlier (within 0.5s) to advance the initial visual attention event."
            recommendations.append(rec)
            provenance_records.append({
                "evidence_type": "RECOMMENDATION",
                "claim": rec,
                "source": "HumanAttentionAgent",
                "timestamp": first_event_time,
            })
        elif first_event_time is None:
            rec = "[RECOMMENDATION] Front-load a prominent visual stimulus (motion contrast, text hook, or subject entrance) in the opening 1.0s to generate a distinct attention delta."
            recommendations.append(rec)
            provenance_records.append({
                "evidence_type": "RECOMMENDATION",
                "claim": rec,
                "source": "HumanAttentionAgent",
                "timestamp": 0.0,
            })

        if stability < 0.65 and not low_variation_warning:
            rec = "[RECOMMENDATION] Pacing fluctuations are pronounced; assess whether visual pacing can be stabilized through consistent textual framing or sustained focal subjects."
            recommendations.append(rec)
            provenance_records.append({
                "evidence_type": "RECOMMENDATION",
                "claim": rec,
                "source": "HumanAttentionAgent",
                "timestamp": None,
            })
        else:
            rec = "[RECOMMENDATION] Maintain current pacing continuity which produces stable predicted visual-attention potential."
            recommendations.append(rec)
            provenance_records.append({
                "evidence_type": "RECOMMENDATION",
                "claim": rec,
                "source": "HumanAttentionAgent",
                "timestamp": None,
            })

        return {
            "agent": "human_attention",
            "findings": findings,
            "evidence": evidence,
            "confidence": round(avg_confidence, 4),
            "uncertainties": [
                "Attention values reflect laboratory eye-tracking gaze potential from NEMAR BBBD benchmarks, "
                "not mobile touch scroll-stop behavior."
            ],
            "recommendations": recommendations,
            "first_attention_event": first_event_time,
            "first_attention_event_time": first_event_time,
            "first_attention_event_score": first_event_score,
            "first_attention_event_display": first_event_display,
            "opening_attention": opening_attention,
            "baseline_attention": round(baseline_score, 4),
            "peak_attention": peak_score,
            "peak_attention_time": peak_time,
            "lowest_attention": lowest_score,
            "lowest_attention_time": lowest_time,
            "attention_stability": stability,
            "temporal_consistency": stability,
            "coefficient_of_variation": cv,
            "generalization_warning": low_variation_warning,
            "trajectory": trajectory,
            "attention_curve": scores,
            "timeline_events": timeline_events,
            "provenance_records": provenance_records,
        }
