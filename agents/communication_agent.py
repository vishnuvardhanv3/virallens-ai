"""Communication Agent.

Specialist agent for spoken and rhetorical communication:
speech opening, direct address, questions, commands, claims, pauses,
verbal emphasis, and call-to-action (CTA) mechanics.
Associates transcript events with temporal attention predictions.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class CommunicationAgent:
    """Analyzes rhetorical syntax and aligns verbal signals with visual attention."""

    @staticmethod
    def analyze(
        technical_signals: Optional[Dict[str, Any]] = None,
        twelvelabs_analysis: Optional[Dict[str, Any]] = None,
        attention_predictions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Analyze verbal language and align phrases with attention dynamics."""
        technical_signals = technical_signals or {}
        twelvelabs_analysis = twelvelabs_analysis or {}
        attention_predictions = attention_predictions or []

        speech_sig = technical_signals.get("speech", {})
        has_speech = bool(speech_sig.get("has_speech"))
        transcript = str(speech_sig.get("speech_text") or "").strip()

        # Rhetorical syntax heuristics
        has_question = "?" in transcript or any(
            transcript.lower().startswith(q) for q in ["how", "why", "what", "did you", "have you", "can you"]
        )
        has_direct_address = bool(re.search(r"\b(you|your|you're|y'all)\b", transcript, re.IGNORECASE))
        has_cta = bool(re.search(r"\b(follow|subscribe|link in bio|comment|share|save this|click|check out)\b", transcript, re.IGNORECASE))
        has_numbers_claims = bool(re.search(r"\b(\d+|first|secret|rule|step|reason|mistake)\b", transcript, re.IGNORECASE))

        word_count = len(transcript.split()) if transcript else 0

        # Style classification
        if has_direct_address and has_question:
            style = "inquisitive_dialogic"
        elif has_numbers_claims:
            style = "structured_instructive"
        elif has_direct_address:
            style = "conversational_direct"
        elif has_speech:
            style = "narrative_monologue"
        else:
            style = "non_verbal_visual_ambient"

        # Align transcript presence with attention
        findings = []
        evidence = []

        if has_speech:
            findings.append(
                f"Transcript comprises {word_count} words categorized under '{style}' delivery style."
            )
            findings.append(
                f"Linguistic curiosity cues: direct address = {has_direct_address}, question syntax = {has_question}, actionable CTA = {has_cta}."
            )
            evidence.append(f"Spoken text transcript: '{transcript[:100]}...'" if len(transcript) > 100 else f"Spoken text transcript: '{transcript}'")
        else:
            findings.append("Absence of spoken dialogue; presentation relies purely on visual and musical cues.")
            evidence.append("No vocal speech track detected by audio transcription pipeline.")

        # Check attention around speech
        if attention_predictions and has_speech:
            speech_windows = [
                w for w in attention_predictions
                if w.get("feature_values", {}).get("audio_speech_presence", 0) > 0.3
            ]
            if speech_windows:
                mean_speech_att = sum(float(w["attention_score"]) for w in speech_windows) / len(speech_windows)
                evidence.append(
                    f"Temporal windows with active speech energy averaged {mean_speech_att:.2f} predicted visual-attention potential."
                )

        recommendations = []
        if not has_speech:
            recommendations.append(
                "Consider adding verbal narration or a spoken hook to engage auditory processing channels."
            )
        elif not has_direct_address:
            recommendations.append(
                "Incorporate direct second-person address ('you') in the opening to foster interpersonal salience."
            )
        elif not has_cta:
            recommendations.append(
                "Conclude with a clear, concise verbal call to action to guide audience next steps."
            )
        else:
            recommendations.append(
                "Verbal communication incorporates direct address and structured rhetoric; maintain current delivery cadence."
            )

        conf = 0.88 if has_speech else 0.75

        return {
            "agent": "communication",
            "findings": findings,
            "evidence": evidence,
            "confidence": conf,
            "uncertainties": [
                "Linguistic analysis is derived from automatic speech recognition (ASR); "
                "subtle colloquialisms, accents, or background music mixing may introduce transcription variance."
            ],
            "recommendations": recommendations,
            "has_speech": has_speech,
            "word_count": word_count,
            "communication_style": style,
            "has_question": has_question,
            "has_direct_address": has_direct_address,
            "has_cta": has_cta,
            "transcript_excerpt": transcript[:150],
        }
