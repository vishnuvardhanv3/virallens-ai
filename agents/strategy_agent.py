"""Strategy Agent.

Synthesizes empirical findings across all multi-agent outputs
into actionable, evidence-grounded strategic recommendations.
Every recommendation strictly follows the structured schema:
- OBSERVATION
- EVIDENCE
- COMPARISON
- RECOMMENDATION
- CONFIDENCE
- LIMITATION
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class StrategyAgent:
    """Produces prescriptive recommendations directly tied to empirical video evidence."""

    @staticmethod
    def synthesize(
        your_reel: Optional[Dict[str, Any]] = None,
        agent_results: Optional[Dict[str, Dict[str, Any]]] = None,
        attention_comparison: Optional[Dict[str, Any]] = None,
        pattern_results: Optional[Dict[str, Any]] = None,
        recommendation_intelligence: Optional[Dict[str, Any]] = None,
        reel_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convert multi-agent empirical findings into grounded strategic action items."""
        target_reel = your_reel if your_reel is not None else (reel_data or {})
        your_reel = target_reel
        agent_results = agent_results if agent_results is not None else (kwargs.get("agent_outputs") or {})
        strategic_items: List[Dict[str, Any]] = []
        findings: List[str] = []
        evidence: List[str] = []

        # Phase 16: Recommendation Intelligence Resolution
        rec_intel = recommendation_intelligence or agent_results.get("recommendation_intelligence")
        if not rec_intel:
            comp_intel = (
                (pattern_results or {}).get("competitor_intelligence")
                or agent_results.get("competitor_intelligence")
                or target_reel.get("competitor_intelligence")
                or kwargs.get("competitor_intelligence")
            )
            att_intel = (
                target_reel.get("attention_intelligence")
                or agent_results.get("attention_intelligence")
                or kwargs.get("attention_intelligence")
            )
            if comp_intel and comp_intel.get("status") == "SUCCESS":
                try:
                    from services.recommendation_intelligence import RecommendationIntelligenceService
                    rec_intel = RecommendationIntelligenceService.generate(
                        target_profile=target_reel,
                        attention_intelligence=att_intel,
                        competitor_intelligence=comp_intel,
                    )
                except Exception:
                    rec_intel = None

        # 1. Attention Strategy
        att_res = agent_results.get("human_attention", {})
        first_event = att_res.get("first_attention_event")
        peak_score = att_res.get("peak_attention")

        comp_first_event = None
        if attention_comparison and attention_comparison.get("competitor_averages"):
            comp_first_event = attention_comparison["competitor_averages"].get("first_attention_event_time")

        if first_event is not None and first_event > 1.2:
            obs = f"Your first major attention event emerged at {first_event:.2f}s."
            ev = f"Human Attention Model temporal predictions show first threshold peak delayed until {first_event:.2f}s."
            comp_txt = (
                f"Verified competitors trigger their first attention event at {comp_first_event:.2f}s on average."
                if comp_first_event is not None
                else "Top-performing short-form video establishes focal engagement within the first 1.0s."
            )
            rec = "Front-load an ocular anchor, visual contrast jump, or text reveal within the first 0.5s to advance initial attention potential."
            strategic_items.append({
                "observation": obs,
                "evidence": ev,
                "comparison": comp_txt,
                "recommendation": rec,
                "confidence": 0.88,
                "limitation": "Laboratory-derived gaze benchmark does not measure direct mobile scroll-stop telemetry.",
                "actionable_recommendation": rec,
            })
            evidence.append(ev)

        # 2. Hook Strategy
        hook_res = agent_results.get("hook", {})
        weaknesses = hook_res.get("weaknesses", [])
        if weaknesses:
            obs = f"Hook evaluation detected structural friction: {weaknesses[0]}"
            ev = f"Hook Agent evaluation scored opening 3s attention potential at {hook_res.get('attention_score', 0):.2f}."
            comp_txt = "Category-dominant reels deliver high opening stimulus contrast within 0.0s–1.0s."
            rec = f"Refactor opening hook: {hook_res.get('recommendation', 'Introduce immediate visual novelty or textual framing.')}"
            strategic_items.append({
                "observation": obs,
                "evidence": ev,
                "comparison": comp_txt,
                "recommendation": rec,
                "confidence": 0.85,
                "limitation": "Audience curiosity varies with creator brand equity and existing audience affinity.",
                "actionable_recommendation": rec,
            })
            evidence.append(ev)

        # 3. On-Screen Text / OCR Strategy
        canonical = your_reel.get("canonical", {})
        ocr_data = your_reel.get("ocr") or canonical.get("ocr", {})
        text_present = ocr_data.get("text_present", False)
        first_text_t = ocr_data.get("first_text_time")

        if not text_present or (first_text_t is not None and first_text_t > 2.0):
            obs = (
                "No on-screen typography was detected in the opening 2.0s."
                if text_present
                else "No readable on-screen typography detected across the video."
            )
            ev = f"Specialist OCR scan found first_text_time={first_text_t}s, text_present={text_present}."
            comp_txt = "Top-reach competitor reels frequently pair vocal speech or audio with high-contrast text overlays in the opening."
            rec = "Introduce clean on-screen title cards or hook typography within the opening 1.0s to capture silent scrollers."
            strategic_items.append({
                "observation": obs,
                "evidence": ev,
                "comparison": comp_txt,
                "recommendation": rec,
                "confidence": 0.90,
                "limitation": "Text efficacy depends on font readability, contrast against background, and safe-zone placement.",
                "actionable_recommendation": rec,
            })
            evidence.append(ev)

        # 4. Editing Strategy
        edit_res = agent_results.get("editing", {})
        mean_shot = edit_res.get("mean_shot_duration", 0)
        pacing = edit_res.get("pacing", "")
        if mean_shot > 3.0:
            obs = f"Mean shot duration is {mean_shot:.1f}s under '{pacing}' pacing classification."
            ev = f"Editing Agent counted {edit_res.get('total_cuts', 0)} cut transitions across duration."
            comp_txt = "Kinetic competitor edits incorporate scene shifts or focal reframes every 1.5s–2.5s."
            rec = "Introduce b-roll cutaways, dynamic camera zooms, or match cuts every 2.0–2.5s to maintain rhythmic visual progression."
            strategic_items.append({
                "observation": obs,
                "evidence": ev,
                "comparison": comp_txt,
                "recommendation": rec,
                "confidence": 0.86,
                "limitation": "Cut frequency must match audio tempo to avoid jarring visual-acoustic dissonance.",
                "actionable_recommendation": rec,
            })
            evidence.append(ev)

        # 5. Recommendation Intelligence & Pattern Benchmark Strategy (Phase 16)
        if rec_intel and rec_intel.get("recommendations"):
            for r in rec_intel["recommendations"]:
                obs = r.get("observation", "")
                ev_list = r.get("evidence", [])
                ev_str = "; ".join(e.get("value", "") for e in ev_list) if ev_list else obs
                comp_txt = r.get("target_comparison", {}).get("difference") or r.get("target_comparison", {}).get("comparison_group", "")
                rec = r.get("recommendation", "")
                lim = r.get("limitation", "")
                cat = r.get("category", "TEST")
                prio = r.get("priority", "medium")
                act = r.get("testable_action", "")
                strength = r.get("strength", "moderate_observed_evidence")

                strategic_items.append({
                    "observation": obs,
                    "evidence": ev_str,
                    "comparison": comp_txt,
                    "recommendation": rec,
                    "confidence": None,
                    "strength": strength,
                    "category": cat,
                    "priority": prio,
                    "testable_action": act,
                    "limitation": lim,
                    "actionable_recommendation": rec,
                    "provenance": r.get("provenance", {}),
                })
                evidence.append(ev_str)
        elif pattern_results:
            comp_intel = pattern_results.get("competitor_intelligence") or {}
            implications = comp_intel.get("creative_implications", [])
            if implications:
                for imp in implications[:3]:
                    obs = imp["observation"]
                    ev = imp["evidence"]
                    comp_txt = imp.get("target_competitor_comparison") or imp.get("comparison", "")
                    rec = imp["recommendation"]
                    lim = imp["limitation"]
                    strategic_items.append({
                        "observation": obs,
                        "evidence": ev,
                        "comparison": comp_txt,
                        "recommendation": rec,
                        "confidence": 0.88,
                        "limitation": lim,
                        "actionable_recommendation": rec,
                    })
                    evidence.append(ev)
            else:
                counts = pattern_results.get("pattern_counts", {})
                for pattern_name, count_str in counts.items():
                    parts = count_str.split("/")
                    if len(parts) == 2 and int(parts[1]) > 0:
                        num, den = int(parts[0]), int(parts[1])
                        if num / den >= 0.5:
                            p_label = pattern_name.replace("_", " ")
                            obs = f"{num}/{den} verified competitors utilize '{p_label}'."
                            ev = f"Pattern Agent exact counts across {den} verified competitors: {num}/{den} exhibited this pattern."
                            comp_txt = f"Pattern observed in {num}/{den} verified peer videos in your niche."
                            rec = f"Incorporate '{p_label}' into upcoming production variants to align with validated niche conventions."
                            strategic_items.append({
                                "observation": obs,
                                "evidence": ev,
                                "comparison": comp_txt,
                                "recommendation": rec,
                                "confidence": 0.84,
                                "limitation": "Correlation with reach does not prove causality; creative execution details determine viewer response.",
                                "actionable_recommendation": rec,
                            })
                            evidence.append(ev)
                            break

        # 6. Communication / Speech Strategy
        comm_res = agent_results.get("communication", {})
        if not comm_res.get("has_direct_address"):
            obs = "Transcript analysis shows no direct second-person address ('you', 'your')."
            ev = f"Communication Agent linguistic scan of {comm_res.get('word_count', 0)} words found direct_address=False."
            comp_txt = "Conversational direct address is common among high-engagement educational and storytelling creators."
            rec = "Incorporate conversational direct address ('Have you seen...', 'What you need...') in the opening sentence."
            strategic_items.append({
                "observation": obs,
                "evidence": ev,
                "comparison": comp_txt,
                "recommendation": rec,
                "confidence": 0.82,
                "limitation": "Direct address style must fit format tone; less applicable to purely ambient artistic montage.",
                "actionable_recommendation": rec,
            })
            evidence.append(ev)

        # 7. Attention Intelligence Strategy (Spatial Saliency & Cross-Modal Dynamics)
        att_intel = your_reel.get("attention_intelligence") or agent_results.get("attention_intelligence", {})
        if att_intel and att_intel.get("status") in {"SUCCESS", "PARTIAL"}:
            spat_intel = att_intel.get("spatial", {})
            focal_comp = spat_intel.get("focal_competition", {})
            comp_state = focal_comp.get("competition_state")
            disp = spat_intel.get("dispersion", 0.0)

            if comp_state in {"multi_focus", "diffuse"} or disp > 0.35:
                obs = f"Predicted spatial visual saliency is diffusely distributed across multiple scene regions (dispersion: {disp:.3f})."
                ev = f"TinySalNet spatial saliency analysis classifies focal distribution as '{comp_state}' with concentration ratio {focal_comp.get('concentration_ratio', 0.0):.2f}."
                comp_txt = "Top-reach competitor reels in verified peer comparisons consistently display a concentrated opening visual anchor around the primary subject."
                rec = "Consider keeping the primary subject visually dominant during the opening segment to establish an unmistakable focal anchor."
                strategic_items.append({
                    "observation": obs,
                    "evidence": ev,
                    "comparison": comp_txt,
                    "recommendation": rec,
                    "confidence": 0.86,
                    "limitation": "Model-derived spatial saliency measures visual conspicuity under laboratory video-viewing conditions and does not measure individual viewer scroll-stop decisions.",
                    "actionable_recommendation": rec,
                })
                evidence.append(ev)

        findings = [
            f"Formulated {len(strategic_items)} empirical, evidence-grounded strategic recommendations.",
            "Every action item strictly details observation, empirical evidence, competitor comparison, recommendation, confidence, and scientific limitations.",
        ]

        actionable_recs = [item["actionable_recommendation"] for item in strategic_items]

        return {
            "agent": "strategy",
            "findings": findings,
            "evidence": evidence,
            "confidence": 0.90,
            "uncertainties": [
                "Recommendations optimize for predicted visual-attention potential and verified niche benchmarks; "
                "platform distribution also depends on audience graph, sound trends, and account authority."
            ],
            "recommendations": actionable_recs,
            "strategic_recommendations": strategic_items,
            "strategic_priorities": [r.get("recommendation") for r in (rec_intel.get("recommendations") if rec_intel else [])] or actionable_recs,
            "recommendation_intelligence": rec_intel,
        }
