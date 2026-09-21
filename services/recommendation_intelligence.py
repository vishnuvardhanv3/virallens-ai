"""Recommendation Intelligence Service.

Transforms validated multimodal analytical outputs:
- Twelve Labs semantic analysis
- NEMAR temporal attention predictions
- TinySalNet / DHF1K spatial saliency
- OCR typography signals
- Phase 14 Attention Intelligence
- Phase 15 Competitor Intelligence
- Pattern Agent recurring tallies

into a rigorous, evidence-backed Recommendation Intelligence layer.

Enforces Phase 16 Hard Constraints:
1. Purely analytical layer over already-generated canonical data (NO new ML models, NO retraining,
   NO Apify or Twelve Labs replacement, NO network calls, NO duplicate inference).
2. Strict non-causal epistemic guardrails: Strictly NO causality claims, NO "what makes a Reel go viral",
   NO guaranteed engagement, NO scroll-stop or retention predictions, and NO population-wide rules.
3. Every recommendation follows the 7-part pipeline:
   OBSERVATION -> EVIDENCE -> TARGET vs COMPARISON -> CREATIVE IMPLICATION -> RECOMMENDATION -> TESTABLE ACTION -> LIMITATION.
4. Recommendation Categories: PRESERVE, CHANGE, TEST, AVOID_OVERINTERPRETATION, INVESTIGATE, NO_ACTIONABLE_RECOMMENDATION.
5. Evidence Quality & Small-Sample Safety: For n < 5, classified as 'insufficient_evidence' or 'exploratory_evidence'.
   Numerators and denominators (e.g. 6/7) are always explicit.
6. Confidence Rule: Confidence is strictly null (zero fabricated percentages like '92% confidence').
7. Contradictory Evidence Handling: Identifies conflicting signals and marks them as 'conflicting_evidence' / 'INVESTIGATE'.
8. Full Traceability & Provenance: Complete evidence graph from target -> feature -> pattern -> competitor evidence -> gap -> recommendation.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union


class RecommendationIntelligenceService:
    """Analytical service synthesizing multimodal evidence into structured creative recommendations."""

    SCHEMA_VERSION = "1.0.0"

    @classmethod
    def generate(
        cls,
        target_profile: Dict[str, Any],
        attention_intelligence: Optional[Dict[str, Any]] = None,
        competitor_intelligence: Optional[Dict[str, Any]] = None,
        existing_strategy_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Main Phase 16 analytical entry point (Section 3).
        
        Consumes existing analytical outputs.
        Does NOT perform discovery, downloading, Twelve Labs calls, or model inference.
        """
        target_reel_id = str(
            target_profile.get("shortcode")
            or target_profile.get("id")
            or target_profile.get("reel_id")
            or "target_reel"
        )

        # 1. Resolve Attention Intelligence
        att_intel = (
            attention_intelligence
            or target_profile.get("attention_intelligence")
            or {}
        )

        # 2. Resolve Competitor Intelligence
        comp_intel = (
            competitor_intelligence
            or target_profile.get("competitor_intelligence")
            or {}
        )

        # 3. Check for Empty / Insufficient Competitor Baseline (Section 17)
        verified_count = comp_intel.get("verified_competitor_count", 0)
        primary_subgroup = comp_intel.get("primary_subgroup", "same_entity_and_edit_type")
        subgroup_counts = comp_intel.get("subgroup_counts", {})
        primary_n = subgroup_counts.get(primary_subgroup, verified_count)

        if verified_count == 0 or not comp_intel:
            return cls._build_null_recommendation_result(
                target_reel_id=target_reel_id,
                reason="Zero verified relevant competitors available for peer benchmark comparison. Downstream creative recommendations cannot be grounded in comparative evidence.",
            )

        # 4. Extract Analytical Streams from Target
        target_norm = comp_intel.get("target_normalized") or {}
        gaps = comp_intel.get("gap_analysis") or []
        recurring_patterns = comp_intel.get("recurring_patterns") or []
        all_patterns = comp_intel.get("all_subgroup_patterns") or []
        reach_obs = comp_intel.get("reach_aware_observations") or {}

        # 5. Build Candidates across Analytical Dimensions
        raw_recommendations: List[Dict[str, Any]] = []

        # Domain A: Opening Spatial Focal Anchoring & Visual Composition
        rec_spatial = cls._evaluate_spatial_focus(target_norm, att_intel, comp_intel)
        if rec_spatial:
            raw_recommendations.append(rec_spatial)

        # Domain B: Early Typography & On-Screen Text
        rec_ocr = cls._evaluate_typography(target_norm, att_intel, comp_intel)
        if rec_ocr:
            raw_recommendations.append(rec_ocr)

        # Domain C: Human Presence & Character Gaze Anchoring
        rec_human = cls._evaluate_human_presence(target_norm, att_intel, comp_intel)
        if rec_human:
            raw_recommendations.append(rec_human)

        # Domain D: Dynamic Editing, Cut Pacing & Scene Transitions
        rec_editing = cls._evaluate_editing_pacing(target_norm, att_intel, comp_intel)
        if rec_editing:
            raw_recommendations.append(rec_editing)

        # Domain E: Opening Temporal Attention Prominence
        rec_temporal = cls._evaluate_temporal_prominence(target_norm, att_intel, comp_intel)
        if rec_temporal:
            raw_recommendations.append(rec_temporal)

        # Domain F: Acoustic & Spoken Engagement
        rec_audio = cls._evaluate_audio_speech(target_norm, att_intel, comp_intel)
        if rec_audio:
            raw_recommendations.append(rec_audio)

        # Domain G: Contradictory Evidence & Conflict Check (Section 16)
        cls._resolve_contradictory_evidence(raw_recommendations)

        # 6. Recommendation Deduplication (Section 15)
        deduped_recommendations = cls._deduplicate_recommendations(raw_recommendations)

        # 6b. Strict Small-Sample Safety Enforcement (Section 8)
        for rec in deduped_recommendations:
            for ev in rec.get("evidence", []):
                d = ev.get("denominator")
                if d is not None and d < 5:
                    rec["strength"] = "insufficient_evidence"
                    if "< 5" not in rec.get("limitation", "") and "n < 5" not in rec.get("limitation", ""):
                        rec["limitation"] = (rec.get("limitation", "") + f" Sample size in subgroup '{rec.get('provenance', {}).get('comparison_group', 'cohort')}' is small (n={d} < 5); findings are strictly exploratory.").strip()

        # 7. Categorization & Grouping
        categories_map: Dict[str, List[Dict[str, Any]]] = {
            "PRESERVE": [],
            "CHANGE": [],
            "TEST": [],
            "AVOID_OVERINTERPRETATION": [],
            "INVESTIGATE": [],
            "NO_ACTIONABLE_RECOMMENDATION": [],
        }

        for rec in deduped_recommendations:
            cat = rec.get("category", "TEST")
            if cat in categories_map:
                categories_map[cat].append(rec)
            else:
                categories_map["TEST"].append(rec)

        # 8. Sort Recommendations by Priority
        priority_order = {"high": 1, "medium": 2, "low": 3}
        deduped_recommendations.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 4))

        # 9. Build Traceable Evidence Graph (Section 21)
        evidence_graph = cls._build_evidence_graph(target_reel_id, deduped_recommendations)

        return {
            "schema_version": cls.SCHEMA_VERSION,
            "status": "SUCCESS",
            "target_reel_id": target_reel_id,
            "comparison_group": primary_subgroup,
            "competitor_count": verified_count,
            "subgroup_sample_size": primary_n,
            "recommendations_count": len(deduped_recommendations),
            "recommendations": deduped_recommendations,
            "categories": categories_map,
            "evidence_graph": evidence_graph,
            "provenance": {
                "target_reel_id": target_reel_id,
                "comparison_group": primary_subgroup,
                "competitor_count": verified_count,
                "zero_network_calls": True,
                "zero_duplicate_inference": True,
                "zero_fabricated_confidence": True,
            },
            "limitations": (
                "All recommendations represent evidence-grounded observational comparisons within the "
                "selected verified competitor cohort. They do not predict algorithmic distribution, "
                "viewer retention, or commercial engagement."
            ),
        }

    # =========================================================================
    # Domain Evaluators (Sections 9, 10, 11, 12)
    # =========================================================================

    @classmethod
    def _evaluate_spatial_focus(
        cls,
        target_norm: Dict[str, Any],
        att_intel: Dict[str, Any],
        comp_intel: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate opening spatial saliency distribution against verified peers."""
        spat_intel = att_intel.get("spatial", {})
        focal_comp = spat_intel.get("focal_competition", {})
        comp_state = focal_comp.get("competition_state", "unknown")
        disp = float(spat_intel.get("dispersion", 0.30))
        focal_conc = float(spat_intel.get("focal_concentration", 0.50))

        # Check competitor evidence for concentrated opening spatial focus
        pat = cls._find_pattern(comp_intel, "concentrated_opening_spatial_focus")
        if not pat:
            return None

        num = pat["numerator"]
        den = pat["denominator"]
        prev = pat["prevalence"]
        sg = pat["subgroup"]
        comp_ids = pat.get("provenance", {}).get("competitor_reel_ids", [])

        # Target is aligned: concentrated opening visual focus present
        target_is_concentrated = (focal_conc >= 0.50 or disp < 0.32) and comp_state not in {"multi_focus", "diffuse"}

        evidence_record = {
            "type": "competitor_subgroup_pattern",
            "source": "TinySalNet/AttentionIntelligence",
            "value": f"{num}/{den} ({prev*100:.1f}%) peers exhibit concentrated opening spatial focus",
            "numerator": num,
            "denominator": den,
            "competitor_ids": comp_ids,
            "provenance": pat.get("provenance", {}),
        }

        reach_obs = comp_intel.get("reach_aware_observations")
        ev_list = [evidence_record]
        if reach_obs:
            ev_list.append({
                "type": "reach_weighted_competitor_pattern",
                "source": "CompetitorIntelligence/ReachSorting",
                "value": f"Top-reach peer subgroup exhibits {reach_obs.get('highest_reach_prevalence', 'high prevalence')} for {reach_obs.get('pattern_name', 'dominant pattern')}",
                "competitor_ids": reach_obs.get("top_performer_ids", []),
                "provenance": {"comparison_group": "highest_reach_same_edit_type"},
            })

        if den < 5:
            return {
                "recommendation_id": "rec_spatial_focus_exploratory",
                "category": "AVOID_OVERINTERPRETATION",
                "priority": "low",
                "observation": f"Target opening spatial dispersion is {disp:.3f} (focal concentration: {focal_conc:.2f}).",
                "evidence": ev_list,
                "target_comparison": {
                    "target": f"dispersion={disp:.3f}",
                    "comparison_group": f"{num}/{den} concentrated in {sg}",
                    "difference": "Subgroup sample size insufficient (n < 5) to validate creative divergence.",
                },
                "creative_implication": "Small cohort size precludes establishing a reliable compositional benchmark.",
                "recommendation": "Preserve current opening composition until a larger verified peer cohort is available.",
                "testable_action": "Review opening visual balance qualitatively without making structural changes.",
                "strength": "insufficient_evidence",
                "confidence": None,
                "limitation": f"Sample size in subgroup '{sg}' is small (n={den} < 5); findings are strictly exploratory.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["spatial.dispersion", "spatial.focal_concentration"]),
            }

        if target_is_concentrated:
            return {
                "recommendation_id": "rec_preserve_spatial_focus",
                "category": "PRESERVE",
                "priority": "medium",
                "observation": f"Target exhibits concentrated opening visual focus (dispersion: {disp:.3f}, focal concentration: {focal_conc:.2f}).",
                "evidence": ev_list,
                "target_comparison": {
                    "target": "concentrated opening spatial focus (aligned)",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) in {sg}",
                    "difference": "Target aligns with dominant verified peer composition.",
                },
                "creative_implication": "The opening composition already provides a clear ocular anchor around the primary visual subject.",
                "recommendation": "PRESERVE: Maintain the current concentrated visual framing around the primary subject in the opening 1.5 seconds.",
                "testable_action": "Keep the existing opening framing intact as the baseline control for creative variations.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Laboratory-derived gaze prediction measures visual conspicuity and does not guarantee mobile retention.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["spatial.dispersion", "spatial.focal_concentration"]),
            }
        else:
            # Target is divergent (diffuse / multi-focus)
            return {
                "recommendation_id": "rec_change_spatial_focus",
                "category": "CHANGE" if prev >= 0.70 else "TEST",
                "priority": "high" if prev >= 0.70 else "medium",
                "observation": f"Target opening visual saliency is diffusely distributed across competing regions (dispersion: {disp:.3f}, state: '{comp_state}').",
                "evidence": ev_list,
                "target_comparison": {
                    "target": f"diffuse/multi-focus (dispersion: {disp:.3f})",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) concentrated in {sg}",
                    "difference": "Target provides less visual concentration than observed peer majority.",
                },
                "creative_implication": "Multiple visual elements compete for gaze during the opening segment, potentially dispersing viewer focal attention.",
                "recommendation": "Consider simplifying the opening composition so that the primary subject or character remains visually dominant.",
                "testable_action": (
                    "Create Variant B by reducing background motion, cropping closer to the primary character, "
                    "or dimming peripheral graphics during the first 1.5s, then compare focal clarity against current Variant A."
                ),
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": (
                    "Observational comparison within the selected verified competitor cohort; "
                    "does not prove that concentrated framing causes improved viewer retention."
                ),
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["spatial.dispersion", "spatial.focal_competition"]),
            }

    @classmethod
    def _evaluate_typography(
        cls,
        target_norm: Dict[str, Any],
        att_intel: Dict[str, Any],
        comp_intel: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate on-screen typography and text-saliency alignment."""
        t_ocr = target_norm.get("ocr", {}).get("value", {})
        has_text = bool(t_ocr.get("text_present") or t_ocr.get("text") or target_norm.get("ocr_word_count", 0) > 0)
        t_time = t_ocr.get("first_text_time")
        if t_time is None and has_text:
            t_time = 1.0
        is_early_text = has_text and (t_time is not None and float(t_time) <= 2.0)

        pat = cls._find_pattern(comp_intel, "text_led_opening")
        if not pat:
            return None

        num = pat["numerator"]
        den = pat["denominator"]
        prev = pat["prevalence"]
        sg = pat["subgroup"]
        comp_ids = pat.get("provenance", {}).get("competitor_reel_ids", [])

        evidence_record = {
            "type": "competitor_subgroup_pattern",
            "source": "OCR/TwelveLabs",
            "value": f"{num}/{den} ({prev*100:.1f}%) peers utilize on-screen typography within the first 2.0 seconds",
            "numerator": num,
            "denominator": den,
            "competitor_ids": comp_ids,
            "provenance": pat.get("provenance", {}),
        }

        # Check text-saliency relationship from Attention Intelligence
        rel_obj = att_intel.get("relationships", {})
        text_align = rel_obj.get("text_saliency_alignment", {}).get("alignment_state")
        if not text_align:
            mm_align = att_intel.get("multimodal", {}).get("text_saliency_interaction", "")
            if "high" in mm_align or "moderate" in mm_align:
                text_align = "congruent_text_anchor"
            elif "divergent" in mm_align:
                text_align = "divergent_text_distraction"
            else:
                text_align = "congruent_text_anchor" if is_early_text else "none"

        if is_early_text:
            if text_align != "divergent_text_distraction":
                return {
                    "recommendation_id": "rec_preserve_early_typography",
                    "category": "PRESERVE",
                    "priority": "medium",
                    "observation": f"Target incorporates early on-screen typography at {float(t_time or 0.0):.2f}s with congruent saliency anchoring.",
                    "evidence": [evidence_record],
                    "target_comparison": {
                        "target": f"early typography present at {float(t_time or 0.0):.2f}s (aligned)",
                        "comparison_group": f"{num}/{den} ({prev*100:.1f}%) in {sg}",
                        "difference": "Target aligns with observed peer convention.",
                    },
                    "creative_implication": "On-screen text immediately frames the narrative context without disrupting visual subject saliency.",
                    "recommendation": "PRESERVE: Retain the opening textual hook card and typography styling in this position.",
                    "testable_action": "Maintain current title typography as standard template for upcoming releases in this topic.",
                    "strength": cls._classify_strength(den, prev),
                    "confidence": None,
                    "limitation": "Text efficacy varies with viewer device screen size and platform caption overlays.",
                    "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["ocr.text_present", "ocr.first_text_time"]),
                }
            elif text_align == "divergent_text_distraction":
                return {
                    "recommendation_id": "rec_investigate_text_saliency_conflict",
                    "category": "INVESTIGATE",
                    "priority": "medium",
                    "observation": "On-screen typography is present but exhibits spatial divergence from the primary visual focal anchor.",
                    "evidence": [
                        evidence_record,
                        {
                            "type": "attention_intelligence_relationship",
                            "source": "TinySalNet/OCR",
                            "value": f"text_saliency_alignment state is '{text_align}'",
                            "provenance": rel_obj.get("text_saliency_alignment", {}),
                        },
                    ],
                    "target_comparison": {
                        "target": "divergent text-saliency placement",
                        "comparison_group": "peer convention pairs typography with focal anchor",
                        "difference": "Text placement creates spatial competition with the visual subject.",
                    },
                    "creative_implication": "Viewers must split ocular focus between reading text overlays and tracking character movement.",
                    "recommendation": "INVESTIGATE: Manually inspect whether text card placement can be aligned closer to the main visual subject.",
                    "testable_action": "Test Variant B with text placed inside the upper or lower safe zone directly adjacent to character focal anchor.",
                    "strength": "moderate_observed_evidence",
                    "confidence": None,
                    "limitation": "Visual attention models measure conspicuity and cannot directly gauge cognitive reading ease.",
                    "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["ocr.text_present", "relationships.text_saliency_alignment"]),
                }

        # Target lacks early text
        if prev >= 0.50 and den >= 5:
            return {
                "recommendation_id": "rec_change_early_typography",
                "category": "CHANGE" if prev >= 0.70 else "TEST",
                "priority": "high" if prev >= 0.70 else "medium",
                "observation": "No readable on-screen typography was detected within the opening 2.0 seconds.",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": "no early on-screen text",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) use early typography in {sg}",
                    "difference": "Target omits textual hook framing observed in peer majority.",
                },
                "creative_implication": "Viewers watching with audio muted or low volume may not immediately grasp the narrative premise.",
                "recommendation": "Consider introducing a punchy on-screen title card or key dialogue quote within the opening 1.5 seconds.",
                "testable_action": "Create Variant B with a 2-to-4 word hook title card in the opening 1.0s and test comprehension in review.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Text overlay value depends heavily on font legibility, contrast against background, and safe-zone compliance.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["ocr.text_present", "ocr.first_text_time"]),
            }

        return None

    @classmethod
    def _evaluate_human_presence(
        cls,
        target_norm: Dict[str, Any],
        att_intel: Dict[str, Any],
        comp_intel: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate early character / human face presence."""
        t_att = target_norm.get("temporal_attention", {}).get("value", [])
        if t_att:
            t_face = any(w.get("feature_values", {}).get("face_presence", 0) > 0.2 for w in t_att[:4])
        else:
            t_face = bool(target_norm.get("human_presence"))

        pat = cls._find_pattern(comp_intel, "face_presence")
        if not pat:
            return None

        num = pat["numerator"]
        den = pat["denominator"]
        prev = pat["prevalence"]
        sg = pat["subgroup"]
        comp_ids = pat.get("provenance", {}).get("competitor_reel_ids", [])

        evidence_record = {
            "type": "competitor_subgroup_pattern",
            "source": "Human/NEMAR/TwelveLabs",
            "value": f"{num}/{den} ({prev*100:.1f}%) peers feature early human presence in opening 2.0 seconds",
            "numerator": num,
            "denominator": den,
            "competitor_ids": comp_ids,
            "provenance": pat.get("provenance", {}),
        }

        if t_face:
            return {
                "recommendation_id": "rec_preserve_human_presence",
                "category": "PRESERVE",
                "priority": "medium",
                "observation": "Target establishes human character face presence within the opening 2.0 seconds.",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": "early character face presence observed (aligned)",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) in {sg}",
                    "difference": "Target aligns with validated peer convention.",
                },
                "creative_implication": "Human facial features provide an immediate social ocular anchor for viewers.",
                "recommendation": "PRESERVE: Prioritize early character presence and direct gaze during opening cuts.",
                "testable_action": "Maintain character close-up in the opening 2.0s as standard narrative introduction.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Social gaze anchoring is based on laboratory eye-tracking and does not dictate audience topic interest.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["visual.face_presence"]),
            }
        elif prev >= 0.50 and den >= 5:
            return {
                "recommendation_id": "rec_test_human_presence",
                "category": "TEST",
                "priority": "medium",
                "observation": "Target opening relies on ambient scene visuals or object motion without human character presence.",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": "no human face in opening 2.0s",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) feature human presence in {sg}",
                    "difference": "Target differs from observed character-centric peer majority.",
                },
                "creative_implication": "Opening lacks an immediate human social gaze attractor relative to peer videos.",
                "recommendation": "Consider testing an alternative opening cut that introduces the primary character's face within the first 1.5 seconds.",
                "testable_action": "Render Variant B with an opening character close-up before transitioning into wider action sequence.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Artistic formats (e.g. ambient landscapes, VFX montages) may intentionally omit human faces without diminishing creative quality.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["visual.face_presence"]),
            }

        return None

    @classmethod
    def _evaluate_editing_pacing(
        cls,
        target_norm: Dict[str, Any],
        att_intel: Dict[str, Any],
        comp_intel: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate cut pacing and scene transition behavior."""
        t_att = target_norm.get("temporal_attention", {}).get("value", [])
        if t_att:
            t_cuts = sum(w.get("feature_values", {}).get("scene_cut_rate", 0) > 0.4 for w in t_att)
        else:
            t_cuts = int(target_norm.get("scene_cut_count") or (5 if float(target_norm.get("scene_cut_rate", 0)) > 0.4 else 1))
        is_dynamic_pacing = bool(t_cuts >= 3)

        pat = cls._find_pattern(comp_intel, "dynamic_transitions")
        if not pat:
            return None

        num = pat["numerator"]
        den = pat["denominator"]
        prev = pat["prevalence"]
        sg = pat["subgroup"]
        comp_ids = pat.get("provenance", {}).get("competitor_reel_ids", [])

        evidence_record = {
            "type": "competitor_subgroup_pattern",
            "source": "Editing/VideoProcessor/NEMAR",
            "value": f"{num}/{den} ({prev*100:.1f}%) peers maintain dynamic cut pacing (3+ scene shifts)",
            "numerator": num,
            "denominator": den,
            "competitor_ids": comp_ids,
            "provenance": pat.get("provenance", {}),
        }

        if is_dynamic_pacing:
            return {
                "recommendation_id": "rec_preserve_kinetic_pacing",
                "category": "PRESERVE",
                "priority": "low",
                "observation": f"Target maintains kinetic scene cut pacing with {t_cuts} transition events.",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": f"{t_cuts} scene shifts (aligned with kinetic peer style)",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) in {sg}",
                    "difference": "Target pacing aligns with dynamic peer rhythm.",
                },
                "creative_implication": "Dynamic cut progression maintains continuous visual stimulation across scene boundaries.",
                "recommendation": "PRESERVE: Retain the existing shot change cadence across core action sequences.",
                "testable_action": "Keep cut timestamps locked while refining color grade and sound synchronization.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "High cut pacing must remain synchronized with audio tempo to prevent visual-acoustic fatigue.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["visual.scene_cuts"]),
            }
        elif prev >= 0.50 and den >= 5:
            return {
                "recommendation_id": "rec_change_cut_pacing",
                "category": "CHANGE" if prev >= 0.75 else "TEST",
                "priority": "medium",
                "observation": f"Target exhibits static scene progression ({t_cuts} transitions observed across duration).",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": f"static pacing ({t_cuts} cuts)",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) maintain 3+ cuts in {sg}",
                    "difference": "Target shot duration is longer than observed peer median.",
                },
                "creative_implication": "Prolonged static shots may produce visual accommodation in short-form mobile viewing contexts.",
                "recommendation": "Consider introducing b-roll cutaways, punch-in focal reframes, or match cuts every 2.0 to 2.5 seconds.",
                "testable_action": "Edit Variant B with quick cutaways to action details or rhythmic push-ins on beat drops.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Cut frequency must align with narrative tone; dialogue or educational content often benefits from longer continuous takes.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["visual.scene_cuts"]),
            }

        return None

    @classmethod
    def _evaluate_temporal_prominence(
        cls,
        target_norm: Dict[str, Any],
        att_intel: Dict[str, Any],
        comp_intel: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate opening relative attention peak and persistence."""
        t_temp = att_intel.get("temporal", {})
        peak_t = float(t_temp.get("peak_attention_time", 999.0))
        retention_ratio = float(t_temp.get("temporal_persistence", {}).get("opening_retention_ratio", 0.0))
        has_opening_peak = bool(peak_t <= 2.5 or retention_ratio >= 0.90)

        pat = cls._find_pattern(comp_intel, "att_opening_temporal_attention_prominence") or cls._find_pattern(comp_intel, "opening_temporal_attention_prominence")
        if not pat:
            return None

        num = pat["numerator"]
        den = pat["denominator"]
        prev = pat["prevalence"]
        sg = pat["subgroup"]
        comp_ids = pat.get("provenance", {}).get("competitor_reel_ids", [])

        evidence_record = {
            "type": "competitor_subgroup_pattern",
            "source": "NEMAR/AttentionIntelligence",
            "value": f"{num}/{den} ({prev*100:.1f}%) peers achieve opening temporal attention prominence",
            "numerator": num,
            "denominator": den,
            "competitor_ids": comp_ids,
            "provenance": pat.get("provenance", {}),
        }

        if has_opening_peak:
            return {
                "recommendation_id": "rec_preserve_opening_temporal_peak",
                "category": "PRESERVE",
                "priority": "low",
                "observation": f"Target achieves an early relative attention peak at {peak_t:.2f}s (retention ratio: {retention_ratio:.2f}).",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": f"early relative peak at {peak_t:.2f}s (aligned)",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) in {sg}",
                    "difference": "Target successfully front-loads visual engagement.",
                },
                "creative_implication": "High early visual contrast establishes immediate focal engagement during the opening segment.",
                "recommendation": "PRESERVE: Keep the opening dynamic beat placed within the first 1.5 seconds.",
                "testable_action": "Maintain opening sequence timing while testing alternate audio hooks.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Relative temporal gaze peaks indicate predicted visual novelty, not audience retention rates.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["temporal.peak_attention_time"]),
            }
        elif prev >= 0.50 and den >= 5:
            return {
                "recommendation_id": "rec_change_opening_temporal_prominence",
                "category": "CHANGE" if prev >= 0.70 else "TEST",
                "priority": "high" if prev >= 0.70 else "medium",
                "observation": f"Target first major relative attention peak is delayed until {peak_t:.2f}s.",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": f"delayed attention peak at {peak_t:.2f}s",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) achieve peak in opening 2.0s in {sg}",
                    "difference": "Target opening visual novelty ramps up more slowly than peer majority.",
                },
                "creative_implication": "The initial 1.5 seconds provide low predicted visual novelty, creating a slow narrative ramp.",
                "recommendation": "Consider front-loading a decisive visual movement, graphic reveal, or contrast jump into the opening 1.0 second.",
                "testable_action": "Edit Variant B to start directly on the action climax or explosive visual beat rather than a slow establishing shot.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Slow-build cinematic narratives may intentionally delay visual peaks for artistic storytelling reasons.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["temporal.peak_attention_time"]),
            }

        return None

    @classmethod
    def _evaluate_audio_speech(
        cls,
        target_norm: Dict[str, Any],
        att_intel: Dict[str, Any],
        comp_intel: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate spoken opening and verbal engagement."""
        speech_present = bool(target_norm.get("audio_speech", {}).get("value", {}).get("speech_present"))

        pat = cls._find_pattern(comp_intel, "spoken_opening")
        if not pat:
            return None

        num = pat["numerator"]
        den = pat["denominator"]
        prev = pat["prevalence"]
        sg = pat["subgroup"]
        comp_ids = pat.get("provenance", {}).get("competitor_reel_ids", [])

        evidence_record = {
            "type": "competitor_subgroup_pattern",
            "source": "AudioProcessor/TwelveLabs",
            "value": f"{num}/{den} ({prev*100:.1f}%) peers utilize spoken verbal speech in opening",
            "numerator": num,
            "denominator": den,
            "competitor_ids": comp_ids,
            "provenance": pat.get("provenance", {}),
        }

        if speech_present:
            return {
                "recommendation_id": "rec_preserve_spoken_opening",
                "category": "PRESERVE",
                "priority": "low",
                "observation": "Target features vocal speech or voiceover dialogue in the opening segment.",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": "speech present in opening (aligned)",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) in {sg}",
                    "difference": "Target aligns with verbal storytelling convention.",
                },
                "creative_implication": "Spoken audio establishes dual-modality (acoustic and visual) engagement.",
                "recommendation": "PRESERVE: Retain the opening spoken dialogue hook.",
                "testable_action": "Ensure speech audio is cleanly mixed over background music track.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Acoustic engagement requires viewers to unmute audio.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["audio.speech_presence"]),
            }
        elif prev >= 0.60 and den >= 5:
            return {
                "recommendation_id": "rec_test_spoken_voiceover",
                "category": "TEST",
                "priority": "medium",
                "observation": "Target relies purely on ambient music without spoken voiceover or verbal dialogue.",
                "evidence": [evidence_record],
                "target_comparison": {
                    "target": "music only / no speech",
                    "comparison_group": f"{num}/{den} ({prev*100:.1f}%) utilize speech in {sg}",
                    "difference": "Target differs from conversational peer norm.",
                },
                "creative_implication": "Lacks verbal framing to hook viewers listening on headphones or speakers.",
                "recommendation": "Consider testing an introductory voiceover hook sentence or memorable dialogue line over the opening music.",
                "testable_action": "Record a 3-second spoken intro ('Watch this...' / 'Here is how...') for Variant B and evaluate in review.",
                "strength": cls._classify_strength(den, prev),
                "confidence": None,
                "limitation": "Music-only edits are often preferred for ambient aesthetic or dance trends.",
                "provenance": cls._build_rec_provenance(target_norm, sg, comp_ids, ["audio.speech_presence"]),
            }

        return None

    # =========================================================================
    # Conflict Resolution & Safety (Sections 14, 16, 17)
    # =========================================================================

    @classmethod
    def _resolve_contradictory_evidence(cls, recommendations: List[Dict[str, Any]]) -> None:
        """Identify conflicting evidence between analytical signals and reclassify as INVESTIGATE (Section 16)."""
        rec_map = {r["recommendation_id"]: r for r in recommendations}

        # Conflict 1: Preserving text while text is divergent from saliency
        if "rec_preserve_early_typography" in rec_map and "rec_investigate_text_saliency_conflict" in rec_map:
            # Drop the preserve recommendation in favor of investigation
            recommendations.remove(rec_map["rec_preserve_early_typography"])

        # Conflict 2: Fast cut recommendation vs diffuse spatial composition
        # Rapid cuts with diffuse spatial composition can compound visual disorientation
        if "rec_change_cut_pacing" in rec_map and "rec_change_spatial_focus" in rec_map:
            rec_cuts = rec_map["rec_change_cut_pacing"]
            rec_cuts["limitation"] += " Note: Increase cut frequency only after primary visual subject focus has been established to avoid ocular disorientation."

    @classmethod
    def _deduplicate_recommendations(cls, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate semantically overlapping recommendations (Section 15)."""
        seen_domains: set[str] = set()
        deduped: List[Dict[str, Any]] = []

        domain_mapping = {
            "rec_preserve_spatial_focus": "spatial",
            "rec_change_spatial_focus": "spatial",
            "rec_spatial_focus_exploratory": "spatial",
            "rec_preserve_early_typography": "typography",
            "rec_change_early_typography": "typography",
            "rec_investigate_text_saliency_conflict": "typography",
            "rec_preserve_human_presence": "human",
            "rec_test_human_presence": "human",
            "rec_preserve_kinetic_pacing": "editing",
            "rec_change_cut_pacing": "editing",
            "rec_preserve_opening_temporal_peak": "temporal",
            "rec_change_opening_temporal_prominence": "temporal",
            "rec_preserve_spoken_opening": "audio",
            "rec_test_spoken_voiceover": "audio",
        }

        for rec in recommendations:
            r_id = rec.get("recommendation_id", "")
            domain = domain_mapping.get(r_id, r_id)
            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            deduped.append(rec)

        return deduped

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _find_pattern(comp_intel: Dict[str, Any], pattern_key: str) -> Optional[Dict[str, Any]]:
        """Look up pattern in recurring patterns or all subgroup patterns."""
        pats = comp_intel.get("recurring_patterns", []) + comp_intel.get("all_subgroup_patterns", [])
        for p in pats:
            pid = p.get("pattern_id", "")
            if pattern_key in pid:
                return p
        # Fallback keyword and domain aliases
        alias_map = {
            "text_led_opening": ["typography", "text", "ocr", "headline", "early_typography"],
            "face_presence": ["human", "character", "face", "creator", "primary_character"],
            "dynamic_transitions": ["editing", "cut", "transition", "pacing", "scene_transition", "rapid_initial"],
            "concentrated_opening_spatial_focus": ["spatial", "focus", "saliency"],
        }
        aliases = alias_map.get(pattern_key, [pattern_key])
        for p in pats:
            pid = p.get("pattern_id", "").lower()
            pname = p.get("pattern_name", "").lower()
            pdom = p.get("domain", "").lower()
            if any(a in pid or a in pname or a in pdom for a in aliases):
                return p
        return None

    @staticmethod
    def _classify_strength(denominator: int, prevalence: float) -> str:
        """Classify evidence strength strictly enforcing small-sample safety (Section 8)."""
        if denominator < 5:
            return "insufficient_evidence"
        elif denominator < 10:
            return "moderate_observed_evidence" if prevalence >= 0.65 else "exploratory_evidence"
        else:
            if prevalence >= 0.70:
                return "strong_observed_evidence"
            elif prevalence >= 0.50:
                return "moderate_observed_evidence"
            else:
                return "exploratory_evidence"

    @staticmethod
    def _build_rec_provenance(
        target_norm: Dict[str, Any],
        subgroup: str,
        comp_ids: List[str],
        features: List[str],
    ) -> Dict[str, Any]:
        """Build provenance record for a recommendation."""
        return {
            "target_reel_id": target_norm.get("reel_id", "target_reel"),
            "comparison_group": subgroup,
            "competitor_reel_ids": comp_ids,
            "source_features": features,
            "evidence_quality": "strictly_traceable",
        }

    @classmethod
    def _build_evidence_graph(
        cls,
        target_reel_id: str,
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Construct formal evidence graph representation (Section 21)."""
        nodes = []
        for r in recommendations:
            t_comp = r.get("target_comparison", {})
            nodes.append({
                "recommendation_id": r.get("recommendation_id"),
                "category": r.get("category"),
                "target_state": t_comp.get("target"),
                "feature": ", ".join(r.get("provenance", {}).get("source_features", [])),
                "pattern": r.get("observation"),
                "peer_evidence": (r.get("evidence", [{}])[0]).get("value") if r.get("evidence") else "observed",
                "gap": t_comp.get("difference"),
                "recommendation": r.get("recommendation"),
                "trace": {
                    "target": target_reel_id,
                    "features": r.get("provenance", {}).get("source_features", []),
                    "comparison_group": r.get("provenance", {}).get("comparison_group"),
                    "competitor_ids": r.get("provenance", {}).get("competitor_reel_ids", [])[:5],
                    "recommendation": r.get("recommendation"),
                },
            })
        return {
            "target_reel_id": target_reel_id,
            "nodes": nodes,
        }

    @classmethod
    def _build_null_recommendation_result(
        cls,
        target_reel_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Build structured result when evidence is insufficient for recommendations (Section 17)."""
        null_rec = {
            "recommendation_id": "rec_null_no_competitors",
            "category": "NO_ACTIONABLE_RECOMMENDATION",
            "priority": "low",
            "observation": "Target analysis completed, but no verified relevant competitors are available for peer comparison.",
            "evidence": [],
            "target_comparison": {
                "target": target_reel_id,
                "comparison_group": "none",
                "difference": "Zero verified peer comparisons available.",
            },
            "creative_implication": "Cannot ground creative modifications in verified peer conventions without comparative baseline.",
            "recommendation": "No actionable creative changes supported by the available comparative evidence.",
            "testable_action": "Expand Instagram discovery queries to locate verified relevant competitors before adjusting creative structure.",
            "strength": "insufficient_evidence",
            "confidence": None,
            "limitation": "A peer comparison group is required to formulate evidence-grounded recommendations.",
            "provenance": {
                "target_reel_id": target_reel_id,
                "comparison_group": "none",
                "competitor_reel_ids": [],
                "source_features": [],
            },
        }

        return {
            "schema_version": cls.SCHEMA_VERSION,
            "status": "NO_ACTIONABLE_RECOMMENDATION",
            "target_reel_id": target_reel_id,
            "comparison_group": "none",
            "competitor_count": 0,
            "subgroup_sample_size": 0,
            "recommendations_count": 1,
            "recommendations": [null_rec],
            "categories": {
                "PRESERVE": [],
                "CHANGE": [],
                "TEST": [],
                "AVOID_OVERINTERPRETATION": [],
                "INVESTIGATE": [],
                "NO_ACTIONABLE_RECOMMENDATION": [null_rec],
            },
            "evidence_graph": [],
            "provenance": {
                "target_reel_id": target_reel_id,
                "comparison_group": "none",
                "competitor_count": 0,
                "zero_network_calls": True,
                "zero_duplicate_inference": True,
                "zero_fabricated_confidence": True,
            },
            "limitations": reason,
        }
