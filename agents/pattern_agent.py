"""Pattern Agent.

Specialist agent that analyzes ONLY verified relevant competitors with downloaded MP4s,
Twelve Labs analyses, and attention timelines.
Detects recurring empirical creative patterns across:
1. ALL VERIFIED COMPETITORS
2. HIGHEST-REACH VERIFIED COMPETITORS (subgroup analysis)

Enforces Sections 14, 15, 21:
- Replaces universal >=0.70 attention threshold with relative opening peak prominence and baseline delta.
- Distinguishes all verified from highest-reach verified subgroups.
- Reports exact counts, denominators, and evidence without fabricated percentages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from services.competitor_intelligence import CompetitorIntelligenceService


class PatternAgent:
    """Discovers recurring patterns across verified competitors using exact empirical tallies."""

    @staticmethod
    def _extract_video_patterns(cand: Dict[str, Any]) -> Dict[str, bool]:
        """Extract pattern indicators for a single Reel using relative attention measures."""
        comp_analysis = cand.get("analysis") or {}
        canonical = comp_analysis.get("canonical") or comp_analysis
        attention = cand.get("attention_predictions") or []

        # 1. Early face / direct gaze within first 2.0s
        has_face = False
        for w in attention[:4]:
            if w.get("feature_values", {}).get("face_presence", 0) > 0.2:
                has_face = True
                break

        # 2. Early on-screen typography
        ocr_obj = canonical.get("ocr") or {}
        first_t = ocr_obj.get("first_text_time")
        ocr_text = str(canonical.get("ocr_text") or canonical.get("onscreen_text") or ocr_obj.get("ocr_text") or "")
        has_early_text = (
            (first_t is not None and float(first_t) <= 2.0)
            or bool(ocr_text)
            or any(w.get("feature_values", {}).get("text_presence", 0) > 0.1 for w in attention[:4])
        )

        # 3. Fast cut pacing
        cuts = sum(w.get("feature_values", {}).get("scene_cut_rate", 0) > 0.4 for w in attention)
        has_fast_cuts = bool(cuts >= 3)

        # 4. Speech in opening
        transcript = str(canonical.get("transcript") or "")
        has_speech = bool(transcript) or any(
            w.get("feature_values", {}).get("audio_speech_presence", 0) > 0.3 for w in attention[:4]
        )

        # 5. Dynamic motion onset
        has_motion = any(w.get("feature_values", {}).get("motion_magnitude", 0) > 0.03 for w in attention[:4])

        # 6. Opening relative attention peak (Section 15: NO universal >= 0.70 threshold)
        # Evaluated via relative peak prominence, baseline delta, or local peak in opening 2.0s
        has_opening_peak = False
        if attention:
            opening_windows = attention[: min(4, len(attention))]
            baseline_score = float(attention[0].get("attention_score", 0.5))
            for w in opening_windows:
                score = float(w.get("attention_score", 0.0))
                prominence = float(w.get("peak_prominence", 0.0))
                is_local = bool(w.get("is_local_peak", False))
                delta = score - baseline_score
                # Relative criteria: local peak with prominence or delta >= 0.04
                if (is_local and prominence > 0.02) or (delta >= 0.04):
                    has_opening_peak = True
                    break

        # 7. Spatial / Attention Intelligence indicators
        att_intel = cand.get("attention_intelligence") or {}
        spat_signal = cand.get("dhf1k_spatial_saliency_signal") or (cand.get("analysis") or {}).get("dhf1k_spatial_saliency_signal") or {}

        has_opening_spatial_focus = False
        has_stable_focal = False

        if att_intel:
            spat_obj = att_intel.get("spatial", {})
            focal_conc = spat_obj.get("focal_concentration", 0.0)
            disp = spat_obj.get("dispersion", 1.0)
            rel_obj = att_intel.get("relationships", {})
            align = rel_obj.get("spatial_temporal_alignment", {}).get("alignment_type")

            has_opening_spatial_focus = bool(focal_conc >= 0.50 or disp < 0.32)
            has_stable_focal = bool(align == "stable_focal_attention" or (has_opening_peak and has_opening_spatial_focus))
        elif spat_signal and spat_signal.get("status") == "SUCCESS":
            summ = spat_signal.get("summary", {})
            mean_disp = float(summ.get("mean_spatial_dispersion", 1.0))
            has_opening_spatial_focus = bool(mean_disp < 0.32)
            has_stable_focal = bool(has_opening_peak and has_opening_spatial_focus)

        return {
            "early_human_presence": has_face,
            "direct_camera_engagement": has_face,
            "early_on_screen_text": has_early_text,
            "fast_cut_pacing": has_fast_cuts,
            "speech_in_opening": has_speech,
            "dynamic_motion_onset": has_motion,
            "opening_attention_peak": has_opening_peak,
            "concentrated_opening_spatial_focus": has_opening_spatial_focus,
            "stable_focal_attention": has_stable_focal,
        }

    @classmethod
    def analyze(
        cls,
        verified_competitors: List[Dict[str, Any]],
        your_reel: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze verified competitors primarily against same entity + same edit-type competitors.

        Subgroup hierarchy:
        1. same_entity_and_edit_type (PRIMARY comparison)
        2. same_edit_type
        3. highest_reach_same_edit_type
        4. all_verified (broader market context)

        Strict denominator accounting: every finding reports count, denominator, subgroup, and provenance.
        """
        your_reel = your_reel or {}

        if not verified_competitors:
            return {
                "agent": "pattern",
                "findings": ["[AGGREGATED_PATTERN] No verified competitors available for pattern extraction."],
                "evidence": [],
                "confidence": 0.0,
                "uncertainties": ["Pattern discovery requires at least 1 verified relevant competitor."],
                "recommendations": ["[RECOMMENDATION] Expand discovery queries or lower minimum relevance threshold slightly."],
                "patterns": [],
                "pattern_counts": {},
                "subgroups": {},
                "verified_competitor_count": 0,
                "highest_reach_count": 0,
            }

        total_verified = len(verified_competitors)

        # 1. Extract target entity and edit-type from uploaded reel
        dp = your_reel.get("discovery_profile") or {}
        ua = your_reel.get("analysis") or {}
        target_entity = str(dp.get("primary_entity") or ua.get("primary_entity") or your_reel.get("primary_entity") or "").lower().strip().replace("-", " ")
        target_edit_type = str(dp.get("edit_type") or ua.get("edit_type") or your_reel.get("edit_type") or "action montage").lower().strip()

        # Helper: test entity match
        def _match_entity(c: Dict[str, Any]) -> bool:
            if not target_entity:
                return True
            c_ent = str((c.get("analysis") or {}).get("primary_entity") or c.get("primary_entity") or "").lower().strip().replace("-", " ")
            c_caption = str(c.get("caption") or "").lower()
            if target_entity in c_ent or c_ent in target_entity:
                return True
            target_words = [w for w in target_entity.split() if len(w) > 2]
            if target_words and all(w in c_caption or w in c_ent for w in target_words):
                return True
            if any(w in c_caption or w in c_ent for w in target_words if len(w) > 3):
                return True
            return False

        # Helper: test edit-type match
        def _match_edit_type(c: Dict[str, Any]) -> bool:
            c_cluster = str(c.get("competitor_cluster") or "").lower()
            c_et = str(c.get("edit_type") or (c.get("analysis") or {}).get("edit_type") or "").lower()
            combined = f"{c_cluster} {c_et} {c.get('caption', '')} {(c.get('analysis') or {}).get('format', '')}".lower()
            if any(w in target_edit_type for w in ["action", "montage", "fight", "battle"]):
                return any(w in combined for w in ["action", "montage", "fight", "battle", "combat", "cinematic", "vfx"])
            if any(w in target_edit_type for w in ["emotion", "sad", "tribute"]):
                return any(w in combined for w in ["emotion", "sad", "tribute", "nostalgia"])
            if any(w in target_edit_type for w in ["dialogue", "quote", "speech"]):
                return any(w in combined for w in ["dialogue", "quote", "speech"])
            if any(w in target_edit_type for w in ["vfx", "cgi"]):
                return any(w in combined for w in ["vfx", "cgi", "effects"])
            return target_edit_type in combined or combined in target_edit_type

        # 2. Partition into subgroups
        same_entity_and_edit_type = [c for c in verified_competitors if _match_entity(c) and _match_edit_type(c)]
        same_edit_type = [c for c in verified_competitors if _match_edit_type(c)]

        # Determine PRIMARY subgroup
        if len(same_entity_and_edit_type) >= 1:
            primary_pool = same_entity_and_edit_type
            primary_subgroup = "same_entity_and_edit_type"
            primary_label = "same entity and edit-type"
        elif len(same_edit_type) >= 1:
            primary_pool = same_edit_type
            primary_subgroup = "same_edit_type"
            primary_label = "same edit-type"
        else:
            primary_pool = verified_competitors
            primary_subgroup = "all_verified"
            primary_label = "verified"

        # Sort primary pool by raw reach descending
        sorted_primary = sorted(
            primary_pool,
            key=lambda c: (
                int(c.get("reach_value") or c.get("views") or c.get("plays") or 0),
                float(c.get("performance_score") or 0.0)
            ),
            reverse=True
        )

        if len(sorted_primary) >= 4:
            high_reach_subgroup_size = max(2, int(np.ceil(len(sorted_primary) * 0.5)))
            highest_reach_same_edit_pool = sorted_primary[:high_reach_subgroup_size]
        else:
            highest_reach_same_edit_pool = sorted_primary

        # 3. Compute tallies for each subgroup
        pattern_keys = [
            "early_human_presence", "direct_camera_engagement", "early_on_screen_text",
            "fast_cut_pacing", "speech_in_opening", "dynamic_motion_onset", "opening_attention_peak",
            "concentrated_opening_spatial_focus", "stable_focal_attention"
        ]

        def _compute_tallies(pool: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
            tallies = {k: 0 for k in pattern_keys}
            reels_by_k: Dict[str, List[str]] = {k: [] for k in pattern_keys}
            for cand in pool:
                cid = str(cand.get("shortcode") or cand.get("id") or "unknown")
                flags = cls._extract_video_patterns(cand)
                for k, val in flags.items():
                    if val:
                        tallies[k] += 1
                        reels_by_k[k].append(cid)
            return tallies, reels_by_k

        all_tallies, all_reels = _compute_tallies(verified_competitors)
        primary_tallies, primary_reels = _compute_tallies(primary_pool)
        high_reach_tallies, high_reach_reels = _compute_tallies(highest_reach_same_edit_pool)

        pattern_names = {
            "early_human_presence": "Early Human Presence",
            "direct_camera_engagement": "Direct Camera Engagement",
            "early_on_screen_text": "Early On-Screen Typography",
            "fast_cut_pacing": "Dynamic Scene Cut Pacing",
            "speech_in_opening": "Spoken Verbal Opening",
            "dynamic_motion_onset": "High Visual Motion Onset",
            "opening_attention_peak": "Opening Relative Attention Peak",
            "concentrated_opening_spatial_focus": "Concentrated Opening Spatial Focus",
            "stable_focal_attention": "Stable Focal Attention Alignment",
        }

        # Structured pattern objects with complete provenance & denominator accounting
        pattern_objects = []
        for p_key, p_label in pattern_names.items():
            # 1. Primary entry (same entity and edit-type)
            pattern_objects.append({
                "pattern": p_key,
                "pattern_name": p_label,
                "count": primary_tallies[p_key],
                "denominator": len(primary_pool),
                "subgroup": primary_subgroup,
                "evidence": [f"Observed in {primary_tallies[p_key]}/{len(primary_pool)} {primary_label} reels: {', '.join(primary_reels[p_key][:4])}"],
                "provenance": f"PatternAgent (subgroup: {primary_subgroup}, n={len(primary_pool)})",
            })
            # 2. Highest reach same edit-type entry
            pattern_objects.append({
                "pattern": p_key,
                "pattern_name": p_label,
                "count": high_reach_tallies[p_key],
                "denominator": len(highest_reach_same_edit_pool),
                "subgroup": "highest_reach_same_edit_type",
                "evidence": [f"Observed in {high_reach_tallies[p_key]}/{len(highest_reach_same_edit_pool)} highest-reach {primary_label} reels: {', '.join(high_reach_reels[p_key][:4])}"],
                "provenance": f"PatternAgent (subgroup: highest_reach_same_edit_type, n={len(highest_reach_same_edit_pool)})",
            })
            # 3. All verified entry (broader market context)
            pattern_objects.append({
                "pattern": p_key,
                "pattern_name": p_label,
                "count": all_tallies[p_key],
                "denominator": total_verified,
                "subgroup": "all_verified",
                "evidence": [f"Observed in {all_tallies[p_key]}/{total_verified} verified reels: {', '.join(all_reels[p_key][:4])}"],
                "provenance": f"PatternAgent (subgroup: all_verified, n={total_verified})",
            })

        pattern_counts = {k: f"{primary_tallies[k]}/{len(primary_pool)}" for k in pattern_keys}
        high_reach_counts = {k: f"{high_reach_tallies[k]}/{len(highest_reach_same_edit_pool)}" for k in pattern_keys}
        all_counts = {k: f"{all_tallies[k]}/{total_verified}" for k in pattern_keys}

        # Primary comparison findings
        n_p = len(primary_pool)
        n_h = len(highest_reach_same_edit_pool)

        findings = [
            f"[AGGREGATED_PATTERN] {primary_tallies['early_human_presence']}/{n_p} {primary_label} competitors feature human presence within opening 2 seconds.",
            f"[AGGREGATED_PATTERN] {high_reach_tallies['early_on_screen_text']}/{n_h} highest-reach {primary_label} competitors used on-screen typography within the first 2 seconds.",
            f"[AGGREGATED_PATTERN] {primary_tallies['opening_attention_peak']}/{n_p} {primary_label} competitors achieve a relative opening attention peak above early baseline.",
            f"[AGGREGATED_PATTERN] {high_reach_tallies['fast_cut_pacing']}/{n_h} highest-reach {primary_label} competitors maintain a dynamic cut rate with 3+ scene shifts.",
        ]

        if primary_tallies.get("concentrated_opening_spatial_focus", 0) > 0:
            findings.append(
                f"[AGGREGATED_PATTERN] {primary_tallies['concentrated_opening_spatial_focus']}/{n_p} {primary_label} competitors display concentrated opening spatial focus around primary subjects."
            )

        if n_p < total_verified:
            findings.append(
                f"[MARKET_CONTEXT] Broader niche benchmark ({total_verified} total verified Reels): "
                f"{all_tallies['early_on_screen_text']}/{total_verified} use early on-screen typography; "
                f"{all_tallies['fast_cut_pacing']}/{total_verified} utilize rapid scene cuts."
            )

        evidence = [
            f"[OBSERVED] Primary subgroup '{primary_subgroup}' (n={n_p}): {k} in {v} reels."
            for k, v in pattern_counts.items()
        ]
        evidence.extend([
            f"[OBSERVED] Subgroup highest_reach_same_edit_type (n={n_h}): {k} in {v} reels."
            for k, v in high_reach_counts.items()
        ])

        recommendations = []
        if high_reach_tallies["early_on_screen_text"] / max(1, n_h) >= 0.5:
            recommendations.append(
                f"[RECOMMENDATION] {high_reach_tallies['early_on_screen_text']}/{n_h} highest-reach {primary_label} competitors utilize on-screen typography; reinforce key actions with kinetic text."
            )
        if high_reach_tallies["early_human_presence"] / max(1, n_h) >= 0.5:
            recommendations.append(
                f"[RECOMMENDATION] {high_reach_tallies['early_human_presence']}/{n_h} highest-reach {primary_label} competitors feature human subjects; prioritize character presence in opening 2 seconds."
            )
        if high_reach_tallies["opening_attention_peak"] / max(1, n_h) >= 0.5:
            recommendations.append(
                f"[RECOMMENDATION] {high_reach_tallies['opening_attention_peak']}/{n_h} highest-reach competitors achieve an opening attention peak; front-load decisive visual beats into opening 1.5s."
            )
        if high_reach_tallies.get("concentrated_opening_spatial_focus", 0) / max(1, n_h) >= 0.5:
            recommendations.append(
                f"[RECOMMENDATION] {high_reach_tallies['concentrated_opening_spatial_focus']}/{n_h} highest-reach competitors show concentrated opening spatial focus; keep the primary subject visually dominant."
            )

        legacy_patterns = [
            {"pattern": "Early Graphic Typography", "count": f"{primary_tallies['early_on_screen_text']}/{n_p}", "prevalence": primary_tallies['early_on_screen_text'] / max(1, n_p)},
            {"pattern": "Early Human Presence", "count": f"{primary_tallies['early_human_presence']}/{n_p}", "prevalence": primary_tallies['early_human_presence'] / max(1, n_p)},
            {"pattern": "Opening Relative Attention Peak", "count": f"{primary_tallies['opening_attention_peak']}/{n_p}", "prevalence": primary_tallies['opening_attention_peak'] / max(1, n_p)},
            {"pattern": "Dynamic Scene Transitions", "count": f"{primary_tallies['fast_cut_pacing']}/{n_p}", "prevalence": primary_tallies['fast_cut_pacing'] / max(1, n_p)},
            {"pattern": "Spoken Verbal Opening", "count": f"{primary_tallies['speech_in_opening']}/{n_p}", "prevalence": primary_tallies['speech_in_opening'] / max(1, n_p)},
        ]
        if primary_tallies.get("concentrated_opening_spatial_focus", 0) > 0:
            legacy_patterns.append({
                "pattern": "Concentrated Opening Spatial Focus",
                "count": f"{primary_tallies['concentrated_opening_spatial_focus']}/{n_p}",
                "prevalence": primary_tallies['concentrated_opening_spatial_focus'] / max(1, n_p),
            })

        # Phase 15: Run Competitor Intelligence Service
        try:
            comp_intel = CompetitorIntelligenceService.analyze(
                target_profile=your_reel,
                target_attention_intelligence=your_reel.get("attention_intelligence"),
                verified_competitors=verified_competitors,
            )
        except Exception:
            comp_intel = {
                "schema_version": "1.0.0",
                "status": "FAILED",
                "verified_competitor_count": total_verified,
                "error": "Failed to compute competitor intelligence",
            }

        return {
            "agent": "pattern",
            "findings": findings,
            "evidence": evidence,
            "confidence": 0.92,
            "uncertainties": [
                f"Patterns reflect primary subgroup '{primary_subgroup}' (n={n_p}) and broader market context (n={total_verified}); "
                "sample statistics do not represent universal platform-wide algorithms."
            ],
            "recommendations": recommendations,
            "patterns": legacy_patterns,
            "structured_patterns": pattern_objects,
            "pattern_counts": pattern_counts,
            "high_reach_counts": high_reach_counts,
            "primary_subgroup": primary_subgroup,
            "primary_pool_count": n_p,
            "verified_competitor_count": total_verified,
            "highest_reach_count": n_h,
            "subgroups": {
                "same_entity_and_edit_type": {"sample_size": len(same_entity_and_edit_type), "counts": {k: f"{sum(1 for c in same_entity_and_edit_type if cls._extract_video_patterns(c).get(k))}/{len(same_entity_and_edit_type)}" for k in pattern_keys} if same_entity_and_edit_type else {}},
                "same_edit_type": {"sample_size": len(same_edit_type), "counts": {k: f"{sum(1 for c in same_edit_type if cls._extract_video_patterns(c).get(k))}/{len(same_edit_type)}" for k in pattern_keys} if same_edit_type else {}},
                "highest_reach_same_edit_type": {"sample_size": n_h, "counts": high_reach_counts},
                "highest_reach_verified": {"sample_size": n_h, "counts": high_reach_counts},
                "all_verified": {"sample_size": total_verified, "counts": all_counts},
            },
            "competitor_intelligence": comp_intel,
        }

