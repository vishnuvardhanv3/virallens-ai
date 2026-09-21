"""Competitor Intelligence Service.

Rigorous evidence-backed analytical engine that compares an uploaded target Reel
against discovered, normalized, selected, downloaded, and Twelve-Labs-verified competitor Reels.

Enforces Phase 15 Hard Constraints:
1. Purely analytical layer over already-generated canonical data (NO new ML models, NO retraining,
   NO Apify or Twelve Labs replacement, NO network calls, NO duplicate inference).
2. Descriptive observations ONLY: Strictly NO causality claims, NO "what makes a Reel go viral",
   NO guaranteed engagement, NO scroll-stop or retention claims, NO viewer psychology inferences,
   and NO population-wide Instagram behavior rules.
3. Explicit denominator accounting: Always reports exact numerator/denominator (e.g. 6/7)
   and subgroup for every finding.
4. Small-sample safety: For n < 5, classifies as 'insufficient_sample' / 'exploratory_observation'
   and prevents strong claims.
5. Missing reach safety: Missing views/plays handled as reach=null and excluded from reach calculations.
6. Full pattern provenance: Every finding is traceable to target Reel, competitor IDs, and specific signal fields.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class CompetitorIntelligenceService:
    """Analytical service providing verified competitor intelligence and pattern comparison."""

    @staticmethod
    def normalize_target(
        target_profile: Dict[str, Any],
        target_attention_intelligence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Normalize target Reel into canonical schema preserving field provenance.
        
        Does NOT fabricate missing values.
        """
        dp = target_profile.get("discovery_profile") or {}
        ua = target_profile.get("analysis") or target_profile.get("canonical") or target_profile
        ts = target_profile.get("technical_signals") or {}
        ocr_obj = target_profile.get("ocr") or ua.get("ocr") or {}

        # Resolve entity, niche, edit_type with provenance
        entity_val = str(dp.get("primary_entity") or ua.get("primary_entity") or target_profile.get("primary_entity") or "").strip()
        niche_val = str(dp.get("niche") or ua.get("niche") or target_profile.get("niche") or "").strip()
        sub_niche_val = str(dp.get("sub_niche") or ua.get("sub_niche") or target_profile.get("sub_niche") or "").strip()
        topic_val = str(dp.get("topic") or ua.get("topic") or target_profile.get("topic") or "").strip()
        edit_type_val = str(dp.get("edit_type") or ua.get("edit_type") or target_profile.get("edit_type") or "action montage").strip()
        format_val = str(dp.get("content_format") or ua.get("format") or target_profile.get("format") or "").strip()
        style_val = str(ua.get("visual_style") or ua.get("style") or "").strip()
        hook_obj = ua.get("hook") or target_profile.get("hook") or {}

        target_reel_id = str(
            target_profile.get("shortcode")
            or target_profile.get("id")
            or target_profile.get("reel_id")
            or "target_reel"
        )

        # Video technical parameters
        dur = float(ts.get("duration") or target_profile.get("duration") or 0.0)
        res_w = ts.get("width") or target_profile.get("width")
        res_h = ts.get("height") or target_profile.get("height")
        resolution = f"{res_w}x{res_h}" if res_w and res_h else None
        fps = float(ts.get("fps") or target_profile.get("fps") or 0.0) if (ts.get("fps") or target_profile.get("fps")) else None

        # Temporal attention
        nemar_timeline = (
            target_profile.get("nemar_attention_signal")
            or target_profile.get("attention_predictions")
            or []
        )
        # Spatial saliency
        tinysal_signal = (
            target_profile.get("dhf1k_spatial_saliency_signal")
            or ua.get("dhf1k_spatial_saliency_signal")
            or {}
        )

        # Attention Intelligence
        att_intel = (
            target_attention_intelligence
            or target_profile.get("attention_intelligence")
            or {}
        )

        # Build feature records with provenance
        def _feat(feature_name: str, value: Any, source: str) -> Dict[str, Any]:
            return {
                "feature": feature_name,
                "value": value,
                "source": source,
                "provenance": {
                    "target_reel_id": target_reel_id,
                    "source_module": source,
                    "field_present": value is not None and value != "",
                },
            }

        return {
            "reel_id": target_reel_id,
            "semantic": {
                "entity": _feat("entity", entity_val or None, "DiscoveryProfile/TwelveLabs"),
                "niche": _feat("niche", niche_val or None, "DiscoveryProfile/TwelveLabs"),
                "sub_niche": _feat("sub_niche", sub_niche_val or None, "TwelveLabs"),
                "topic": _feat("topic", topic_val or None, "DiscoveryProfile/TwelveLabs"),
                "edit_type": _feat("edit_type", edit_type_val or None, "DiscoveryProfile/TwelveLabs"),
                "content_format": _feat("content_format", format_val or None, "TwelveLabs"),
                "style": _feat("style", style_val or None, "TwelveLabs"),
                "hook_type": _feat("hook_type", hook_obj.get("type"), "TwelveLabs"),
            },
            "video": {
                "duration": _feat("duration", dur if dur > 0 else None, "TechnicalSignals"),
                "resolution": _feat("resolution", resolution, "TechnicalSignals"),
                "fps": _feat("fps", fps, "TechnicalSignals"),
            },
            "visual_subjects": _feat("visual_subjects", ua.get("visual_subjects") or ua.get("subjects"), "TwelveLabs"),
            "ocr": _feat(
                "ocr",
                {
                    "text": ocr_obj.get("ocr_text") or ocr_obj.get("text") or ua.get("ocr_text"),
                    "first_text_time": ocr_obj.get("first_text_time"),
                    "text_present": bool(ocr_obj.get("text_present", False) or ocr_obj.get("ocr_text")),
                },
                "OCRService",
            ),
            "audio_speech": _feat(
                "audio_speech",
                {
                    "speech_present": bool(ua.get("audio", {}).get("speech_present", False) or ua.get("transcript")),
                    "transcript": ua.get("transcript"),
                    "rms": ts.get("audio_rms"),
                    "silence_ratio": ts.get("silence_ratio"),
                },
                "TwelveLabs/AudioProcessor",
            ),
            "temporal_attention": _feat("nemar_temporal_signal", nemar_timeline, "NEMAR_BBBD"),
            "spatial_saliency": _feat("tinysalnet_spatial_signal", tinysal_signal, "TinySalNet_DHF1K"),
            "attention_intelligence": _feat("attention_intelligence", att_intel, "AttentionIntelligenceService"),
        }

    @staticmethod
    def normalize_competitor(comp: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a verified competitor into canonical feature representation with complete provenance.
        
        Section 5 Minimum Fields:
        - identity: canonical_reel_id, canonical_url, creator, source_queries
        - reach: views/plays, reach_source, reach_valid
        - semantic: entity, niche, sub_niche, topic, edit_type, content_format
        - video: duration, resolution, fps
        - visual: motion, scene cuts, brightness, contrast, complexity, face presence, text presence
        - audio: RMS, silence ratio, speech presence, spectral characteristics
        - attention: NEMAR temporal signal, TinySalNet spatial signal, Attention Intelligence
        - OCR: text segments, confidence, timing
        - relevance: entity_match, topic_match, niche_match, sub_niche_match, edit_type_match, format_match, overall_relevance
        """
        c_id = str(comp.get("shortcode") or comp.get("id") or comp.get("reel_id") or "unknown")
        c_url = str(comp.get("url") or comp.get("source_url") or f"https://www.instagram.com/reel/{c_id}/")
        c_creator = str(comp.get("creator") or comp.get("username") or "unknown")
        s_queries = comp.get("source_queries") or ([comp["source_query"]] if comp.get("source_query") else [])

        # 1. Reach handling (Section 12: missing reach -> reach = null, reach_valid = False)
        raw_views = comp.get("views")
        raw_plays = comp.get("plays")
        raw_reach_val = comp.get("reach_value")

        reach_num: Optional[int] = None
        for cand_val in (raw_views, raw_plays, raw_reach_val):
            if cand_val is not None:
                try:
                    ival = int(cand_val)
                    if ival >= 0:
                        reach_num = ival
                        break
                except (ValueError, TypeError):
                    continue

        reach_valid = reach_num is not None
        reach_source = comp.get("reach_metric") or ("apify_views" if reach_valid else None)

        # 2. Semantic features
        analysis = comp.get("analysis") or {}
        canonical = analysis.get("canonical") or analysis

        c_entity = canonical.get("primary_entity") or comp.get("primary_entity") or canonical.get("primary_subject")
        c_niche = canonical.get("niche") or comp.get("niche")
        c_sub_niche = canonical.get("sub_niche") or comp.get("sub_niche")
        c_topic = canonical.get("topic") or comp.get("topic")
        c_edit_type = comp.get("edit_type") or canonical.get("edit_type") or comp.get("competitor_cluster")
        c_format = canonical.get("format") or comp.get("format")

        # 3. Video technical
        tech = comp.get("technical_signals") or {}
        dur = float(tech.get("duration") or comp.get("duration") or 0.0)
        res_w = tech.get("width") or comp.get("width")
        res_h = tech.get("height") or comp.get("height")
        resolution = f"{res_w}x{res_h}" if res_w and res_h else None
        fps = float(tech.get("fps") or comp.get("fps") or 0.0) if (tech.get("fps") or comp.get("fps")) else None

        # 4. Attention and Visual features from NEMAR and TinySalNet
        att_predictions = comp.get("attention_predictions") or []
        spat_signal = comp.get("dhf1k_spatial_saliency_signal") or analysis.get("dhf1k_spatial_saliency_signal") or {}
        att_intel = comp.get("attention_intelligence") or {}

        # Visual aggregates from attention window feature values
        motion_vals = []
        scene_cut_vals = []
        face_vals = []
        text_vals = []
        brightness_vals = []
        contrast_vals = []
        complexity_vals = []

        for w in att_predictions:
            fv = w.get("feature_values") or {}
            if "motion_magnitude" in fv:
                motion_vals.append(float(fv["motion_magnitude"]))
            if "scene_cut_rate" in fv:
                scene_cut_vals.append(float(fv["scene_cut_rate"]))
            if "face_presence" in fv:
                face_vals.append(float(fv["face_presence"]))
            if "text_presence" in fv:
                text_vals.append(float(fv["text_presence"]))
            if "brightness" in fv:
                brightness_vals.append(float(fv["brightness"]))
            if "contrast" in fv:
                contrast_vals.append(float(fv["contrast"]))
            if "visual_complexity" in fv:
                complexity_vals.append(float(fv["visual_complexity"]))

        visual_summary = {
            "mean_motion": float(np.mean(motion_vals)) if motion_vals else None,
            "scene_cuts": int(sum(c > 0.4 for c in scene_cut_vals)) if scene_cut_vals else (analysis.get("visual", {}).get("scene_count")),
            "face_presence": bool(any(f > 0.2 for f in face_vals[:4])) if face_vals else False,
            "text_presence": bool(any(t > 0.1 for t in text_vals[:4])) if text_vals else False,
            "brightness": float(np.mean(brightness_vals)) if brightness_vals else None,
            "contrast": float(np.mean(contrast_vals)) if contrast_vals else None,
            "complexity": float(np.mean(complexity_vals)) if complexity_vals else None,
        }

        # 5. Audio
        audio_obj = analysis.get("audio") or {}
        audio_summary = {
            "rms": tech.get("audio_rms"),
            "silence_ratio": tech.get("silence_ratio"),
            "speech_presence": bool(audio_obj.get("speech_present", False) or canonical.get("transcript")),
            "spectral_characteristics": tech.get("spectral_centroid"),
        }

        # 6. OCR
        ocr_obj = canonical.get("ocr") or comp.get("ocr") or {}
        ocr_text = canonical.get("ocr_text") or canonical.get("onscreen_text") or ocr_obj.get("ocr_text")

        ocr_summary = {
            "text_segments": ocr_obj.get("segments") or ([str(ocr_text)] if ocr_text else []),
            "confidence": ocr_obj.get("confidence"),
            "timing": {
                "first_text_time": ocr_obj.get("first_text_time"),
            },
            "text_present": bool(ocr_text or ocr_obj.get("text_present", False) or visual_summary["text_presence"]),
        }

        # 7. Relevance breakdown
        relevance_breakdown = {
            "overall_relevance": float(comp.get("video_relevance_score") or comp.get("relevance_score") or 0.0),
            "status": comp.get("video_verification_status") or comp.get("status"),
            "reason": comp.get("video_relevance_reason") or comp.get("relevance_reason"),
            "entity_match": None,
            "topic_match": None,
            "niche_match": None,
            "sub_niche_match": None,
            "edit_type_match": None,
            "format_match": None,
        }

        return {
            "identity": {
                "canonical_reel_id": c_id,
                "canonical_url": c_url,
                "creator": c_creator,
                "source_queries": s_queries,
            },
            "reach": {
                "views": reach_num,
                "plays": reach_num,
                "reach_source": reach_source,
                "reach_valid": reach_valid,
                "likes": comp.get("likes"),
                "comments": comp.get("comments"),
            },
            "semantic": {
                "entity": str(c_entity).strip() if c_entity else None,
                "niche": str(c_niche).strip() if c_niche else None,
                "sub_niche": str(c_sub_niche).strip() if c_sub_niche else None,
                "topic": str(c_topic).strip() if c_topic else None,
                "edit_type": str(c_edit_type).strip() if c_edit_type else None,
                "content_format": str(c_format).strip() if c_format else None,
            },
            "video": {
                "duration": dur if dur > 0 else None,
                "resolution": resolution,
                "fps": fps,
            },
            "visual": visual_summary,
            "audio": audio_summary,
            "attention": {
                "nemar_temporal_signal": att_predictions,
                "tinysalnet_spatial_signal": spat_signal,
                "attention_intelligence": att_intel,
            },
            "ocr": ocr_summary,
            "relevance": relevance_breakdown,
            "provenance": {
                "canonical_reel_id": c_id,
                "source_provider": comp.get("provider", "apify"),
                "has_attention_predictions": bool(att_predictions),
                "has_spatial_signal": bool(spat_signal and spat_signal.get("status") == "SUCCESS"),
                "has_attention_intelligence": bool(att_intel and att_intel.get("status") in {"SUCCESS", "PARTIAL"}),
            },
        }

    @staticmethod
    def partition_subgroups(
        target_norm: Dict[str, Any],
        competitors_norm: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Partition verified competitors into canonical comparison groups (Section 6).
        
        Groups:
        A. same_entity_and_edit_type
        B. same_edit_type
        C. all_verified
        D. highest_reach_same_edit_type (filtered to reach_valid=True)
        """
        target_sem = target_norm.get("semantic", {})
        target_entity = str((target_sem.get("entity") or {}).get("value") or "").lower().strip().replace("-", " ")
        target_edit_type = str((target_sem.get("edit_type") or {}).get("value") or "action montage").lower().strip()

        def _entity_matches(comp: Dict[str, Any]) -> bool:
            if not target_entity:
                return True
            c_sem = comp.get("semantic", {})
            c_ent = str(c_sem.get("entity") or "").lower().strip().replace("-", " ")
            if not c_ent:
                return False
            if target_entity == c_ent:
                return True
            t_tokens = set(target_entity.split())
            c_tokens = set(c_ent.split())
            # Match if any significant token (>3 chars) matches between entities
            sig_overlap = any(w in c_tokens for w in t_tokens if len(w) > 3)
            return bool(sig_overlap or (target_entity in c_ent and len(target_entity) > 3))

        def _edit_type_matches(comp: Dict[str, Any]) -> bool:
            c_sem = comp.get("semantic", {})
            c_et = str(c_sem.get("edit_type") or "").lower().strip()
            c_fmt = str(c_sem.get("content_format") or "").lower().strip()
            combined = f"{c_et} {c_fmt}"
            if any(w in target_edit_type for w in ["action", "montage", "fight", "battle"]):
                return any(w in combined for w in ["action", "montage", "fight", "battle", "combat", "cinematic", "vfx"])
            if any(w in target_edit_type for w in ["emotion", "sad", "tribute"]):
                return any(w in combined for w in ["emotion", "sad", "tribute", "nostalgia"])
            if any(w in target_edit_type for w in ["dialogue", "quote", "speech"]):
                return any(w in combined for w in ["dialogue", "quote", "speech"])
            return target_edit_type in combined or combined in target_edit_type

        same_entity_and_edit_type = [c for c in competitors_norm if _entity_matches(c) and _edit_type_matches(c)]
        same_edit_type = [c for c in competitors_norm if _edit_type_matches(c)]
        all_verified = list(competitors_norm)

        # Primary pool for highest reach derivation
        primary_pool = same_entity_and_edit_type or same_edit_type or all_verified

        # Highest reach subgroup: strictly requires reach_valid=True (Section 12)
        valid_reach_pool = [c for c in primary_pool if c.get("reach", {}).get("reach_valid")]
        sorted_reach = sorted(
            valid_reach_pool,
            key=lambda c: int(c.get("reach", {}).get("views") or 0),
            reverse=True,
        )

        if len(sorted_reach) >= 4:
            cutoff = max(2, int(math.ceil(len(sorted_reach) * 0.5)))
            highest_reach_same_edit_type = sorted_reach[:cutoff]
        else:
            highest_reach_same_edit_type = sorted_reach

        return {
            "same_entity_and_edit_type": same_entity_and_edit_type,
            "same_edit_type": same_edit_type,
            "all_verified": all_verified,
            "highest_reach_same_edit_type": highest_reach_same_edit_type,
        }

    @staticmethod
    def compare_numeric_features(
        target_val: Optional[float],
        comp_vals: List[float],
    ) -> Dict[str, Any]:
        """Compute robust non-causal continuous comparisons for a feature (Section 8).
        
        Calculates:
        - target value
        - competitor median
        - competitor mean
        - absolute difference
        - relative difference
        - percentile/rank when sample size supports it (n >= 3)
        """
        valid_comp = [float(v) for v in comp_vals if v is not None and not math.isnan(v)]
        if not valid_comp:
            return {
                "target_value": target_val,
                "competitor_median": None,
                "competitor_mean": None,
                "absolute_difference": None,
                "relative_difference": None,
                "percentile": None,
                "sample_size": 0,
            }

        med = float(np.median(valid_comp))
        mean_val = float(np.mean(valid_comp))

        abs_diff = None
        rel_diff = None
        pct = None

        if target_val is not None and not math.isnan(target_val):
            abs_diff = round(float(target_val) - med, 4)
            if med != 0:
                rel_diff = round(abs_diff / abs(med), 4)
            if len(valid_comp) >= 3:
                # Empirical percentile of target within competitor distribution
                pct = round(float(np.mean([1.0 if c <= target_val else 0.0 for c in valid_comp]) * 100.0), 1)

        return {
            "target_value": round(float(target_val), 4) if target_val is not None else None,
            "competitor_median": round(med, 4),
            "competitor_mean": round(mean_val, 4),
            "absolute_difference": abs_diff,
            "relative_difference": rel_diff,
            "percentile": pct,
            "sample_size": len(valid_comp),
        }

    @classmethod
    def classify_pattern_strength(
        cls,
        numerator: int,
        denominator: int,
        prevalence: float,
    ) -> Tuple[str, str]:
        """Classify descriptive pattern strength strictly following small-sample safety (Sections 13 & 14).
        
        Returns: (strength_classification, limitation_statement)
        """
        if denominator < 5:
            return (
                "insufficient_sample",
                f"Comparison subgroup sample size is small (n={denominator} < 5); findings are strictly exploratory.",
            )
        elif denominator < 10:
            if prevalence >= 0.70:
                return (
                    "moderate_observed_pattern",
                    f"Observed in {numerator}/{denominator} verified peers (5 <= n < 10); sample size requires cautious descriptive interpretation.",
                )
            elif prevalence >= 0.50:
                return (
                    "moderate_observed_pattern",
                    f"Observed in {numerator}/{denominator} verified peers (5 <= n < 10); represents an observed tendency within this small cohort.",
                )
            else:
                return (
                    "weak_observed_pattern",
                    f"Present in minority ({numerator}/{denominator}) of comparison group; limited pattern support.",
                )
        else:
            if prevalence >= 0.70:
                return (
                    "strong_observed_pattern",
                    f"Observed across {numerator}/{denominator} verified peers; robust descriptive prevalence within the verified cohort.",
                )
            elif prevalence >= 0.50:
                return (
                    "moderate_observed_pattern",
                    f"Observed in {numerator}/{denominator} verified peers; moderate descriptive consistency.",
                )
            else:
                return (
                    "weak_observed_pattern",
                    f"Observed in minority ({numerator}/{denominator}) of comparison group.",
                )

    @classmethod
    def detect_categorical_patterns(
        cls,
        target_norm: Dict[str, Any],
        subgroups: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Detect recurring categorical patterns across comparison groups (Section 9).
        
        Patterns:
        1. opening_visual_focus
        2. text_led_opening
        3. face_presence (early human presence)
        4. dynamic_transitions
        5. concentrated_spatial_focus
        6. distributed_spatial_focus
        7. spoken_opening
        8. high_motion_opening
        9. visual_subject_consistency
        """
        findings: List[Dict[str, Any]] = []

        # Target extraction
        t_ocr = target_norm.get("ocr", {}).get("value", {})
        has_any_text = bool(t_ocr.get("text_present") or t_ocr.get("text"))
        t_has_text = has_any_text and (t_ocr.get("first_text_time") is None or float(t_ocr.get("first_text_time")) <= 2.0)
        t_speech = bool(target_norm.get("audio_speech", {}).get("value", {}).get("speech_present"))

        t_att = target_norm.get("temporal_attention", {}).get("value", [])
        t_face = any(w.get("feature_values", {}).get("face_presence", 0) > 0.2 for w in t_att[:4]) if t_att else False
        t_motion = any(w.get("feature_values", {}).get("motion_magnitude", 0) > 0.03 for w in t_att[:4]) if t_att else False
        t_cuts = sum(w.get("feature_values", {}).get("scene_cut_rate", 0) > 0.4 for w in t_att) if t_att else 0
        t_fast_cuts = bool(t_cuts >= 3)

        t_ai = target_norm.get("attention_intelligence", {}).get("value", {})
        t_spat = t_ai.get("spatial", {})
        t_focal_conc = float(t_spat.get("focal_concentration", 0.0))
        t_disp = float(t_spat.get("dispersion", 1.0))
        t_conc_spatial = bool(t_focal_conc >= 0.50 or t_disp < 0.32)
        t_dist_spatial = bool(t_disp > 0.38 or t_focal_conc < 0.35)

        target_reel_id = target_norm.get("reel_id", "target_reel")

        # Pattern definitions mapping function: competitor -> bool
        def _check_text_led(c: Dict[str, Any]) -> bool:
            ocr = c.get("ocr", {})
            t_time = (ocr.get("timing") or {}).get("first_text_time")
            return bool(ocr.get("text_present") or (t_time is not None and float(t_time) <= 2.0))

        def _check_face(c: Dict[str, Any]) -> bool:
            return bool(c.get("visual", {}).get("face_presence"))

        def _check_cuts(c: Dict[str, Any]) -> bool:
            cuts = c.get("visual", {}).get("scene_cuts")
            return bool(cuts is not None and cuts >= 3)

        def _check_speech(c: Dict[str, Any]) -> bool:
            return bool(c.get("audio", {}).get("speech_presence"))

        def _check_motion(c: Dict[str, Any]) -> bool:
            att = c.get("attention", {}).get("nemar_temporal_signal") or []
            return any(w.get("feature_values", {}).get("motion_magnitude", 0) > 0.03 for w in att[:4])

        def _check_conc_spatial(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            spat = ai.get("spatial", {})
            if spat:
                return bool(float(spat.get("focal_concentration", 0.0)) >= 0.50 or float(spat.get("dispersion", 1.0)) < 0.32)
            # Fallback to TinySalNet summary
            tsig = c.get("attention", {}).get("tinysalnet_spatial_signal") or {}
            disp = float(tsig.get("summary", {}).get("mean_spatial_dispersion", 1.0))
            return bool(disp < 0.32)

        def _check_dist_spatial(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            spat = ai.get("spatial", {})
            if spat:
                return bool(float(spat.get("dispersion", 0.0)) > 0.38 or float(spat.get("focal_concentration", 1.0)) < 0.35)
            tsig = c.get("attention", {}).get("tinysalnet_spatial_signal") or {}
            disp = float(tsig.get("summary", {}).get("mean_spatial_dispersion", 0.0))
            return bool(disp > 0.38)

        def _check_visual_focus(c: Dict[str, Any]) -> bool:
            return _check_conc_spatial(c) or _check_face(c)

        def _check_subject_consistency(c: Dict[str, Any]) -> bool:
            ent = c.get("semantic", {}).get("entity")
            return bool(ent and len(ent) > 2 and ent != "unknown")

        specs = [
            (
                "opening_visual_focus",
                "Opening Visual Focal Anchor",
                _check_visual_focus,
                t_conc_spatial or t_face,
                ["visual.face_presence", "spatial.focal_concentration"],
            ),
            (
                "text_led_opening",
                "Early Typography Hook",
                _check_text_led,
                t_has_text,
                ["ocr.text_present", "ocr.first_text_time"],
            ),
            (
                "face_presence",
                "Early Human Presence",
                _check_face,
                t_face,
                ["visual.face_presence"],
            ),
            (
                "dynamic_transitions",
                "Kinetic Cut Pacing",
                _check_cuts,
                t_fast_cuts,
                ["visual.scene_cuts"],
            ),
            (
                "concentrated_spatial_focus",
                "Concentrated Opening Spatial Saliency",
                _check_conc_spatial,
                t_conc_spatial,
                ["spatial.dispersion", "spatial.focal_concentration"],
            ),
            (
                "distributed_spatial_focus",
                "Distributed Spatial Visual Composition",
                _check_dist_spatial,
                t_dist_spatial,
                ["spatial.dispersion"],
            ),
            (
                "spoken_opening",
                "Spoken Verbal Opening",
                _check_speech,
                t_speech,
                ["audio.speech_presence"],
            ),
            (
                "high_motion_opening",
                "Dynamic Opening Visual Motion",
                _check_motion,
                t_motion,
                ["visual.motion_magnitude"],
            ),
            (
                "visual_subject_consistency",
                "Primary Entity Subject Continuity",
                _check_subject_consistency,
                bool(target_norm.get("semantic", {}).get("entity", {}).get("value")),
                ["semantic.entity"],
            ),
        ]

        # Evaluate across subgroups
        for sg_key, pool in subgroups.items():
            den = len(pool)
            if den == 0:
                continue

            for pat_id, pat_name, fn, t_val, feat_names in specs:
                matching_reels: List[str] = []
                for comp in pool:
                    if fn(comp):
                        matching_reels.append(comp["identity"]["canonical_reel_id"])

                num = len(matching_reels)
                prev = round(num / den, 3)

                strength, lim = cls.classify_pattern_strength(num, den, prev)

                findings.append({
                    "pattern_id": f"{pat_id}:{sg_key}",
                    "pattern_name": pat_name,
                    "subgroup": sg_key,
                    "numerator": num,
                    "denominator": den,
                    "prevalence": prev,
                    "target_value": "present" if t_val else "absent",
                    "competitor_value": f"{num}/{den} present ({prev*100:.1f}%)",
                    "evidence": [
                        f"Observed in {num}/{den} ({prev*100:.1f}%) reels in subgroup '{sg_key}': "
                        f"{', '.join(matching_reels[:5]) if matching_reels else 'none'}"
                    ],
                    "provenance": {
                        "target_reel_id": target_reel_id,
                        "competitor_reel_ids": matching_reels,
                        "subgroup_sample_size": den,
                        "features": feat_names,
                    },
                    "strength": strength,
                    "limitation": lim,
                })

        return findings

    @classmethod
    def detect_attention_patterns(
        cls,
        target_norm: Dict[str, Any],
        subgroups: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Detect recurring Phase 14 Attention Intelligence patterns (Section 10).
        
        Patterns:
        A. concentrated_opening_spatial_focus
        B. stable_focal_attention
        C. opening_temporal_attention_prominence
        D. spatial_focus_movement
        E. focal_competition
        F. text_saliency_alignment
        G. face_saliency_alignment
        H. scene_transition_attention_alignment
        I. temporal_recovery
        J. temporal_attention_drop
        """
        findings: List[Dict[str, Any]] = []
        target_reel_id = target_norm.get("reel_id", "target_reel")

        t_ai = target_norm.get("attention_intelligence", {}).get("value", {})
        t_temp = t_ai.get("temporal", {})
        t_spat = t_ai.get("spatial", {})
        t_rels = t_ai.get("relationships", {})

        def _is_opening_prominent(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            t_obj = ai.get("temporal", {})
            if t_obj:
                peak_t = float(t_obj.get("peak_attention_time", 999.0))
                prom = float(t_obj.get("temporal_persistence", {}).get("opening_retention_ratio", 0.0))
                return bool(peak_t <= 2.5 or prom >= 0.90)
            # Fallback to raw NEMAR
            att = c.get("attention", {}).get("nemar_temporal_signal") or []
            if att:
                baseline = float(att[0].get("attention_score", 0.5))
                for w in att[:4]:
                    if float(w.get("attention_score", 0.0)) - baseline >= 0.04 or float(w.get("peak_prominence", 0.0)) > 0.02:
                        return True
            return False

        def _is_stable_focal(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            rel = ai.get("relationships", {}).get("spatial_temporal_alignment", {})
            return bool(rel.get("alignment_type") == "stable_focal_attention")

        def _is_focal_competition(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            fcomp = ai.get("spatial", {}).get("focal_competition", {})
            return bool(fcomp.get("competition_state") in {"multi_focus", "diffuse"})

        def _is_spatial_movement(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            drift = ai.get("spatial", {}).get("centroid_drift", {})
            return bool(drift.get("drift_intensity") in {"dynamic", "moderate"})

        def _is_text_saliency_aligned(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            talign = ai.get("relationships", {}).get("text_saliency_alignment", {})
            return bool(talign.get("alignment_state") == "congruent_text_anchor")

        def _is_face_saliency_aligned(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            falign = ai.get("relationships", {}).get("face_saliency_alignment", {})
            return bool(falign.get("alignment_state") == "human_anchor")

        def _is_scene_transition_aligned(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            talign = ai.get("relationships", {}).get("transition_attention_alignment", {})
            return bool(talign.get("alignment_state") in {"transition_driven_surge", "kinetic_pacing"})

        def _has_recovery(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            return bool(len(ai.get("temporal", {}).get("attention_recoveries", [])) > 0)

        def _has_drop(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            return bool(len(ai.get("temporal", {}).get("attention_drops", [])) > 0)

        def _is_conc_opening(c: Dict[str, Any]) -> bool:
            ai = c.get("attention", {}).get("attention_intelligence") or {}
            spat = ai.get("spatial", {})
            if spat:
                return bool(float(spat.get("focal_concentration", 0.0)) >= 0.50 or float(spat.get("dispersion", 1.0)) < 0.32)
            tsig = c.get("attention", {}).get("tinysalnet_spatial_signal") or {}
            return bool(float(tsig.get("summary", {}).get("mean_spatial_dispersion", 1.0)) < 0.32)

        specs = [
            (
                "concentrated_opening_spatial_focus",
                "Concentrated Opening Spatial Focus",
                _is_conc_opening,
                bool(float(t_spat.get("focal_concentration", 0.0)) >= 0.50 or float(t_spat.get("dispersion", 1.0)) < 0.32),
                ["attention_intelligence.spatial.focal_concentration", "attention_intelligence.spatial.dispersion"],
            ),
            (
                "stable_focal_attention",
                "Stable Focal Attention Alignment",
                _is_stable_focal,
                bool(t_rels.get("spatial_temporal_alignment", {}).get("alignment_type") == "stable_focal_attention"),
                ["attention_intelligence.relationships.spatial_temporal_alignment"],
            ),
            (
                "opening_temporal_attention_prominence",
                "Opening Temporal Attention Prominence",
                _is_opening_prominent,
                bool(float(t_temp.get("peak_attention_time", 999.0)) <= 2.5 or float(t_temp.get("temporal_persistence", {}).get("opening_retention_ratio", 0.0)) >= 0.90),
                ["attention_intelligence.temporal.peak_attention_time", "nemar_temporal_signal"],
            ),
            (
                "spatial_focus_movement",
                "Dynamic Spatial Saliency Shift",
                _is_spatial_movement,
                bool(t_spat.get("centroid_drift", {}).get("drift_intensity") in {"dynamic", "moderate"}),
                ["attention_intelligence.spatial.centroid_drift"],
            ),
            (
                "focal_competition",
                "Spatial Multi-Region Focal Competition",
                _is_focal_competition,
                bool(t_spat.get("focal_competition", {}).get("competition_state") in {"multi_focus", "diffuse"}),
                ["attention_intelligence.spatial.focal_competition"],
            ),
            (
                "text_saliency_alignment",
                "Congruent Text-Saliency Alignment",
                _is_text_saliency_aligned,
                bool(t_rels.get("text_saliency_alignment", {}).get("alignment_state") == "congruent_text_anchor"),
                ["attention_intelligence.relationships.text_saliency_alignment"],
            ),
            (
                "face_saliency_alignment",
                "Human Face Focal Anchoring",
                _is_face_saliency_aligned,
                bool(t_rels.get("face_saliency_alignment", {}).get("alignment_state") == "human_anchor"),
                ["attention_intelligence.relationships.face_saliency_alignment"],
            ),
            (
                "scene_transition_attention_alignment",
                "Transition-Driven Attention Surge",
                _is_scene_transition_aligned,
                bool(t_rels.get("transition_attention_alignment", {}).get("alignment_state") in {"transition_driven_surge", "kinetic_pacing"}),
                ["attention_intelligence.relationships.transition_attention_alignment"],
            ),
            (
                "temporal_recovery",
                "Post-Dip Attention Recovery Event",
                _has_recovery,
                bool(len(t_temp.get("attention_recoveries", [])) > 0),
                ["attention_intelligence.temporal.attention_recoveries"],
            ),
            (
                "temporal_attention_drop",
                "Mid-Reel Attention Drop Event",
                _has_drop,
                bool(len(t_temp.get("attention_drops", [])) > 0),
                ["attention_intelligence.temporal.attention_drops"],
            ),
        ]

        for sg_key, pool in subgroups.items():
            den = len(pool)
            if den == 0:
                continue

            for pat_id, pat_name, fn, t_val, feat_names in specs:
                matching_reels: List[str] = []
                for comp in pool:
                    if fn(comp):
                        matching_reels.append(comp["identity"]["canonical_reel_id"])

                num = len(matching_reels)
                prev = round(num / den, 3)

                strength, lim = cls.classify_pattern_strength(num, den, prev)

                findings.append({
                    "pattern_id": f"att_{pat_id}:{sg_key}",
                    "pattern_name": pat_name,
                    "subgroup": sg_key,
                    "numerator": num,
                    "denominator": den,
                    "prevalence": prev,
                    "target_value": "present" if t_val else "absent",
                    "competitor_value": f"{num}/{den} present ({prev*100:.1f}%)",
                    "evidence": [
                        f"{num}/{den} ({prev*100:.1f}%) competitors in subgroup '{sg_key}' exhibit {pat_name.lower()}."
                    ],
                    "provenance": {
                        "target_reel_id": target_reel_id,
                        "competitor_reel_ids": matching_reels,
                        "subgroup_sample_size": den,
                        "features": feat_names,
                    },
                    "strength": strength,
                    "limitation": lim,
                })

        return findings

    @classmethod
    def perform_gap_analysis(
        cls,
        target_norm: Dict[str, Any],
        patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Perform descriptive gap analysis for every recurring pattern (Section 11).
        
        Strictly descriptive: target vs comparison group aggregate.
        """
        gaps: List[Dict[str, Any]] = []

        for p in patterns:
            num = p["numerator"]
            den = p["denominator"]
            prev = p["prevalence"]
            sg = p["subgroup"]
            pat_name = p["pattern_name"]
            t_val = p["target_value"]

            # Only analyze patterns with meaningful representation (prevalence >= 0.50)
            if prev < 0.50:
                continue

            target_matches_majority = (t_val == "present")

            if target_matches_majority:
                gap_status = "aligned"
                desc = (
                    f"The target exhibits '{pat_name}' (present), which aligns with the majority "
                    f"({num}/{den}, {prev*100:.1f}%) of the '{sg}' comparison cohort."
                )
            else:
                gap_status = "divergent"
                desc = (
                    f"The target exhibits '{pat_name}' as {t_val}, differing from {num}/{den} "
                    f"({prev*100:.1f}%) of the verified '{sg}' comparison cohort."
                )

            gaps.append({
                "pattern_id": p["pattern_id"],
                "pattern_name": pat_name,
                "subgroup": sg,
                "numerator": num,
                "denominator": den,
                "prevalence": prev,
                "target_value": t_val,
                "competitor_aggregate": f"{num}/{den} ({prev*100:.1f}%)",
                "gap_status": gap_status,
                "interpretation": desc,
                "limitation": (
                    "Observational comparison within the selected verified competitor group; "
                    "does not establish a causal relationship with viewer retention or distribution."
                ),
                "provenance": p.get("provenance"),
            })

        return gaps

    @classmethod
    def analyze_reach_aware_patterns(
        cls,
        target_norm: Dict[str, Any],
        competitors_norm: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform descriptive reach-aware comparisons (Section 12).
        
        Strictly descriptive:
        - raw views/plays when available
        - reach percentiles
        - excludes records where reach = null
        - NO causal claims ("high views caused by feature X" is FORBIDDEN).
        """
        valid_reach_comps = [c for c in competitors_norm if c.get("reach", {}).get("reach_valid")]
        excluded_count = len(competitors_norm) - len(valid_reach_comps)

        if not valid_reach_comps:
            return {
                "status": "INSUFFICIENT_DATA",
                "valid_reach_count": 0,
                "excluded_count": excluded_count,
                "observations": ["No verified competitors provided valid numeric view counts."],
                "limitations": "Missing reach data precludes reach-stratified analysis.",
            }

        sorted_comps = sorted(
            valid_reach_comps,
            key=lambda c: int(c.get("reach", {}).get("views") or 0),
            reverse=True,
        )

        views_list = [int(c["reach"]["views"]) for c in sorted_comps]
        median_reach = float(np.median(views_list))
        min_reach = int(views_list[-1])
        max_reach = int(views_list[0])

        top_half_n = max(1, len(sorted_comps) // 2)
        top_cohort = sorted_comps[:top_half_n]
        lower_cohort = sorted_comps[top_half_n:]

        # Descriptive feature comparison between top-reach and lower-reach cohorts
        def _cohort_prev(cohort: List[Dict[str, Any]], fn) -> Tuple[int, int]:
            cnt = sum(1 for c in cohort if fn(c))
            return cnt, len(cohort)

        text_fn = lambda c: bool(c.get("ocr", {}).get("text_present"))
        face_fn = lambda c: bool(c.get("visual", {}).get("face_presence"))
        cut_fn = lambda c: bool(c.get("visual", {}).get("scene_cuts", 0) and c["visual"]["scene_cuts"] >= 3)

        top_text_num, top_text_den = _cohort_prev(top_cohort, text_fn)
        low_text_num, low_text_den = _cohort_prev(lower_cohort, text_fn)

        top_face_num, top_face_den = _cohort_prev(top_cohort, face_fn)
        low_face_num, low_face_den = _cohort_prev(lower_cohort, face_fn)

        top_cut_num, top_cut_den = _cohort_prev(top_cohort, cut_fn)
        low_cut_num, low_cut_den = _cohort_prev(lower_cohort, cut_fn)

        observations = [
            f"Verified comparison cohort reach spans from {min_reach:,} to {max_reach:,} views (cohort median: {int(median_reach):,} views; n={len(valid_reach_comps)}).",
            f"Within the top-reach partition (n={top_text_den}), on-screen text was observed in {top_text_num}/{top_text_den} reels, compared to {low_text_num}/{low_text_den} in the lower partition.",
            f"Early human presence appeared in {top_face_num}/{top_face_den} top-reach reels vs {low_face_num}/{low_face_den} lower-partition reels.",
            f"Dynamic scene cut pacing (3+ cuts) was observed in {top_cut_num}/{top_cut_den} top-reach reels vs {low_cut_num}/{low_cut_den} lower-partition reels.",
        ]

        if excluded_count > 0:
            observations.append(f"{excluded_count} competitor(s) with missing or unparseable view metrics were excluded from reach calculations.")

        return {
            "status": "SUCCESS",
            "valid_reach_count": len(valid_reach_comps),
            "excluded_count": excluded_count,
            "cohort_median_views": median_reach,
            "min_views": min_reach,
            "max_views": max_reach,
            "observations": observations,
            "top_reach_cohort_size": top_half_n,
            "stratified_tally": {
                "early_text": {"top": f"{top_text_num}/{top_text_den}", "lower": f"{low_text_num}/{low_text_den}"},
                "face_presence": {"top": f"{top_face_num}/{top_face_den}", "lower": f"{low_face_num}/{low_face_den}"},
                "dynamic_cuts": {"top": f"{top_cut_num}/{top_cut_den}", "lower": f"{low_cut_num}/{low_cut_den}"},
            },
            "limitations": (
                "Reach data is utilized strictly as an observational sorting variable. "
                "Observable correlations with view counts do NOT establish causality; "
                "distribution outcomes depend on external algorithmic distribution, creator graph, and audio trends."
            ),
        }

    @classmethod
    def deduplicate_patterns(cls, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate semantically overlapping patterns into unique findings (Section 15).
        
        Ensures:
        - Unique pattern_id
        - No synonym duplicates describing identical underlying signals
        """
        seen_ids: set[str] = set()
        deduped: List[Dict[str, Any]] = []

        # Synonym equivalence clusters: map alias -> canonical_pattern_id_root
        alias_map = {
            "opening_visual_concentration": "concentrated_opening_spatial_focus",
            "opening_spatial_focus_concentration": "concentrated_opening_spatial_focus",
            "opening_focal_concentration": "concentrated_opening_spatial_focus",
            "text_led_opening": "text_led_opening",
            "early_typography": "text_led_opening",
        }

        for p in patterns:
            raw_id = p["pattern_id"]
            root = raw_id.split(":")[0]
            canonical_root = alias_map.get(root, root)
            canonical_id = f"{canonical_root}:{p.get('subgroup', 'all')}"

            if canonical_id in seen_ids:
                continue

            seen_ids.add(canonical_id)
            p_copy = dict(p)
            p_copy["pattern_id"] = canonical_id
            deduped.append(p_copy)

        return deduped

    @classmethod
    def formulate_creative_implications(
        cls,
        target_norm: Dict[str, Any],
        gaps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Formulate rigorous evidence-backed recommendations from gap analysis (Section 17).
        
        Follows strictly:
        OBSERVATION -> EVIDENCE -> TARGET/COMPETITOR COMPARISON -> RECOMMENDATION -> LIMITATION
        """
        implications: List[Dict[str, Any]] = []

        rec_catalog = {
            "concentrated_opening_spatial_focus": (
                "Consider framing the opening composition so that the primary visual subject or character "
                "occupies a single dominant focal region during the first 1.5 seconds."
            ),
            "text_led_opening": (
                "Consider introducing high-contrast on-screen typography or a punchy hook card within the "
                "first 1.5 seconds to establish narrative framing for scrollers viewing without audio."
            ),
            "early_human_presence": (
                "Consider introducing character face presence or direct subject gaze within the opening "
                "2 seconds to establish an immediate human focal anchor."
            ),
            "dynamic_transitions": (
                "Consider introducing a scene reframe, match cut, or focal cutaway every 1.5s to 2.5s "
                "to maintain kinetic visual progression."
            ),
            "opening_temporal_attention_prominence": (
                "Consider front-loading a decisive visual movement or graphic beat within the opening 1.5 "
                "seconds to establish an immediate relative engagement peak."
            ),
            "stable_focal_attention": (
                "Align audio accents and visual focal points so that peak visual saliency and acoustic events "
                "converge onto the same subject anchor."
            ),
        }

        for g in gaps:
            # Generate recommendations primarily when target diverges from observed dominant peer pattern
            if g.get("gap_status") == "divergent" and g.get("prevalence", 0) >= 0.50:
                p_root = g["pattern_id"].split(":")[0].replace("att_", "")
                rec_text = rec_catalog.get(
                    p_root,
                    f"Consider testing creative variants incorporating '{g['pattern_name']}' to align with verified peer conventions."
                )

                implications.append({
                    "pattern_name": g["pattern_name"],
                    "observation": f"The target exhibits '{g['pattern_name']}' as {g['target_value']}.",
                    "evidence": (
                        f"{g['numerator']}/{g['denominator']} ({g['prevalence']*100:.1f}%) competitors in the "
                        f"'{g['subgroup']}' subgroup exhibit {g['pattern_name'].lower()}."
                    ),
                    "target_competitor_comparison": g["interpretation"],
                    "recommendation": rec_text,
                    "limitation": g["limitation"],
                    "provenance": g.get("provenance"),
                })

        return implications

    @classmethod
    def analyze(
        cls,
        target_profile: Dict[str, Any],
        target_attention_intelligence: Optional[Dict[str, Any]],
        verified_competitors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Main Phase 15 analytical entry point (Section 3).
        
        Consumes existing target + competitor outputs.
        Does NOT perform discovery, downloading, Twelve Labs calls, or model inference.
        """
        # 1. Canonical target normalization
        target_norm = cls.normalize_target(target_profile, target_attention_intelligence)

        # 2. Canonical competitor normalization
        competitors_norm = [cls.normalize_competitor(c) for c in verified_competitors]

        # 3. Subgroup partitioning
        subgroups = cls.partition_subgroups(target_norm, competitors_norm)

        # 4. Continuous feature comparisons across primary subgroup
        primary_pool = (
            subgroups["same_entity_and_edit_type"]
            or subgroups["same_edit_type"]
            or subgroups["all_verified"]
        )

        numeric_comparisons = cls._compute_all_numeric_comparisons(target_norm, primary_pool)

        # 5. Categorical pattern detection
        raw_cat_patterns = cls.detect_categorical_patterns(target_norm, subgroups)

        # 6. Attention-based pattern detection
        raw_att_patterns = cls.detect_attention_patterns(target_norm, subgroups)

        # 7. Deduplicate patterns
        all_patterns = cls.deduplicate_patterns(raw_cat_patterns + raw_att_patterns)

        # 8. Target vs Competitor Gap Analysis
        gap_analysis = cls.perform_gap_analysis(target_norm, all_patterns)

        # 9. Reach-aware descriptive observations
        reach_analysis = cls.analyze_reach_aware_patterns(target_norm, competitors_norm)

        # 10. Evidence-backed creative implications
        creative_implications = cls.formulate_creative_implications(target_norm, gap_analysis)

        # Extract primary subgroup counts for fast consumption
        p_subgroup_name = (
            "same_entity_and_edit_type" if subgroups["same_entity_and_edit_type"]
            else ("same_edit_type" if subgroups["same_edit_type"] else "all_verified")
        )

        primary_patterns = [p for p in all_patterns if p.get("subgroup") == p_subgroup_name]

        return {
            "schema_version": "1.0.0",
            "status": "SUCCESS" if competitors_norm else "SKIPPED_NO_COMPETITORS",
            "verified_competitor_count": len(competitors_norm),
            "unique_creators": len(set(c["identity"]["creator"] for c in competitors_norm)),
            "primary_subgroup": p_subgroup_name,
            "target_normalized": target_norm,
            "competitors_normalized": competitors_norm,
            "subgroup_counts": {k: len(v) for k, v in subgroups.items()},
            "numeric_comparisons": numeric_comparisons,
            "recurring_patterns": primary_patterns,
            "all_subgroup_patterns": all_patterns,
            "gap_analysis": gap_analysis,
            "reach_aware_observations": reach_analysis,
            "creative_implications": creative_implications,
            "provenance": {
                "target_reel_id": target_norm.get("reel_id"),
                "competitor_ids": [c["identity"]["canonical_reel_id"] for c in competitors_norm],
                "subgroups_evaluated": list(subgroups.keys()),
                "zero_network_calls": True,
                "zero_duplicate_inference": True,
            },
        }

    @classmethod
    def _compute_all_numeric_comparisons(
        cls,
        target_norm: Dict[str, Any],
        comp_pool: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute numeric feature comparisons against target Reel across continuous signals."""
        # Visual motion
        t_motion = None
        t_att = target_norm.get("temporal_attention", {}).get("value", [])
        if t_att:
            m_vals = [float(w.get("feature_values", {}).get("motion_magnitude", 0)) for w in t_att]
            t_motion = float(np.mean(m_vals)) if m_vals else None

        comp_motions = [c.get("visual", {}).get("mean_motion") for c in comp_pool]

        # Scene cuts
        t_cuts = float(sum(w.get("feature_values", {}).get("scene_cut_rate", 0) > 0.4 for w in t_att)) if t_att else None
        comp_cuts = [float(c.get("visual", {}).get("scene_cuts") or 0.0) for c in comp_pool]

        # Spatial dispersion
        t_ai = target_norm.get("attention_intelligence", {}).get("value", {})
        t_disp = t_ai.get("spatial", {}).get("dispersion")
        if t_disp is None:
            t_tsig = target_norm.get("spatial_saliency", {}).get("value", {})
            t_disp = t_tsig.get("summary", {}).get("mean_spatial_dispersion")

        comp_disps = []
        for c in comp_pool:
            c_ai = c.get("attention", {}).get("attention_intelligence") or {}
            c_disp = c_ai.get("spatial", {}).get("dispersion")
            if c_disp is None:
                c_tsig = c.get("attention", {}).get("tinysalnet_spatial_signal") or {}
                c_disp = c_tsig.get("summary", {}).get("mean_spatial_dispersion")
            if c_disp is not None:
                comp_disps.append(float(c_disp))

        # Peak attention score
        t_peak = t_ai.get("temporal", {}).get("peak_attention_score")
        if t_peak is None and t_att:
            t_peak = max(float(w.get("attention_score", 0.0)) for w in t_att)

        comp_peaks = []
        for c in comp_pool:
            c_ai = c.get("attention", {}).get("attention_intelligence") or {}
            c_peak = c_ai.get("temporal", {}).get("peak_attention_score")
            if c_peak is None:
                c_att = c.get("attention", {}).get("nemar_temporal_signal") or []
                if c_att:
                    c_peak = max(float(w.get("attention_score", 0.0)) for w in c_att)
            if c_peak is not None:
                comp_peaks.append(float(c_peak))

        return {
            "motion_magnitude": cls.compare_numeric_features(t_motion, comp_motions),
            "scene_cut_rate": cls.compare_numeric_features(t_cuts, comp_cuts),
            "spatial_dispersion": cls.compare_numeric_features(float(t_disp) if t_disp is not None else None, comp_disps),
            "peak_attention_score": cls.compare_numeric_features(float(t_peak) if t_peak is not None else None, comp_peaks),
        }
