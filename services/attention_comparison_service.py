"""Attention Comparison Service & Reach-Attention Association Analysis.

Compares the uploaded Reel predicted visual-attention trajectory against verified competitor trajectories.
Computes comparative metrics and reach-attention associations without claiming causality or commercial scroll-stop.

Enforces Section 15 & 16:
- Replaces universal >= 0.70 threshold with relative baseline delta / peak prominence.
- Implements "Reach–attention association" layer (NEVER claiming 'attention causes virality' or 'attention caused views').
- Reports sample size, unique creators, and explicit scientific limitations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class AttentionComparisonService:
    """Service comparing temporal visual attention curves across Reels and computing reach associations."""

    @staticmethod
    def _extract_curve_stats(predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract empirical curve metrics from temporal attention predictions using relative measures."""
        if not predictions:
            return None

        # Strictly sort chronologically
        sorted_preds = sorted(
            predictions,
            key=lambda x: (float(x.get("start_time", x.get("start", 0))), float(x.get("end_time", x.get("end", 0))))
        )

        scores = [float(p["attention_score"]) for p in sorted_preds if "attention_score" in p]
        if not scores:
            return None

        # Peak
        peak_idx = int(np.argmax(scores))
        peak_score = round(scores[peak_idx], 4)
        peak_time = round(float(sorted_preds[peak_idx].get("start_time", sorted_preds[peak_idx].get("start", 0))), 3)

        # Lowest
        min_idx = int(np.argmin(scores))
        lowest_score = round(scores[min_idx], 4)
        lowest_time = round(float(sorted_preds[min_idx].get("start_time", sorted_preds[min_idx].get("start", 0))), 3)

        # Baseline (opening 2.0s or first 4 windows)
        baseline_slice = scores[: min(4, len(scores))]
        baseline_score = float(np.mean(baseline_slice)) if baseline_slice else 0.5

        # First major event: relative delta >= 0.04 or local peak with stimulus (Section 3 & 15)
        first_event_time: Optional[float] = None
        for p in sorted_preds:
            score = float(p.get("attention_score", 0))
            feats = p.get("feature_values", {})
            has_stimulus = any([
                feats.get("face_presence", 0) > 0.2,
                feats.get("text_presence", 0) > 0.1,
                feats.get("scene_cut_rate", 0) > 0.5,
                feats.get("motion_magnitude", 0) > 0.03,
                feats.get("audio_speech_presence", 0) > 0.3,
            ])
            if (score - baseline_score >= 0.04 or p.get("is_local_peak")) and has_stimulus:
                first_event_time = round(float(p.get("start_time", p.get("start", 0))), 3)
                break

        # Opening strength (0.0–1.0s)
        opening_scores = [
            scores[i] for i, p in enumerate(sorted_preds)
            if float(p.get("start_time", p.get("start", 0))) <= 1.0
        ]
        opening_strength = round(float(np.mean(opening_scores)), 4) if opening_scores else round(scores[0], 4)
        opening_delta = round(opening_strength - baseline_score, 4)

        # Decline slope (from peak to subsequent 2 seconds if present)
        subsequent = scores[peak_idx : min(len(scores), peak_idx + 4)]
        if len(subsequent) > 1:
            decline_slope = round(float(subsequent[0] - subsequent[-1]) / (len(subsequent) * 0.5), 4)
        else:
            decline_slope = 0.0

        # Recovery behavior (detect secondary rise after a trough)
        recovery_detected = False
        if len(scores) >= 6:
            mid = len(scores) // 2
            if np.min(scores[:mid]) < np.max(scores[mid:]):
                recovery_detected = True

        stability = round(max(0.0, min(1.0, 1.0 - float(np.std(scores)) * 2.0)), 4)

        return {
            "mean_score": round(float(np.mean(scores)), 4),
            "baseline_score": round(baseline_score, 4),
            "peak_score": peak_score,
            "peak_time": peak_time,
            "lowest_score": lowest_score,
            "lowest_time": lowest_time,
            "first_event_time": first_event_time,
            "opening_strength": opening_strength,
            "opening_delta": opening_delta,
            "decline_slope": decline_slope,
            "recovery_detected": recovery_detected,
            "stability": stability,
            "curve": [round(s, 4) for s in scores],
        }

    @classmethod
    def analyze_reach_attention_association(
        cls,
        verified_competitors: List[Dict[str, Any]],
        competitor_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze empirical association between competitor reach and predicted visual-attention potential.

        Section 16:
        - Terminology: 'Reach–attention association'
        - NEVER: 'attention causes virality' or 'attention caused views'
        - Permitted: 'Higher-reach verified competitors showed greater average opening attention increase in this sample.'
        - Reports: sample size, unique creators count, limitations.
        """
        sample_size = len(competitor_stats)
        if sample_size == 0:
            return {
                "association_title": "Reach–attention association",
                "sample_size": 0,
                "unique_creators": 0,
                "correlation": None,
                "summary": "No verified competitor attention data available for reach-attention analysis.",
                "insights": [],
                "limitations": "Zero verified competitors available in sample.",
            }

        unique_creators = len(set(
            str(c.get("creator") or c.get("username") or "unknown").lower()
            for c in verified_competitors
        ))

        # Gather paired data: (reach_value, opening_strength, peak_score, opening_delta)
        pairs: List[Dict[str, Any]] = []
        for cand in verified_competitors:
            cid = str(cand.get("shortcode") or cand.get("id") or "")
            if cid in competitor_stats:
                stats = competitor_stats[cid]
                reach = int(cand.get("reach_value") or cand.get("views") or cand.get("plays") or 0)
                pairs.append({
                    "id": cid,
                    "reach": reach,
                    "opening_strength": stats.get("opening_strength", 0.5),
                    "opening_delta": stats.get("opening_delta", 0.0),
                    "peak_score": stats.get("peak_score", 0.5),
                })

        insights = []
        correlation_val = None

        if len(pairs) >= 5:
            # Descriptive median split comparison
            sorted_pairs = sorted(pairs, key=lambda x: x["reach"], reverse=True)
            mid = len(sorted_pairs) // 2
            high_reach_group = sorted_pairs[:mid]
            lower_reach_group = sorted_pairs[mid:]

            high_opening_avg = float(np.mean([p["opening_strength"] for p in high_reach_group]))
            low_opening_avg = float(np.mean([p["opening_strength"] for p in lower_reach_group]))
            high_delta_avg = float(np.mean([p["opening_delta"] for p in high_reach_group]))
            low_delta_avg = float(np.mean([p["opening_delta"] for p in lower_reach_group]))

            reaches = [p["reach"] for p in pairs]
            openings = [p["opening_strength"] for p in pairs]
            if len(set(reaches)) > 1 and len(set(openings)) > 1:
                correlation_val = round(float(np.corrcoef(reaches, openings)[0, 1]), 3)

            if high_delta_avg > low_delta_avg:
                insights.append(
                    f"Higher-reach verified competitors showed greater average opening attention increase in this sample "
                    f"({high_delta_avg:+.3f} vs {low_delta_avg:+.3f})."
                )
            else:
                insights.append(
                    f"Higher-reach verified competitors showed comparable opening attention patterns "
                    f"({high_opening_avg:.2f} vs {low_opening_avg:.2f}) relative to lower-reach verified reels in this sample."
                )

            summary = (
                f"In this verified sample (n={len(pairs)}, {unique_creators} creators), reach-attention association "
                f"was evaluated comparing top vs lower-reach tiers."
            )
        else:
            summary = (
                f"Sample size (n={len(pairs)}, {unique_creators} creator{'s' if unique_creators != 1 else ''}) is "
                f"too small to support inferential correlation analysis. Descriptive observations only."
            )
            if pairs:
                avg_op = float(np.mean([p["opening_strength"] for p in pairs]))
                insights.append(f"Sample mean opening predicted visual-attention potential was {avg_op:.2f}.")

        limitations = (
            "Reach–attention associations reflect observational patterns across a small verified sample. "
            "They do NOT establish that visual attention causes views, reach, or commercial virality. "
            "Platform distribution involves algorithmic ranking, audience targeting, and audio trends."
        )

        return {
            "association_title": "Reach–attention association",
            "sample_size": len(pairs),
            "unique_creators": unique_creators,
            "correlation": correlation_val,
            "summary": summary,
            "insights": insights,
            "limitations": limitations,
        }

    @classmethod
    def compare(
        cls,
        your_predictions: List[Dict[str, Any]],
        competitor_predictions_map: Dict[str, List[Dict[str, Any]]],
        verified_competitors: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compare uploaded Reel attention against all verified competitors and compute reach association."""
        your_stats = cls._extract_curve_stats(your_predictions)
        if not your_stats:
            return {
                "status": "INSUFFICIENT_DATA",
                "your_stats": None,
                "competitor_stats": {},
                "summary": "Uploaded Reel has no valid attention predictions.",
                "insights": [],
                "reach_attention_association": None,
            }

        competitor_stats: Dict[str, Any] = {}
        for cid, preds in competitor_predictions_map.items():
            c_stats = cls._extract_curve_stats(preds)
            if c_stats:
                competitor_stats[cid] = c_stats

        if not competitor_stats:
            return {
                "status": "NO_COMPETITOR_ATTENTION_DATA",
                "your_stats": your_stats,
                "competitor_stats": {},
                "summary": "No verified competitor attention data available for comparative benchmarking.",
                "insights": [],
                "reach_attention_association": None,
            }

        # Comparative aggregates
        comp_openings = [s["opening_strength"] for s in competitor_stats.values()]
        comp_peaks = [s["peak_score"] for s in competitor_stats.values()]
        comp_stabilities = [s["stability"] for s in competitor_stats.values()]
        valid_events = [s["first_event_time"] for s in competitor_stats.values() if s["first_event_time"] is not None]

        avg_comp_opening = round(float(np.mean(comp_openings)), 4)
        avg_comp_peak = round(float(np.mean(comp_peaks)), 4)
        avg_comp_first_event = round(float(np.mean(valid_events)), 2) if valid_events else None
        avg_comp_stability = round(float(np.mean(comp_stabilities)), 4)

        # Gap analysis
        opening_diff = round(your_stats["opening_strength"] - avg_comp_opening, 4)
        peak_diff = round(your_stats["peak_score"] - avg_comp_peak, 4)
        if your_stats.get("first_event_time") is not None and avg_comp_first_event is not None:
            event_time_diff = round(your_stats["first_event_time"] - avg_comp_first_event, 2)
        else:
            event_time_diff = None

        insights = []
        if opening_diff >= 0.05:
            insights.append(
                f"Your opening attention potential ({your_stats['opening_strength']:.2f}) outperforms competitor average ({avg_comp_opening:.2f}) by +{opening_diff:.2f}."
            )
        elif opening_diff <= -0.05:
            insights.append(
                f"Your opening attention potential ({your_stats['opening_strength']:.2f}) trails competitor benchmark ({avg_comp_opening:.2f}) by {opening_diff:.2f}."
            )
        else:
            insights.append(
                f"Your opening attention potential ({your_stats['opening_strength']:.2f}) closely aligns with competitor average ({avg_comp_opening:.2f})."
            )

        if event_time_diff is not None:
            if event_time_diff > 0.5:
                insights.append(
                    f"Competitors trigger their first major attention event earlier (average: {avg_comp_first_event:.2f}s vs your {your_stats['first_event_time']:.2f}s)."
                )
            elif event_time_diff < -0.3:
                insights.append(
                    f"Your first major attention event ({your_stats['first_event_time']:.2f}s) occurs faster than competitor benchmark ({avg_comp_first_event:.2f}s)."
                )

        # Compute Reach-Attention Association
        reach_association = cls.analyze_reach_attention_association(
            verified_competitors or [], competitor_stats
        )

        return {
            "status": "SUCCESS",
            "your_stats": your_stats,
            "competitor_stats": competitor_stats,
            "benchmark": {
                "competitor_count": len(competitor_stats),
                "avg_opening_strength": avg_comp_opening,
                "avg_peak_score": avg_comp_peak,
                "avg_first_event_time": avg_comp_first_event,
                "avg_stability": avg_comp_stability,
                "opening_gap": opening_diff,
                "peak_gap": peak_diff,
                "first_event_gap_sec": event_time_diff,
            },
            "insights": insights,
            "reach_attention_association": reach_association,
        }
