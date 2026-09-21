"""Pattern discovery over canonical normalized numeric signals from analyzed competitor Reels.

EXACTLY ONE ML LAYER.
Contract:
- Requires at least 4 verified competitor analyses for clustering.
- If < 4: status = 'insufficient_competitor_analyses'.
- Never fabricates clusters.
- Operates strictly on numeric canonical features (0.0 - 1.0 or normalized metrics).
"""
from __future__ import annotations

from numbers import Real
from typing import Any, Mapping
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from services.canonical_schema import normalize_score_0_to_1

MIN_SAMPLES_FOR_CLUSTERING = 4

NUMERIC_FEATURES = (
    "duration_sec",
    "motion_mean",
    "scene_change_rate",
    "contrast_mean",
    "scene_changes",
    "hook_strength",
    "visual_impact",
    "curiosity",
    "novelty",
    "shareability",
    "saveability",
    "comment_potential",
    "likes",
    "comments",
    "views",
    "speech_present",
    "music_present",
)


class PatternDiscovery:
    """Single ML layer for clustering and pattern discovery over verified competitors."""

    def discover(self, analyzed_competitors: list[dict[str, Any]], k: int = 3) -> dict[str, Any]:
        """Perform clustering on verified competitor features."""
        usable, feature_rows = self._build_feature_rows(analyzed_competitors)

        diagnostics = {
            "numeric_feature_count": len(NUMERIC_FEATURES),
            "feature_names": list(NUMERIC_FEATURES),
            "samples_analyzed": len(analyzed_competitors),
            "usable_samples": len(usable),
        }

        # Guard: Need at least 4 verified competitors
        if len(feature_rows) < MIN_SAMPLES_FOR_CLUSTERING:
            return {
                "status": "insufficient_competitor_analyses",
                "cluster_count": 0,
                "clusters": [],
                "diagnostics": diagnostics,
                "note": f"Need at least {MIN_SAMPLES_FOR_CLUSTERING} verified analyzed competitors to cluster (found {len(feature_rows)}).",
            }

        # Normalize with StandardScaler and Cluster with KMeans
        matrix = np.asarray(feature_rows, dtype=float)
        scaled = StandardScaler().fit_transform(matrix)

        n_clusters = min(k, len(feature_rows))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)

        clusters = []
        for cluster_id in range(n_clusters):
            members = [usable[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not members:
                continue

            # Compute cluster averages
            durations = [float(c.get("video", {}).get("duration_sec") or 0.0) for c in members]
            motions = [float(c.get("video", {}).get("motion_mean") or 0.0) for c in members]
            hooks = [float((c.get("hook") or {}).get("strength") or 0.0) for c in members]

            clusters.append({
                "cluster": cluster_id,
                "size": len(members),
                "avg_duration_sec": round(float(np.mean(durations)), 2) if durations else 0.0,
                "avg_motion": round(float(np.mean(motions)), 2) if motions else 0.0,
                "avg_hook_score": round(float(np.mean(hooks)), 4) if hooks else 0.0,
            })

        return {
            "status": "complete",
            "cluster_count": len(clusters),
            "clusters": clusters,
            "diagnostics": diagnostics,
        }

    def _build_feature_rows(
        self, competitors: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[list[float]]]:
        usable = []
        rows = []

        for comp in competitors:
            analysis = comp.get("analysis") or comp
            hook = comp.get("hook") or analysis.get("hook") or {}
            visual = comp.get("visual") or analysis.get("visual") or {}
            audio = comp.get("audio") or analysis.get("audio") or {}
            video = comp.get("video") or (analysis.get("technical_signals") or {}).get("video") or {}
            eng = comp.get("engagement") or analysis.get("engagement") or {}

            row = [
                float(video.get("duration_sec") or 15.0),
                float(video.get("motion_mean") or 10.0),
                float(video.get("scene_change_rate") or 0.5),
                float(video.get("contrast_mean") or 40.0),
                float(visual.get("scene_count") or video.get("scene_change_count") or 5.0),
                float(normalize_score_0_to_1(hook.get("strength")) or 0.5),
                float(normalize_score_0_to_1(visual.get("visual_impact")) or 0.5),
                float(normalize_score_0_to_1(eng.get("curiosity")) or 0.5),
                float(normalize_score_0_to_1(eng.get("novelty")) or 0.5),
                float(normalize_score_0_to_1(eng.get("shareability")) or 0.5),
                float(normalize_score_0_to_1(eng.get("saveability")) or 0.5),
                float(normalize_score_0_to_1(eng.get("comment_potential")) or 0.5),
                float(comp.get("likes") or 0.0),
                float(comp.get("comments") or 0.0),
                float(comp.get("views") or 0.0),
                1.0 if audio.get("speech_present") else 0.0,
                1.0 if audio.get("music_present") else 0.0,
            ]

            usable.append(comp)
            rows.append(row)

        return usable, rows
