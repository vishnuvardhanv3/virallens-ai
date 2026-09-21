"""Competitor Agent - Centralized candidate ranking, performance scoring, and diverse selection.

Implements Sections 8, 9, 10, 12, 13:
- Primary reach metric: views or plays from actual data source (zero fabrication).
- Centralized performance score:
  reach_score = normalized log(reach)
  engagement_score = normalized engagement using available fields
  performance_score = 0.70 * reach_score + 0.30 * engagement_score
  (falls back to reach_score if engagement is missing)
- Creator Diversity: maximum 2 verified competitors per creator.
- Reel ID deduplication.
- Documented selection_reason for every competitor.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _safe_int_reach(c: Dict[str, Any]) -> int:
    """Safely extract raw reach integer from candidate dictionary without raising on missing or None values."""
    for k in ("reach_value", "views", "plays", "view_count", "videoViewCount", "videoPlayCount"):
        val = c.get(k)
        if val is not None:
            try:
                iv = int(val)
                if iv >= 0:
                    return iv
            except (ValueError, TypeError):
                continue
    return 0


def _extract_reach(c: Dict[str, Any]) -> Tuple[int, str]:
    """Extract primary reach metric from available fields without fabrication."""
    # 1. Existing reach_value if positive
    rv = c.get("reach_value")
    if rv is not None:
        try:
            iv = int(rv)
            if iv > 0:
                metric = str(c.get("reach_metric") or "views")
                return iv, metric
        except (ValueError, TypeError):
            pass

    # 2. Views
    for k in ("views", "videoViewCount", "view_count", "viewCount"):
        val = c.get(k)
        if val is not None:
            try:
                iv = int(val)
                if iv > 0:
                    return iv, "views"
            except (ValueError, TypeError):
                pass

    # 3. Plays
    for k in ("plays", "videoPlayCount", "playCount"):
        val = c.get(k)
        if val is not None:
            try:
                iv = int(val)
                if iv > 0:
                    return iv, "plays"
            except (ValueError, TypeError):
                pass

    # 4. Fallback to likes as reach proxy only if zero views/plays
    if c.get("likes") is not None:
        try:
            iv = int(c.get("likes"))
            if iv > 0:
                return iv, "likes_proxy"
        except (ValueError, TypeError):
            pass

    return 0, "unavailable"


def _extract_engagement(c: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Extract total engagement using available fields without fabricating missing values."""
    likes = c.get("likes")
    comments = c.get("comments")
    shares = c.get("shares")
    saves = c.get("saves")

    available = []
    total = 0.0
    if likes is not None and int(likes) >= 0:
        total += float(likes)
        available.append("likes")
    if comments is not None and int(comments) >= 0:
        total += float(comments)
        available.append("comments")
    if shares is not None and int(shares) >= 0:
        total += float(shares)
        available.append("shares")
    if saves is not None and int(saves) >= 0:
        total += float(saves)
        available.append("saves")

    if not available:
        return None, "none"
    return total, "+".join(available)


def compute_performance_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute centralized performance score for each candidate.

    performance_score = 0.70 * reach_score + 0.30 * engagement_score
    If engagement is unavailable: performance_score = reach_score.
    """
    if not candidates:
        return []

    # First pass: collect log-reaches and engagements for min-max scaling
    reaches: List[float] = []
    engagements: List[Optional[float]] = []

    for c in candidates:
        reach_val, _ = _extract_reach(c)
        log_reach = math.log1p(max(0, reach_val))
        reaches.append(log_reach)

        eng_val, _ = _extract_engagement(c)
        engagements.append(eng_val)

    min_reach = min(reaches) if reaches else 0.0
    max_reach = max(reaches) if reaches else 1.0
    reach_denom = max(1e-5, max_reach - min_reach)

    valid_engs = [e for e in engagements if e is not None]
    min_eng = min(valid_engs) if valid_engs else 0.0
    max_eng = max(valid_engs) if valid_engs else 1.0
    eng_denom = max(1e-5, max_eng - min_eng)

    for idx, c in enumerate(candidates):
        reach_val, reach_metric = _extract_reach(c)
        log_reach = reaches[idx]
        norm_reach = (log_reach - min_reach) / reach_denom if max_reach > min_reach else 1.0
        c["reach_score"] = round(float(norm_reach), 4)
        c["reach_value"] = reach_val
        c["reach_metric"] = reach_metric

        eng_val = engagements[idx]
        if eng_val is not None and valid_engs and max_eng > min_eng:
            norm_eng = (eng_val - min_eng) / eng_denom
            c["engagement_score"] = round(float(norm_eng), 4)
            c["performance_score"] = round(0.70 * norm_reach + 0.30 * norm_eng, 4)
            c["performance_model"] = f"0.70*reach({reach_metric}) + 0.30*engagement"
        else:
            c["engagement_score"] = None
            c["performance_score"] = round(float(norm_reach), 4)
            c["performance_model"] = f"reach_only({reach_metric})"

    return candidates


def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prioritize and rank candidate Reels centrally using performance & relevance.

    Guarantees that highest reach candidates appear first, with performance
    score and semantic relevance preserved.
    """
    if not candidates:
        return []

    scored = compute_performance_scores(candidates)

    def sort_key(c: Dict[str, Any]) -> Tuple[int, float, float]:
        reach = _safe_int_reach(c)
        perf = float(c.get("performance_score") or 0.0)
        rel = float(c.get("video_relevance_score") or c.get("relevance_score") or 0.0)
        return (reach, perf, rel)

    ranked = sorted(scored, key=sort_key, reverse=True)
    for idx, c in enumerate(ranked, 1):
        c["rank"] = idx
        c["status"] = "RANKED"

    return ranked


def select_final_competitors(
    verified_competitors: List[Dict[str, Any]],
    max_competitors: int = 10,
    max_per_creator: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select final competitors from verified relevant pool enforcing creator diversity and Reel deduplication.

    Section 12 & 13:
    1. Deduplicate by Reel ID / shortcode.
    2. Default: maximum 2 final verified competitors per creator.
    3. If fewer than 5 unique creators satisfy threshold, allow additional videos to fill cap.
    4. Document selection_reason for each competitor.
    5. Returned competitors are strictly sorted descending by raw reach.
    """
    if not verified_competitors:
        return [], {
            "verified_relevant_reels": 0,
            "unique_creators": 0,
            "selected_count": 0,
            "diversity_cap_applied": False,
        }

    # Step 1: Deduplicate by Reel ID / shortcode
    seen_ids = set()
    deduped: List[Dict[str, Any]] = []
    for c in verified_competitors:
        rid = str(c.get("shortcode") or c.get("id") or "").strip()
        if not rid or rid not in seen_ids:
            if rid:
                seen_ids.add(rid)
            deduped.append(c)

    # Count unique creators in the verified candidate pool
    creator_counts_all: Dict[str, int] = {}
    for c in deduped:
        creator = str(c.get("creator") or c.get("username") or "unknown").lower()
        creator_counts_all[creator] = creator_counts_all.get(creator, 0) + 1

    total_unique_creators = len(creator_counts_all)

    # Ensure performance scores and reach values are populated
    if any("performance_score" not in c for c in deduped):
        deduped = compute_performance_scores(deduped)

    # Sort candidates by raw reach descending, then performance_score and relevance
    sorted_candidates = sorted(
        deduped,
        key=lambda c: (
            _safe_int_reach(c),
            float(c.get("performance_score") or 0.0),
            float(c.get("video_relevance_score") or 0.0),
        ),
        reverse=True
    )

    selected: List[Dict[str, Any]] = []
    creator_allocations: Dict[str, int] = {}
    deferred_overflow: List[Dict[str, Any]] = []

    # First pass: enforce max_per_creator
    for c in sorted_candidates:
        creator = str(c.get("creator") or c.get("username") or "unknown").lower()
        current_count = creator_allocations.get(creator, 0)

        if current_count < max_per_creator:
            creator_allocations[creator] = current_count + 1
            c["selection_reason"] = (
                f"High-reach verified competitor (reach: {c.get('reach_value', 0)} [{c.get('reach_metric')}]), "
                f"performance score: {c.get('performance_score')}, creator: @{creator} (allocation {current_count + 1}/{max_per_creator})"
            )
            selected.append(c)
            if len(selected) >= max_competitors:
                break
        else:
            deferred_overflow.append(c)

    # Second pass: If fewer than 5 unique creators exist and we haven't reached max_competitors, allow overflow
    if len(selected) < max_competitors and (total_unique_creators < 5 or not selected):
        for c in deferred_overflow:
            if len(selected) >= max_competitors:
                break
            creator = str(c.get("creator") or c.get("username") or "unknown").lower()
            creator_allocations[creator] = creator_allocations.get(creator, 0) + 1
            c["selection_reason"] = (
                f"Diversity relaxation (fewer than 5 unique creators available): creator @{creator} "
                f"selected for verified depth with reach {c.get('reach_value', 0)}"
            )
            selected.append(c)

    final_unique_creators = len(set(
        str(c.get("creator") or c.get("username") or "unknown").lower() for c in selected
    ))

    diagnostics = {
        "verified_relevant_reels": len(deduped),
        "unique_creators": final_unique_creators,
        "total_unique_creators_available": total_unique_creators,
        "selected_count": len(selected),
        "diversity_cap_applied": bool(total_unique_creators >= 5),
    }

    # Final sort: Guarantee selected competitors are strictly descending by raw reach metric
    selected.sort(
        key=lambda c: (
            _safe_int_reach(c),
            float(c.get("performance_score") or 0.0),
            float(c.get("video_relevance_score") or 0.0),
        ),
        reverse=True,
    )

    return selected, diagnostics


def _infer_competitor_cluster_key(cand: Dict[str, Any]) -> str:
    """Classify competitor into a standardized edit-type cluster."""
    comp_analysis = cand.get("analysis") or {}
    if isinstance(comp_analysis.get("analysis"), dict):
        comp_data = comp_analysis["analysis"]
    else:
        comp_data = comp_analysis

    explicit = str(
        cand.get("edit_type")
        or comp_data.get("edit_type")
        or comp_analysis.get("edit_type")
        or ""
    ).lower().strip()
    if explicit:
        if any(w in explicit for w in ["action", "montage", "fight", "battle"]):
            return "action montage"
        if any(w in explicit for w in ["emotion", "sad", "tribute"]):
            return "emotional edit"
        if any(w in explicit for w in ["dialogue", "quote", "speech"]):
            return "dialogue edit"
        if any(w in explicit for w in ["vfx", "cgi", "effects"]):
            return "vfx edit"
        if any(w in explicit for w in ["meme", "funny", "comedy", "parody"]):
            return "meme/comedy edit"

    # Infer from metadata, caption, topic, format, summary, editing_style, sub_niche, discovery_keywords
    combined_parts = [str(cand.get(k) or "") for k in ("caption", "format", "topic", "summary")]
    for k in ("format", "topic", "summary", "sub_niche", "editing_style", "visual_style"):
        combined_parts.append(str(comp_data.get(k) or ""))
        combined_parts.append(str(comp_analysis.get(k) or ""))
    hook = comp_data.get("hook") or {}
    if isinstance(hook, dict):
        combined_parts.append(str(hook.get("type") or ""))
        combined_parts.append(str(hook.get("summary") or ""))
    dk = comp_data.get("discovery_keywords") or []
    if isinstance(dk, list):
        combined_parts.extend(str(x) for x in dk)

    combined = " ".join(combined_parts).lower()

    if any(w in combined for w in ["meme", "funny", "comedy", "parody", "humor"]):
        return "meme/comedy edit"
    if any(w in combined for w in ["emotional", "sad", "tribute", "nostalgia", "feel", "pain"]):
        return "emotional edit"
    if any(w in combined for w in ["dialogue", "quote", "speech", "monologue", "conversation"]):
        return "dialogue edit"
    if any(w in combined for w in ["vfx", "cgi", "after effects", "visual effects"]):
        return "vfx edit"
    if any(w in combined for w in ["action", "montage", "fight", "battle", "combat", "cinematic edit", "action sequence", "beat sync", "scene"]):
        return "action montage"

    return "other"  # Unclassified edit cluster


def cluster_competitors(
    verified_competitors: List[Dict[str, Any]],
    user_edit_type: str = "mixed",
) -> Dict[str, Any]:
    """Group verified competitors into clusters by edit type.

    Designates the cluster matching user_edit_type as PRIMARY_CLUSTER.
    Secondary clusters are preserved for broader market context.
    All clusters are sorted internally by raw reach descending.
    """
    user_key = str(user_edit_type or "mixed").lower().strip()
    if any(w in user_key for w in ["action", "montage", "fight"]):
        normalized_primary_key = "action montage"
    elif any(w in user_key for w in ["emotion", "sad", "tribute"]):
        normalized_primary_key = "emotional edit"
    elif any(w in user_key for w in ["dialogue", "quote", "speech"]):
        normalized_primary_key = "dialogue edit"
    elif any(w in user_key for w in ["vfx", "cgi"]):
        normalized_primary_key = "vfx edit"
    elif any(w in user_key for w in ["meme", "funny", "comedy"]):
        normalized_primary_key = "meme/comedy edit"
    else:
        normalized_primary_key = user_key or "other"

    canonical_names = {
        "action montage": "Action Montage",
        "emotional edit": "Emotional Edit",
        "dialogue edit": "Dialogue Edit",
        "vfx edit": "VFX Edit",
        "meme/comedy edit": "Meme / Comedy Edit",
        "other": "Other Edit Style",
    }

    clusters_map: Dict[str, List[Dict[str, Any]]] = {}

    for cand in verified_competitors:
        c_key = _infer_competitor_cluster_key(cand)
        cand["competitor_cluster"] = c_key
        cand["is_primary_cluster"] = (c_key == normalized_primary_key)
        clusters_map.setdefault(c_key, []).append(cand)

    def _cluster_summary(key: str, cands: List[Dict[str, Any]], is_primary: bool) -> Dict[str, Any]:
        sorted_cands = sorted(
            cands,
            key=lambda c: (
                _safe_int_reach(c),
                float(c.get("performance_score") or 0.0),
                float(c.get("video_relevance_score") or 0.0),
            ),
            reverse=True,
        )
        creators = list(dict.fromkeys(
            str(c.get("creator") or c.get("username") or "unknown").lower() for c in sorted_cands
        ))
        reaches = [_safe_int_reach(c) for c in sorted_cands]
        avg_reach = float(sum(reaches) / len(reaches)) if reaches else 0.0
        return {
            "cluster_key": key,
            "cluster_name": canonical_names.get(key, key.title()),
            "is_primary": is_primary,
            "count": len(sorted_cands),
            "unique_creators": len(creators),
            "creators": creators,
            "avg_reach": round(avg_reach, 1),
            "competitors": sorted_cands,
        }

    primary_cands = clusters_map.get(normalized_primary_key, [])
    primary_cluster = _cluster_summary(normalized_primary_key, primary_cands, is_primary=True)

    secondary_clusters = []
    for k, cands in clusters_map.items():
        if k != normalized_primary_key:
            secondary_clusters.append(_cluster_summary(k, cands, is_primary=False))

    secondary_clusters.sort(key=lambda c: c["count"], reverse=True)

    return {
        "primary_cluster": primary_cluster,
        "secondary_clusters": secondary_clusters,
        "PRIMARY_CLUSTER": primary_cluster,
        "SECONDARY_CLUSTERS": secondary_clusters,
        "primary_cluster_key": normalized_primary_key,
        "primary_cluster_name": primary_cluster["cluster_name"],
        "primary_count": primary_cluster["count"],
        "secondary_count": sum(c["count"] for c in secondary_clusters),
        "total_verified": len(verified_competitors),
    }
