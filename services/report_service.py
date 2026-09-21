"""Report Service - generates comprehensive JSON and Markdown audit reports with scientific precision.

Enforces Sections 17 & 18:
- Standardized display precision: scores to 2 decimals (0.54), times to 2 decimals (26.50s), exact counts.
- Preserves provenance architecture: OBSERVED, MODEL_DERIVED, SEMANTICALLY_VERIFIED, AGGREGATED_PATTERN, RECOMMENDATION.
- Includes creator diversity, reach-attention associations, and generalization warnings.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from config.settings import REPORTS_DIR
from services.metrics_service import format_percentage, format_number


def _fmt_score(val: Any) -> str:
    """Format floating point attention/relevance score to 2 decimals without trailing junk."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_time(val: Any) -> str:
    """Format timestamp to 2 decimals (e.g. 26.50s)."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.2f}s"
    except (ValueError, TypeError):
        return str(val)


def _extract_sort_reach(c: dict[str, Any]) -> tuple[int, float]:
    """Safely extract integer reach value and performance score for descending sorting."""
    reach = 0
    for k in ("reach_value", "views", "plays", "view_count", "videoViewCount", "videoPlayCount"):
        v = c.get(k)
        if v is not None:
            try:
                iv = int(v)
                if iv >= 0:
                    reach = iv
                    break
            except (ValueError, TypeError):
                pass
    perf = float(c.get("performance_score") or 0.0)
    return (reach, perf)


def sort_timeline_events(timeline_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort timeline segments by numeric start_time ascending.

    Preserves each segment's start_time, end_time, score, event metadata, and all existing fields.
    Never sorts by formatted display strings.
    """
    if not timeline_events:
        return []

    def _get_numeric_start(item: dict[str, Any]) -> tuple[float, float]:
        val = item.get("start_time")
        if val is None:
            val = item.get("start")
        try:
            start_num = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            start_num = 0.0

        end_val = item.get("end_time")
        if end_val is None:
            end_val = item.get("end")
        try:
            end_num = float(end_val) if end_val is not None else start_num
        except (ValueError, TypeError):
            end_num = start_num

        return (start_num, end_num)

    return sorted(timeline_events, key=_get_numeric_start)


def save_report(payload: dict[str, Any], base_name: str | None = None) -> tuple[Path, Path]:
    """Generate and persist both JSON and Markdown reports with strictly descending reach ordering."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"virallens_report_{base_name or 'audit'}_{timestamp}"

    json_path = REPORTS_DIR / f"{stem}.json"
    md_path = REPORTS_DIR / f"{stem}.md"

    # Canonical sorting: ensure verified_competitors and selected_candidates are strictly descending by reach
    if "verified_competitors" in payload and isinstance(payload["verified_competitors"], list):
        payload["verified_competitors"].sort(key=_extract_sort_reach, reverse=True)
    if "selected_candidates" in payload and isinstance(payload["selected_candidates"], list):
        payload["selected_candidates"].sort(key=_extract_sort_reach, reverse=True)

    # Chronological timeline sorting: ensure human_attention timeline_events are strictly chronological
    agent_results = payload.get("agent_results", {})
    if isinstance(agent_results, dict):
        human_att = agent_results.get("human_attention", {})
        if isinstance(human_att, dict) and "timeline_events" in human_att:
            human_att["timeline_events"] = sort_timeline_events(human_att["timeline_events"])

    # Write JSON report with machine-readable precision
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Build Markdown report with appropriate human display precision
    md_content = _build_markdown_report(payload)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def _build_markdown_report(data: dict[str, Any]) -> str:
    reel = data.get("your_reel", {})
    queries = data.get("queries", [])
    diag = data.get("discovery_diagnostics", {})
    verified = data.get("verified_competitors", [])
    if isinstance(verified, list):
        verified = sorted(verified, key=_extract_sort_reach, reverse=True)
    rejected = data.get("rejected_competitors", [])
    patterns = data.get("patterns", {})
    agent_results = data.get("agent_results", {})
    att_comp = data.get("attention_comparison", {})
    human_att = agent_results.get("human_attention", {})

    first_event_display = human_att.get("first_attention_event_display")
    if not first_event_display:
        fet = human_att.get("first_attention_event_time")
        first_event_display = f"{_fmt_time(fet)} (Score: {_fmt_score(human_att.get('first_attention_event_score'))})" if fet is not None else "No major attention event detected in the analyzed window"

    target_video = data.get("target_video") or reel.get("target_video") or reel.get("video_path") or ""

    lines = [
        "# ViralLens AI - Reel Intelligence & Competitor Audit Report",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if target_video:
        lines.append(f"Target Video: `{target_video}`")
    lines.extend([
        "",
        "> **Scientific Disclaimer**: Laboratory-Derived Attention Model (NEMAR BBBD)",
        "> This signal represents predicted visual-attention potential based on eye-tracking gaze fixations from laboratory stimuli.",
        "> It does NOT measure commercial Instagram scroll-stop behavior or viewer retention.",
        "",
    ])

    if human_att.get("generalization_warning"):
        lines.extend([
            "> [!WARNING]",
            "> **Model Generalization Warning**: Model-derived attention variation is low for this video.",
            "> Interpret temporal differences cautiously because the model was trained on a limited laboratory stimulus set.",
            "",
        ])

    lines.extend([
        "## 1. Uploaded Reel Profile",
    ])
    if target_video:
        lines.append(f"- **Target Video**: `{target_video}`")
    lines.extend([
        f"- **Niche**: {reel.get('niche', 'N/A')}",
        f"- **Topic**: {reel.get('topic', 'N/A')}",
        f"- **Primary Entity**: {reel.get('primary_entity') or reel.get('primary_subject', 'N/A')}",
        f"- **Format**: {reel.get('format', 'N/A')}",
        f"- **Tone**: {reel.get('tone', 'N/A')}",
        f"- **Hook Type**: {(reel.get('hook') or {}).get('type', 'N/A')}",
        f"- **Hook Strength**: {format_percentage((reel.get('hook') or {}).get('strength'))}",
        f"- **Hook Summary**: {(reel.get('hook') or {}).get('first_3_seconds', 'N/A')}",
        "",
        "## 2. Predicted Visual-Attention Potential (NEMAR BBBD Benchmark)",
        f"- **Model**: Random Forest Regressor (100 estimators, 14 multimodal features)",
        f"- **Baseline Attention (Early Video)**: {_fmt_score(human_att.get('baseline_attention'))}",
        f"- **First Major Attention Event**: {first_event_display}",
        f"- **Peak Attention Potential**: {_fmt_score(human_att.get('peak_attention'))} at {_fmt_time(human_att.get('peak_attention_time'))}",
        f"- **Lowest Attention Potential**: {_fmt_score(human_att.get('lowest_attention'))} at {_fmt_time(human_att.get('lowest_attention_time'))}",
        f"- **Temporal Curve Stability**: {_fmt_score(human_att.get('attention_stability'))}",
        f"- **Temporal Trajectory**: {human_att.get('trajectory', 'N/A')}",
        "",
        "### Major Chronological Attention Timeline Events",
    ])

    timeline_events = sort_timeline_events(human_att.get("timeline_events", []))
    for ev in timeline_events[:8]:
        lines.append(
            f"- **{ev.get('timestamp')}**: Potential = {_fmt_score(ev.get('attention_score'))} | Event: {ev.get('major_observed_event')}"
        )

    # Section 2b: Experimental DHF1K Spatial Saliency (Shadow Mode)
    shadow_signal = data.get("dhf1k_spatial_saliency_signal") or reel.get("dhf1k_spatial_saliency_signal") or {}
    lines.extend([
        "",
        "## 2b. Experimental DHF1K Spatial Saliency (Shadow-Mode Research Signal)",
        "> **Notice**: This section reports experimental research output from a prototype 2D convolutional model (TinySalNet).",
        "> It executes strictly in shadow mode for observational telemetry and does NOT affect competitor selection, ranking, recommendations, or production attention scoring.",
        "",
    ])

    if shadow_signal.get("status") == "SUCCESS":
        prov = shadow_signal.get("model_provenance", {})
        samp = shadow_signal.get("sampling_metadata", {})
        summ = shadow_signal.get("summary", {})
        telem = shadow_signal.get("telemetry", {})
        lines.extend([
            f"- **Model**: {prov.get('architecture', 'TinySalNet')}",
            f"- **Model Provenance**: {prov.get('training_supervision', 'DHF1K saliency supervision')} (`{prov.get('model_hash', 'N/A')[:16]}...`)",
            f"- **Sampled Frames**: {samp.get('sampled_frame_count', 0)} frames ({samp.get('sampling_rule', '1 fps')})",
            f"- **Mean Saliency Spatial Dispersion**: {summ.get('mean_spatial_dispersion', 'N/A')}",
            f"- **Mean Spatial Entropy**: {summ.get('mean_spatial_entropy_bits', 'N/A')} bits",
            f"- **Mean Inter-Frame Saliency Shift**: {summ.get('mean_inter_frame_shift', 'N/A')}",
            f"- **Inference Latency**: {telem.get('shadow_ms_per_frame', 'N/A')} ms/frame (Total wall time: {telem.get('shadow_total_wall_time', 'N/A')}s)",
            f"- **Execution Device**: {prov.get('execution_device', 'CPU')}",
        ])
    elif shadow_signal.get("status") == "FAILED":
        lines.extend([
            "- **Status**: FAILED (Structured Error Preserved)",
            f"- **Error Details**: `{shadow_signal.get('error', 'Unknown error')}`",
            "- **Fallback Policy**: Zero fabrication policy enforced; production pipeline unaffected.",
        ])
    else:
        lines.extend([
            "- **Status**: NOT_TRIGGERED / PENDING",
            "- **Notice**: Shadow spatial service was not active for this run.",
        ])

    # Section 2c: Attention Intelligence (Canonical Integration)
    att_intel = data.get("attention_intelligence") or reel.get("attention_intelligence")
    if att_intel and isinstance(att_intel, dict) and att_intel.get("status") in {"SUCCESS", "PARTIAL"}:
        t_obj = att_intel.get("temporal", {})
        s_obj = att_intel.get("spatial", {})
        rel_obj = att_intel.get("relationships", {})
        interps = att_intel.get("interpretation", [])
        focal_comp = s_obj.get("focal_competition", {})
        t_pers = t_obj.get("temporal_persistence", {})
        drops = t_obj.get("attention_drops", [])
        recs = t_obj.get("attention_recoveries", [])
        changes_desc = f"{len(drops)} drops, {len(recs)} recoveries detected" if (drops or recs) else "Stable trajectory without sharp drop events"

        lines.extend([
            "",
            "## 2c. Attention Intelligence",
            "> **Canonical Attention Intelligence**: Unified integration of frozen NEMAR temporal attention and TinySalNet spatial visual saliency.",
            "",
            "### Temporal Attention",
            f"- **Opening Signal**: {_fmt_score(t_obj.get('opening_signal'))}",
            f"- **Peak Attention**: {_fmt_score(t_obj.get('peak'))}",
            f"- **Peak Timestamp**: {_fmt_time(t_obj.get('peak_timestamp'))} (relative position: {t_obj.get('relative_peak_position', 0):.1%})",
            f"- **Attention Persistence**: {t_pers.get('duration_above_baseline_sec', 0)}s above median ({t_pers.get('persistence_ratio', 0):.1%} of duration)",
            f"- **Attention Modulation**: {changes_desc}",
            "",
            "### Spatial Saliency",
            f"- **Focal Centroid**: (x={s_obj.get('centroid', {}).get('x', 'N/A')}, y={s_obj.get('centroid', {}).get('y', 'N/A')})",
            f"- **Peak Focus**: (x={s_obj.get('peak_focus', {}).get('x', 'N/A')}, y={s_obj.get('peak_focus', {}).get('y', 'N/A')})",
            f"- **Spatial Concentration**: {_fmt_score(s_obj.get('focal_concentration'))}",
            f"- **Spatial Dispersion**: {s_obj.get('dispersion', 'N/A')}",
            f"- **Spatial Entropy**: {s_obj.get('entropy', 'N/A')} bits",
            f"- **Temporal Centroid Drift**: {s_obj.get('temporal_centroid_drift', 'N/A')}",
            "",
            "### Attention Dynamics",
            f"- **Focus Movement**: {s_obj.get('descriptors', [''])[0] if s_obj.get('descriptors') else 'N/A'}",
            f"- **Temporal Peaks & Shifts**: {t_obj.get('descriptors', [''])[0] if t_obj.get('descriptors') else 'N/A'}",
            f"- **Scene-Transition Alignment**: {rel_obj.get('transition_attention_alignment', {}).get('description', 'N/A')}",
            "",
            "### Focal Competition",
            f"- **Visual Focus State**: {focal_comp.get('competition_state', 'N/A').replace('_', ' ').title()}",
            f"- **Dominant Region Mass**: {focal_comp.get('dominant_region_mass', 'N/A')}",
            f"- **Secondary Region Mass**: {focal_comp.get('secondary_region_mass', 'N/A')}",
            f"- **Concentration Ratio**: {focal_comp.get('concentration_ratio', 'N/A')}",
            f"- **Focal Distribution Rationale**: {focal_comp.get('rationale', 'N/A')}",
            "",
            "### Text / Face Interaction",
            f"- **OCR Relationship**: {rel_obj.get('text_saliency_interaction', {}).get('description', 'No text interaction')}",
            f"- **Face Relationship**: {rel_obj.get('face_saliency_interaction', {}).get('description', 'No face interaction')}",
            "",
            "### Creative Interpretation",
        ])
        for interp in interps:
            lines.append(f"- {interp}")

        lines.extend([
            "",
            "### Experimental Boundaries",
            "> **Important Disclosure**: This section provides model-derived visual-attention and spatial-saliency signals. It does not measure actual Instagram viewer behavior, retention, or scroll-stop events.",
            "",
        ])

    lines.extend([
        "",
        "## 3. Multi-Agent System Findings & Evidence Provenance",
    ])

    for agent_key, a_res in agent_results.items():
        if isinstance(a_res, dict) and "findings" in a_res:
            lines.append(f"### Agent: {agent_key.title()}")
            lines.append(f"- **Confidence**: {_fmt_score(a_res.get('confidence'))}")
            lines.append("- **Findings**:")
            for f in a_res.get("findings", []):
                lines.append(f"  - {f}")
            if a_res.get("evidence"):
                lines.append("- **Empirical Evidence**:")
                for e in a_res.get("evidence", [])[:4]:
                    lines.append(f"  - {e}")
            lines.append("")

    lines.extend([
        "## 4. Generated Search Queries",
    ])

    for idx, q in enumerate(queries, 1):
        if isinstance(q, dict):
            lines.append(f"{idx}. `{q.get('query')}` (Level: {q.get('search_level')})")
        else:
            lines.append(f"{idx}. `{q}`")

    unique_creators = diag.get("unique_creators", len(set(
        str(c.get("creator") or c.get("username") or "unknown").lower() for c in verified
    )))

    disc_health_val = str(diag.get("discovery_health") or "unknown").lower()
    query_results = diag.get("query_results") or []
    successful_q = diag.get("successful_queries") or [q.get("query") for q in query_results if q.get("status") == "success"]
    blocked_q = diag.get("blocked_queries") or [q.get("query") for q in query_results if q.get("status") == "blocked"]
    failed_q = diag.get("failed_queries") or [q.get("query") for q in query_results if q.get("status") in {"error", "timeout"}]

    lines.extend([
        "",
        "## 5. Instagram Discovery & Pipeline Diagnostics",
        f"- **Discovery Health**: `{disc_health_val.upper()}`",
        f"- **Candidates Discovered**: {diag.get('candidates_returned', diag.get('raw_candidates', 0))}",
        f"- **Queries Attempted**: {len(query_results) if query_results else len(queries)}",
        f"- **Queries Succeeded**: {len(successful_q)}",
        f"- **Queries Blocked**: {len(blocked_q)}",
        f"- **Queries Failed**: {len(failed_q)}",
        f"- **Raw Candidates Found**: {diag.get('raw_candidates', 0)}",
        f"- **Normalized Candidates**: {diag.get('normalized_candidates', 0)}",
        f"- **Ranked Candidates**: {diag.get('ranked_candidates', 0)}",
        f"- **Downloaded Competitor MP4s**: {diag.get('downloaded_competitors', 0)}",
        f"- **Twelve Labs Analyses**: {diag.get('twelvelabs_analyzed_competitors', 0)}",
        f"- **Verified Relevant Reels**: {len(verified)}",
        f"- **Unique Creators**: {unique_creators}",
        f"- **Rejected Competitors (Unrelated)**: {diag.get('rejected_competitors', 0)}",
    ])

    if disc_health_val == "failed" or diag.get("candidates_returned", 0) == 0:
        lines.extend([
            "",
            "> [!CAUTION]",
            "> **Discovery Notice**: Instagram discovery did not return usable candidates during this verification run.",
            "> Resilience handling passed, but full competitor-analysis verification could not be completed because no candidates were discovered.",
        ])
    elif blocked_q:
        lines.extend([
            "",
            "> [!WARNING]",
            "> **Partial Discovery Warning**: Some Instagram discovery queries were blocked by Instagram. Results below are based on successfully retrieved queries.",
        ])

    if query_results:
        lines.extend([
            "",
            "### Discovery Query Execution Audit",
            "| Query | Status | Results | Retries | Error |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ])
        for qr in query_results:
            q_txt = qr.get("query", "")
            q_stat = str(qr.get("status", "unknown")).upper()
            q_res = qr.get("results_count", 0)
            q_ret = qr.get("retry_count", 0)
            q_err = str(qr.get("error") or "None").replace("|", "/")
            lines.append(f"| `{q_txt}` | `{q_stat}` | {q_res} | {q_ret} | {q_err} |")

    lines.extend([
        "",
        f"## 6. Verified Competitors ({len(verified)} Reels, {unique_creators} Creators)",
    ])

    if not verified:
        lines.extend([
            "",
            "_No verified competitors discovered or analyzed in this run. Downstream competitor benchmarking was bypassed._",
            "",
        ])
    else:
        for comp in verified:
            cid = comp.get("shortcode") or comp.get("id", "unknown")
            url = comp.get("url") or comp.get("source_url", "")
            creator = comp.get("creator") or comp.get("username", "unknown")
            views = format_number(comp.get("views") or comp.get("reach_value"))
            likes = format_number(comp.get("likes"))
            perf = _fmt_score(comp.get("performance_score"))
            lines.append(f"### Reel: [{cid}]({url}) by @{creator}")
            lines.append(f"- **Reach ({comp.get('reach_metric', 'views')})**: {views} | **Likes**: {likes} | **Performance Score**: {perf}")
            lines.append(f"- **Relevance Score**: {_fmt_score(comp.get('video_relevance_score'))} ({comp.get('evidence_label', 'Semantic verification')})")
            lines.append(f"- **Verification Reason**: {comp.get('video_relevance_reason', 'Verified relevant')}")
            if comp.get("selection_reason"):
                lines.append(f"- **Selection Reason**: {comp.get('selection_reason')}")
            lines.append("")

    # Section 7: Competitor Attention Benchmark & Reach Association
    if att_comp and att_comp.get("status") == "SUCCESS" and len(verified) > 0:
        bench = att_comp.get("benchmark", {})
        reach_assoc = att_comp.get("reach_attention_association", {})
        lines.extend([
            "## 7. Competitor Attention Benchmark & Reach Association",
            f"- **Verified Competitors Benchmarked**: {bench.get('competitor_count', 0)}",
            f"- **Average Competitor Opening Strength**: {_fmt_score(bench.get('avg_opening_strength'))}",
            f"- **Average Competitor Peak Score**: {_fmt_score(bench.get('avg_peak_score'))}",
            f"- **Average First Major Attention Event**: {_fmt_time(bench.get('avg_first_event_time'))}",
            "- **Benchmark Insights**:",
        ])
        for ins in att_comp.get("insights", []):
            lines.append(f"  - {ins}")
        lines.append("")

        if reach_assoc:
            lines.extend([
                "### Reach–Attention Association Analysis",
                f"- **Summary**: {reach_assoc.get('summary', 'N/A')}",
            ])
            for ins in reach_assoc.get("insights", []):
                lines.append(f"- **Observation**: {ins}")
            lines.append(f"- **Scientific Limitations**: {reach_assoc.get('limitations', 'Observational study only.')}")
            lines.append("")

    # Phase 15: Competitor Intelligence Engine Section
    comp_intel = data.get("competitor_intelligence") or (agent_results.get("pattern") or {}).get("competitor_intelligence") or reel.get("competitor_intelligence")
    if comp_intel and comp_intel.get("status") == "SUCCESS":
        target_norm = comp_intel.get("target_normalized", {})
        t_sem = target_norm.get("semantic", {})
        sg_counts = comp_intel.get("subgroup_counts", {})
        recurring = comp_intel.get("recurring_patterns", [])
        gaps = comp_intel.get("gap_analysis", [])
        reach_obs = comp_intel.get("reach_aware_observations", {})
        creatives = comp_intel.get("creative_implications", [])
        all_pats = comp_intel.get("all_subgroup_patterns", [])
        att_patterns = [p for p in all_pats if p.get("pattern_id", "").startswith("att_")]

        lines.extend([
            "",
            "# Competitor Intelligence",
            "> **Rigorous Evidence-Backed Competitor Intelligence Engine**: Compares target Reel against verified, Twelve-Labs-analyzed competitor group.",
            "> Strictly non-causal descriptive findings; does not measure commercial retention or scroll-stop causality.",
            "",
            "## Verified Comparison Group",
            f"- **Total Verified Competitors**: {comp_intel.get('verified_competitor_count', len(verified))}",
            f"- **Unique Creators**: {comp_intel.get('unique_creators', unique_creators)}",
            f"- **Target Entity**: {(t_sem.get('entity') or {}).get('value') or reel.get('primary_entity', 'N/A')}",
            f"- **Target Niche**: {(t_sem.get('niche') or {}).get('value') or reel.get('niche', 'N/A')}",
            f"- **Target Edit Type**: {(t_sem.get('edit_type') or {}).get('value') or 'action montage'}",
            "",
            "## Comparison Groups",
            f"- **same_entity_and_edit_type**: {sg_counts.get('same_entity_and_edit_type', 0)} reels",
            f"- **same_edit_type**: {sg_counts.get('same_edit_type', 0)} reels",
            f"- **all_verified**: {sg_counts.get('all_verified', 0)} reels",
            f"- **highest_reach_same_edit_type**: {sg_counts.get('highest_reach_same_edit_type', 0)} reels",
            "",
            "## Recurring Patterns",
        ])

        if recurring:
            for pat in recurring[:8]:
                lines.extend([
                    f"### Pattern: {pat.get('pattern_name')}",
                    f"- **Subgroup**: `{pat.get('subgroup')}`",
                    f"- **Denominator Accounting**: **{pat.get('numerator')}/{pat.get('denominator')}** ({float(pat.get('prevalence', 0))*100:.1f}%)",
                    f"- **Target Value**: `{pat.get('target_value')}` | **Competitor Value**: `{pat.get('competitor_value')}`",
                    f"- **Evidence**: {'; '.join(pat.get('evidence', []))}",
                    f"- **Strength**: `{pat.get('strength')}`",
                    f"- **Limitation**: {pat.get('limitation')}",
                    f"- **Provenance**: Subgroup n={pat.get('denominator')}, Target `{target_norm.get('reel_id')}`",
                    "",
                ])
        else:
            lines.append("- _No dominant recurring categorical patterns above threshold in primary subgroup._\n")

        lines.extend([
            "## Target vs Competitors",
        ])
        if gaps:
            for g in gaps[:6]:
                lines.extend([
                    f"- **{g.get('pattern_name')}** (`{g.get('gap_status')}`): {g.get('interpretation')}",
                    f"  - _Evidence_: {g.get('competitor_aggregate')} across subgroup `{g.get('subgroup')}`",
                    f"  - _Limitation_: {g.get('limitation')}",
                ])
            lines.append("")
        else:
            lines.append("- _Target aligns with observed competitor baselines across primary features._\n")

        lines.extend([
            "## Attention Patterns",
        ])
        if att_patterns:
            for ap in att_patterns[:5]:
                lines.extend([
                    f"- **{ap.get('pattern_name')}** (`{ap.get('subgroup')}`): {ap.get('numerator')}/{ap.get('denominator')} competitors ({float(ap.get('prevalence', 0))*100:.1f}%)",
                    f"  - Target status: `{ap.get('target_value')}` | Evidence: {'; '.join(ap.get('evidence', []))}",
                ])
            lines.append("")
        else:
            lines.append("- _No attention-specific patterns extracted._\n")

        lines.extend([
            "## Reach-Aware Observations",
        ])
        if reach_obs and reach_obs.get("observations"):
            for ro in reach_obs.get("observations", []):
                lines.append(f"- {ro}")
            if reach_obs.get("limitations"):
                lines.append(f"- > **Limitation**: {reach_obs.get('limitations')}")
            lines.append("")
        else:
            lines.append("- _Reach data missing or insufficient for stratified observations._\n")

        lines.extend([
            "## Evidence-Backed Creative Implications",
        ])
        if creatives:
            for c_idx, ci in enumerate(creatives, 1):
                lines.extend([
                    f"### Implication {c_idx}: {ci.get('pattern_name')}",
                    f"- **Observation**: {ci.get('observation')}",
                    f"- **Evidence**: {ci.get('evidence')}",
                    f"- **Target/Competitor Comparison**: {ci.get('target_competitor_comparison')}",
                    f"- **Recommendation**: {ci.get('recommendation')}",
                    f"- **Limitation**: {ci.get('limitation')}",
                    "",
                ])
        else:
            lines.append("- _No divergent creative gaps identified; target aligns with verified subgroup patterns._\n")

    # Phase 16: Recommendation Intelligence Section
    rec_intel = (
        data.get("recommendation_intelligence")
        or agent_results.get("recommendation_intelligence")
        or (agent_results.get("strategy") or {}).get("recommendation_intelligence")
        or reel.get("recommendation_intelligence")
    )
    if rec_intel and str(rec_intel.get("status", "")).startswith(("SUCCESS", "NO_ACTIONABLE")):
        recs = rec_intel.get("recommendations", [])
        cats = rec_intel.get("categories", {})
        raw_graph = rec_intel.get("evidence_graph", [])
        graph = raw_graph.get("nodes", []) if isinstance(raw_graph, dict) else (raw_graph if isinstance(raw_graph, list) else [])

        lines.extend([
            "",
            "# Recommendation Intelligence",
            "> **Rigorous Evidence-Backed Decision Support**: Formulates testable creative hypotheses based on verified multimodal peer evidence.",
            "> Observational findings only; does not predict commercial retention, virality, or scroll-stop causality.",
            "",
            "## Priority Recommendations",
        ])

        if recs:
            for idx, r in enumerate(recs, 1):
                ev_list = r.get("evidence", [])
                ev_str = "; ".join(e.get("value", "") for e in ev_list) if ev_list else r.get("observation", "")
                diff = (r.get("target_comparison") or {}).get("difference", "N/A")
                lines.extend([
                    f"### Recommendation {idx}: {r.get('recommendation')}",
                    f"- **Category**: `{r.get('category')}` | **Priority**: `{r.get('priority')}` | **Strength**: `{r.get('strength')}`",
                    f"- **Observation**: {r.get('observation')}",
                    f"- **Evidence**: {ev_str}",
                    f"- **Target vs Competitor Comparison**: {diff}",
                    f"- **Recommendation**: {r.get('recommendation')}",
                    f"- **Testable Action**: {r.get('testable_action')}",
                    f"- **Limitation**: {r.get('limitation')}",
                    "",
                ])
        else:
            lines.append("- _No actionable recommendations generated._\n")

        lines.extend([
            "## Preserve",
        ])
        preserve_recs = cats.get("PRESERVE", [])
        if preserve_recs:
            for pr in preserve_recs:
                lines.append(f"- **{pr.get('recommendation')}**: {pr.get('observation')} (Strength: `{pr.get('strength')}`)")
            lines.append("")
        else:
            lines.append("- _No specific elements marked for preservation._\n")

        lines.extend([
            "## Change",
        ])
        change_recs = cats.get("CHANGE", [])
        if change_recs:
            for cr in change_recs:
                lines.append(f"- **{cr.get('recommendation')}**: {cr.get('observation')}")
                lines.append(f"  - _Testable Action_: {cr.get('testable_action')}")
            lines.append("")
        else:
            lines.append("- _No high-confidence divergent creative changes required._\n")

        lines.extend([
            "## Test",
        ])
        test_recs = cats.get("TEST", [])
        if test_recs:
            for tr in test_recs:
                lines.append(f"- **{tr.get('recommendation')}**: {tr.get('observation')}")
                lines.append(f"  - _Creative Hypothesis_: {tr.get('testable_action')}")
            lines.append("")
        else:
            lines.append("- _No exploratory tests proposed._\n")

        lines.extend([
            "## No Action / Insufficient Evidence",
        ])
        no_action_recs = cats.get("NO_ACTIONABLE_RECOMMENDATION", []) + cats.get("AVOID_OVERINTERPRETATION", [])
        if no_action_recs:
            for nar in no_action_recs:
                lines.append(f"- **{nar.get('recommendation_id')}**: {nar.get('observation')} ({nar.get('limitation')})")
            lines.append("")
        else:
            lines.append("- _All evaluated creative dimensions provided sufficient evidence for classification._\n")

        lines.extend([
            "## Evidence Trace",
        ])
        if graph:
            for node in graph[:6]:
                tr = node.get("trace", {})
                c_ids = ", ".join(tr.get("competitor_ids", [])[:4]) or "none"
                lines.append(f"- **`{node.get('recommendation_id')}`** (`{node.get('category')}`): Traced to target `{tr.get('target')}` across subgroup `{tr.get('comparison_group')}` (Peers: `{c_ids}`)")
            lines.append("")
        else:
            lines.append("- _Evidence graph not recorded for this run._\n")

    # Section 8: Pattern Agent Recurring Niche Patterns
    pat_res = agent_results.get("pattern", {})
    if pat_res and pat_res.get("findings"):
        lines.extend([
            "## 8. Pattern Agent: Recurring Niche Patterns & Subgroups",
        ])
        for f in pat_res.get("findings", []):
            lines.append(f"- {f}")
        lines.append("")

    # Section 9: Strategy Agent
    strat_res = agent_results.get("strategy", {})
    if strat_res and strat_res.get("strategic_recommendations"):
        lines.extend([
            "## 9. Evidence-Backed Strategic Recommendations",
        ])
        for idx, item in enumerate(strat_res.get("strategic_recommendations", []), 1):
            lines.append(f"### Action {idx}: {item.get('actionable_recommendation')}")
            lines.append(f"- **Observation**: {item.get('observation')}")
            lines.append(f"- **Evidence**: {item.get('evidence')}")
            lines.append(f"- **Implication**: {item.get('implication') or item.get('recommendation')}")
            lines.append("")

    return "\n".join(lines)


class ReportService:
    """Canonical ReportService class providing static helpers for report generation and persistence."""

    @staticmethod
    def generate_markdown_report(data: dict[str, Any]) -> str:
        return _build_markdown_report(data)

    @staticmethod
    def save_report(payload: dict[str, Any], base_name: str | None = None) -> tuple[Path, Path]:
        return save_report(payload, base_name=base_name)
