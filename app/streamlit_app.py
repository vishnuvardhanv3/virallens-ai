"""ViralLens AI - Next-Gen Instagram Reel Intelligence & Competitor Dashboard.

Architecture:
1. User Upload MP4
2. Twelve Labs Video Intelligence (Pegasus 1.5)
3. Laboratory-Derived Human Visual Attention Model (NEMAR BBBD)
4. Multi-Agent Analytical Deep Dive (Attention, Hook, Behavior, Emotion, Editing, Communication, Strategy)
5. Multi-Word Niche-Anchored Query Generation via Discovery Agent
6. Apify Official Client Discovery
7. Real Competitor MP4 Download & Validation
8. Twelve Labs Competitor Analysis for EVERY selected video
9. Semantic Relevance Verification (verified_relevant vs analyzed_not_relevant)
10. Competitor Attention Model Inference & Benchmark Comparison
11. Empirical Pattern Agent Analysis (Exact counts on verified competitors)
12. Evidence-Backed Strategy Synthesis & Audit Report
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st

from config import settings
from core.master_agent import MasterAgent
from services.metrics_service import format_number, format_percentage
from services.canonical_schema import normalize_score_0_to_1

st.set_page_config(
    page_title="ViralLens AI - Video Intelligence & Human Attention Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-End Styling: Glassmorphism, Dark Mode, Elegant Cards
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #151a2e 0%, #0b0d17 65%, #05070c 100%);
    color: #e2e8f0;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-verified { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
.badge-analyzed { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
.badge-downloaded { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
.badge-ranked { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
.badge-discovered { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }
.badge-rejected { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
.badge-failed { background: rgba(220, 38, 38, 0.25); color: #fca5a5; border: 1px solid rgba(220, 38, 38, 0.5); }
.badge-lab { background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); }

.glass-card {
    background: rgba(23, 28, 48, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.metric-box {
    text-align: center;
    padding: 14px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.metric-val {
    font-size: 1.6rem;
    font-weight: 800;
    color: #38bdf8;
    margin-top: 4px;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.disclaimer-banner {
    background: rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #bae6fd;
    margin-bottom: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("⚡ ViralLens AI — Multimodal Reel Intelligence")
st.caption("Human Visual Attention Model • Twelve Labs Multimodal Understanding • Multi-Agent Architecture • Apify Discovery")

# Sidebar Configuration & Diagnostic Info
with st.sidebar:
    st.header("⚙️ System Status")
    tl_ok = bool(settings.TWELVELABS_API_KEY)
    apify_ok = bool(settings.APIFY_API_TOKEN)

    res_sidebar = st.session_state.get("pipeline_result")
    diag_sidebar = res_sidebar.get("discovery_diagnostics", {}) if res_sidebar else {}
    health_val = str(diag_sidebar.get("discovery_health") or "").lower()

    if health_val == "healthy":
        disc_badge = "🟢 Healthy"
    elif health_val == "partial":
        disc_badge = "🟡 Partial"
    elif health_val == "degraded":
        disc_badge = "🟠 Degraded"
    elif health_val == "failed":
        disc_badge = "🔴 Failed"
    else:
        disc_badge = "⚪ Ready"

    from core.startup_check import validate_system_health, HealthState
    sys_health = validate_system_health()

    if sys_health.overall_state == HealthState.READY:
        sys_badge = "🟢 READY"
    elif sys_health.overall_state == HealthState.DEGRADED:
        sys_badge = "🟡 DEGRADED"
    else:
        sys_badge = "🔴 FAILED"

    st.markdown(f"**System**: {sys_badge}")
    if sys_health.overall_state == HealthState.FAILED:
        for r in sys_health.failed_reasons:
            st.error(f"⚠️ {r}")
    elif sys_health.overall_state == HealthState.DEGRADED:
        for r in sys_health.degraded_reasons:
            st.caption(f"ℹ️ {r}")

    disc_provider = str(diag_sidebar.get("provider") or getattr(settings, "DISCOVERY_PROVIDER", "apify")).title()
    st.markdown(f"**Twelve Labs**: {'🟢 Configured' if tl_ok else '🔴 Missing Key'}")
    st.markdown(f"**Apify API**: {'🟢 Configured' if apify_ok else '🔴 Missing Token'}")
    st.markdown(f"**Provider**: `{disc_provider}`")
    st.markdown(f"**Health**: {disc_badge}")
    st.markdown(f"**Attention Model**: 🟢 READY (NEMAR Trained)")
    st.markdown(f"**Actor**: `{settings.APIFY_INSTAGRAM_REELS_ACTOR}`")
    st.markdown(f"**Target Pool**: `{getattr(settings, 'TARGET_DISCOVERY_CANDIDATES', 30)}`")
    st.markdown(f"**Overall Cap**: `{settings.INSTAGRAM_MAX_RESULTS}`")
    st.divider()
    st.markdown("🔒 *Zero Fabrication Policy:* All findings reflect genuine video analysis, laboratory eye-tracking attention models, and verified competitor metrics.")

# Session State Init
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "live_logs" not in st.session_state:
    st.session_state.live_logs = []
if "early_target_data" not in st.session_state:
    st.session_state.early_target_data = None
if "current_active" not in st.session_state:
    st.session_state.current_active = None

# Section 1: UPLOAD
st.subheader("1. 📤 Upload Reel")
uploaded_file = st.file_uploader(
    "Choose an Instagram Reel MP4 file", type=["mp4", "mov", "mkv", "webm"], help="Upload an MP4 Reel video for multimodal analysis"
)

if uploaded_file is not None:
    temp_dir = Path(tempfile.gettempdir()) / "virallens_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload_{uploaded_file.name}"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col_vid, col_action = st.columns([1, 2])
    with col_vid:
        st.video(str(temp_path))
    with col_action:
        st.markdown(f"**Filename:** `{uploaded_file.name}`")
        st.markdown(f"**Size:** `{round(temp_path.stat().st_size / (1024*1024), 2)} MB`")

        start_btn = st.button("🚀 Analyze & Discover Competitors", type="primary", use_container_width=True)

    if start_btn:
        st.session_state.live_logs = []
        st.session_state.early_target_data = None
        progress_ui = st.empty()
        early_preview_ui = st.empty()
        log_box = st.empty()

        stage_defs = [
            ("asset_prep", "📦 Video Optimization & Pre-processing"),
            ("twelve_labs", "🎬 Multimodal Understanding (Twelve Labs AI)"),
            ("local_signals", "🧠 Attention Inference (NEMAR) & OCR"),
            ("domain_agents", "🤖 Specialized Multi-Agent Deep Dive"),
            ("discovery", "🔍 Instagram Reel Discovery (Apify)"),
            ("competitors", "📊 Competitor Analysis & Verification"),
            ("strategy_report", "📝 Strategy Synthesis & Audit Report"),
        ]

        stage_state = {
            k: {"label": label, "status": "pending", "duration": 0.0, "start": None}
            for k, label in stage_defs
        }
        query_progress_lines: list[str] = []
        captured_logs: list[str] = []
        logs_lock = threading.Lock()
        main_thread_id = threading.get_ident()
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
            main_ctx = get_script_run_ctx()
        except Exception:
            main_ctx = None

        try:
            st.session_state.current_active = "asset_prep"
        except Exception:
            pass
        stage_state["asset_prep"]["status"] = "running"
        stage_state["asset_prep"]["start"] = time.perf_counter()

        def render_checklist():
            now = time.perf_counter()
            html = "<div class='glass-card' style='padding: 16px 20px;'>"
            html += "<div style='font-weight: 700; font-size: 1rem; color: #38bdf8; margin-bottom: 12px;'>⚡ Pipeline Execution Progress</div>"
            for k, d in stage_state.items():
                st_val = d["status"]
                lbl = d["label"]
                if st_val == "done":
                    dur_str = f"{d['duration']:.1f}s"
                    html += f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 5px 0; color: #34d399;'><span><span class='badge badge-verified'>✓ DONE</span> <b>{lbl}</b></span><span style='font-family: monospace; font-size: 0.85rem; color: #94a3b8;'>{dur_str}</span></div>"
                elif st_val == "running":
                    elapsed = now - (d["start"] or now)
                    html += f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 5px 0; color: #38bdf8;'><span><span class='badge badge-lab'>⏳ ACTIVE</span> <b>{lbl}</b></span><span style='font-family: monospace; font-size: 0.85rem; color: #38bdf8;'>{elapsed:.1f}s</span></div>"
                else:
                    html += f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 5px 0; color: #64748b;'><span><span class='badge badge-discovered'>○ PENDING</span> {lbl}</span><span style='font-family: monospace; font-size: 0.85rem; color: #475569;'>--</span></div>"

                # If in discovery stage and have per-query updates, render them cleanly under discovery
                if k == "discovery" and query_progress_lines:
                    html += "<div style='margin-left: 24px; padding: 6px 10px; background: rgba(0,0,0,0.25); border-radius: 6px; margin-top: 4px; margin-bottom: 4px; font-family: monospace; font-size: 0.80rem;'>"
                    for q_line in query_progress_lines:
                        q_col = "#34d399" if "successful" in q_line or "cached" in q_line else ("#f87171" if "blocked" in q_line else "#94a3b8")
                        html += f"<div style='color: {q_col}; padding: 2px 0;'>{q_line}</div>"
                    html += "</div>"

            html += "</div>"
            try:
                progress_ui.markdown(html, unsafe_allow_html=True)
            except Exception:
                pass

        def set_stage(new_stage: str):
            now = time.perf_counter()
            current_id = threading.get_ident()
            if current_id != main_thread_id and main_ctx is not None:
                try:
                    add_script_run_ctx(threading.current_thread(), main_ctx)
                except Exception:
                    pass

            active = None
            try:
                active = getattr(st.session_state, "current_active", None)
            except Exception:
                active = None
            if active and active != new_stage and active in stage_state:
                if stage_state[active]["status"] == "running":
                    stage_state[active]["status"] = "done"
                    stage_state[active]["duration"] = round(now - (stage_state[active]["start"] or now), 2)
            try:
                st.session_state.current_active = new_stage
            except Exception:
                pass
            if new_stage in stage_state and stage_state[new_stage]["status"] != "done":
                stage_state[new_stage]["status"] = "running"
                if not stage_state[new_stage]["start"]:
                    stage_state[new_stage]["start"] = now
            try:
                render_checklist()
            except Exception:
                pass

        def on_progress(msg: str):
            with logs_lock:
                captured_logs.append(msg)
                current_id = threading.get_ident()
                if current_id != main_thread_id:
                    if main_ctx is not None:
                        try:
                            add_script_run_ctx(threading.current_thread(), main_ctx)
                        except Exception:
                            return
                    else:
                        return

                try:
                    if hasattr(st.session_state, "live_logs") and isinstance(st.session_state.live_logs, list):
                        st.session_state.live_logs.append(msg)
                    else:
                        st.session_state.live_logs = list(captured_logs)
                except Exception:
                    pass

                low = msg.lower()
                if "query " in low and ("—" in msg or "blocked" in low or "successful" in low or "cached" in low):
                    query_progress_lines.append(msg.strip())
                if "asset preparation" in low:
                    set_stage("asset_prep")
                elif "twelve labs" in low and ("uploaded reel" in low or "starting twelve labs" in low):
                    set_stage("twelve_labs")
                elif "extracting local" in low or "ocr" in low or "human attention model" in low:
                    set_stage("local_signals")
                elif "multi-agent system" in low or "domain agents" in low:
                    set_stage("domain_agents")
                elif "discovery agent" in low or "executing apify" in low or "raw candidates" in low:
                    set_stage("discovery")
                elif "download: competitor" in low or "twelve labs: competitor" in low or "relevance:" in low or "ranking: selected" in low:
                    set_stage("competitors")
                elif "strategy agent" in low or "report: saved" in low:
                    set_stage("strategy_report")

                try:
                    render_checklist()
                    snippet = "\n".join(captured_logs[-10:])
                    log_box.code(snippet, language="log")
                except Exception:
                    pass

        def on_target_ready(data: dict[str, Any]):
            try:
                st.session_state.early_target_data = data
            except Exception:
                pass
            reel_info = data.get("your_reel", {})
            profile_info = data.get("discovery_profile", {})
            agents_info = data.get("agent_results", {})
            hook_info = agents_info.get("hook", {})
            att_info = agents_info.get("human_attention", {})
            ent = profile_info.get("primary_entity") or reel_info.get("primary_entity") or "N/A"
            niche = profile_info.get("niche") or reel_info.get("niche") or "N/A"
            sub = profile_info.get("sub_niche") or "N/A"
            etype = profile_info.get("edit_type") or "N/A"
            hk_type = hook_info.get("hook_type") or "N/A"
            hk_score = hook_info.get("attention_score") or 0.0

            try:
                early_preview_ui.markdown(
                    f"""
<div class='glass-card' style='border-left: 4px solid #34d399; margin-top: 10px; margin-bottom: 12px;'>
    <div style='font-size: 0.85rem; font-weight: 700; color: #34d399; text-transform: uppercase;'>
        🎯 Target Video Intelligence (Ready Immediately)
    </div>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-top: 8px;'>
        <div><span style='color: #94a3b8; font-size: 0.75rem;'>Primary Entity</span><br><b style='color: #f8fafc;'>{ent}</b></div>
        <div><span style='color: #94a3b8; font-size: 0.75rem;'>Niche</span><br><b style='color: #f8fafc;'>{str(niche).title()}</b></div>
        <div><span style='color: #94a3b8; font-size: 0.75rem;'>Sub-Niche</span><br><b style='color: #38bdf8;'>{str(sub).title()}</b></div>
        <div><span style='color: #94a3b8; font-size: 0.75rem;'>Edit Type</span><br><b style='color: #fbbf24;'>{str(etype).title()}</b></div>
        <div><span style='color: #94a3b8; font-size: 0.75rem;'>Hook Classification</span><br><b style='color: #c084fc;'>{hk_type}</b></div>
        <div><span style='color: #94a3b8; font-size: 0.75rem;'>Hook Potential (0–3s)</span><br><b style='color: #34d399;'>{hk_score:.2f}</b></div>
    </div>
    <div style='font-size: 0.75rem; color: #94a3b8; margin-top: 8px;'>
        ⏳ <i>Instagram competitor discovery in progress below...</i>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        render_checklist()
        agent = MasterAgent(progress_callback=on_progress, on_target_ready=on_target_ready)
        try:
            res = agent.run(temp_path)
            now = time.perf_counter()
            for k, d in stage_state.items():
                if d["status"] == "running":
                    d["status"] = "done"
                    d["duration"] = round(now - (d["start"] or now), 2)
            render_checklist()

            try:
                st.session_state.live_logs = captured_logs
                st.session_state.pipeline_result = res
            except Exception:
                pass
            disc_health = str(res.get("discovery_diagnostics", {}).get("discovery_health") or "").lower()
            cands_count = len(res.get("selected_candidates", [])) or res.get("discovery_diagnostics", {}).get("candidates_returned", 0)
            if cands_count == 0 and disc_health == "failed":
                st.warning("⚠️ Apify failure handling passed; no competitor analysis was possible in this run.")
            elif cands_count == 0:
                st.error("❌ Competitor discovery did not produce usable results.")
            elif disc_health in {"partial", "degraded"}:
                st.warning("⚠️ Competitor discovery completed (partial): Some Instagram discovery queries were blocked by Instagram.")
            else:
                st.success("✅ Complete multi-stage intelligence analysis finished successfully!")
        except Exception as exc:
            st.error(f"❌ Pipeline Execution Error: {exc}")

# Display Results If Available
if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result
    reel = res.get("your_reel", {})
    queries = res.get("queries", [])
    diag = res.get("discovery_diagnostics", {})
    verified = res.get("verified_competitors", [])
    rejected = res.get("rejected_competitors", [])
    analyzed = res.get("analyzed_competitors", [])
    agent_results = res.get("agent_results", {})
    att_comp = res.get("attention_comparison", {})
    human_att = agent_results.get("human_attention", {})
    attention_predictions = reel.get("attention_predictions", [])

    st.divider()

    # Section 2: HUMAN VISUAL ATTENTION (LABORATORY-DERIVED SIGNAL)
    st.subheader("2. 🧠 Predicted Visual-Attention Potential")
    st.markdown(
        """
<div class='disclaimer-banner'>
    <b>🔬 Laboratory-Derived Attention Model (NEMAR BBBD Benchmark):</b><br>
    This signal represents <b>predicted visual-attention potential</b> estimated from laboratory eye-tracking gaze benchmarks.
    It is <b>NOT</b> an Instagram scroll-stop measurement, actual viewer retention, or guaranteed audience behavior.
</div>
""",
        unsafe_allow_html=True,
    )

    if human_att.get("generalization_warning"):
        st.warning(
            "⚠️ **Low Temporal Variation**: Low temporal variation in the model output; "
            "temporal differences should be interpreted cautiously rather than as proof of sustained audience attention."
        )

    ha1, ha2, ha3, ha4, ha5, ha6 = st.columns(6)
    with ha1:
        base_s = human_att.get("baseline_attention")
        base_str = f"{base_s:.2f}" if base_s is not None else "0.50"
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Baseline Attention</div><div class='metric-val'>{base_str}</div></div>", unsafe_allow_html=True)
    with ha2:
        first_ev_t = human_att.get("first_attention_event_time")
        ev_str = f"{first_ev_t:.2f}s" if first_ev_t is not None else "None detected"
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>First Major Event</div><div class='metric-val' style='font-size: 1.1rem;'>{ev_str}</div></div>", unsafe_allow_html=True)
    with ha3:
        pk_score = human_att.get("peak_attention")
        pk_str = f"{pk_score:.2f}" if pk_score is not None else "N/A"
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Peak Potential</div><div class='metric-val'>{pk_str}</div></div>", unsafe_allow_html=True)
    with ha4:
        pk_time = human_att.get("peak_attention_time")
        pkt_str = f"{pk_time:.2f}s" if pk_time is not None else "N/A"
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Peak Time</div><div class='metric-val'>{pkt_str}</div></div>", unsafe_allow_html=True)
    with ha5:
        low_score = human_att.get("lowest_attention")
        low_str = f"{low_score:.2f}" if low_score is not None else "N/A"
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Lowest Attention</div><div class='metric-val'>{low_str}</div></div>", unsafe_allow_html=True)
    with ha6:
        stab_score = human_att.get("attention_stability")
        stab_str = f"{stab_score:.2f}" if stab_score is not None else "N/A"
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Curve Stability</div><div class='metric-val'>{stab_str}</div></div>", unsafe_allow_html=True)

    # Attention Curve Chart (Strictly Chronological)
    if attention_predictions:
        sorted_preds = sorted(
            attention_predictions,
            key=lambda x: (float(x.get("start_time", x.get("start", 0))), float(x.get("end_time", x.get("end", 0))))
        )
        st.markdown("#### 📈 Predicted Visual-Attention Potential Curve (0.5s Chronological Windows)")
        times = [float(p.get("start_time", p.get("start", 0))) for p in sorted_preds]
        scores = [float(p.get("attention_score", 0)) for p in sorted_preds]

        chart_df = pd.DataFrame({"Time (s)": times, "Your Reel": scores}).set_index("Time (s)")

        # Add competitor average curve if available
        if att_comp and att_comp.get("competitor_stats"):
            comp_curves = [s.get("curve", []) for s in att_comp["competitor_stats"].values() if s.get("curve")]
            if comp_curves:
                max_len = len(scores)
                avg_comp_curve = []
                for idx in range(max_len):
                    vals = [c[idx] for c in comp_curves if len(c) > idx]
                    avg_comp_curve.append(sum(vals) / len(vals) if vals else 0.5)
                chart_df["Competitor Benchmark Avg"] = avg_comp_curve

        st.line_chart(chart_df, color=["#38bdf8", "#34d399"] if "Competitor Benchmark Avg" in chart_df else ["#38bdf8"])

    # Chronological Attention Timeline (Strictly sorted by numeric start_time ascending)
    from services.report_service import sort_timeline_events
    raw_timeline = human_att.get("timeline_events", [])
    timeline_events = sort_timeline_events(raw_timeline)

    if timeline_events:
        st.markdown("#### ⏱️ Chronological Attention Timeline")
        st.markdown("`0s ────────────────────────────── END`")
        display_events = timeline_events[:8]
        row_size = 4
        for r_start in range(0, len(display_events), row_size):
            chunk = display_events[r_start:r_start + row_size]
            row_cols = st.columns(len(chunk))
            for c_idx, ev in enumerate(chunk):
                with row_cols[c_idx]:
                    score_val = ev.get("attention_score", 0)
                    score_col = "#34d399" if score_val >= 0.70 else ("#38bdf8" if score_val >= 0.50 else "#f87171")
                    st.markdown(
                        f"<div class='glass-card' style='padding: 12px; margin-bottom: 8px;'>"
                        f"<div style='font-size: 0.75rem; color: #94a3b8; font-weight: 600;'>{ev.get('timestamp')}</div>"
                        f"<div style='font-size: 1.2rem; font-weight: 800; color: {score_col};'>Potential: {score_val:.2f}</div>"
                        f"<div style='font-size: 0.8rem; color: #cbd5e1; margin-top: 4px;'><b>Event:</b> {ev.get('major_observed_event')}</div>"
                        f"<div style='font-size: 0.72rem; color: #64748b; margin-top: 2px;'>{ev.get('attention_mechanism')}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # Section 2b: EXPERIMENTAL SPATIAL SALIENCY (SHADOW MODE RESEARCH PROTOTYPE)
    shadow_data = res.get("dhf1k_spatial_saliency_signal") or reel.get("dhf1k_spatial_saliency_signal") or {}
    if shadow_data:
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.subheader("2b. 🧪 Experimental Spatial Saliency (Shadow Mode)")
        st.markdown(
            """
<div class='disclaimer-banner'>
    <b>🔬 Experimental Research Output (TinySalNet Saliency Model):</b><br>
    This signal is an experimental 2D visual-attention density estimate running strictly in shadow mode for observational telemetry.<br>
    <b>Not used to determine competitor selection, ranking, recommendations, or the primary attention score.</b>
</div>
""",
            unsafe_allow_html=True,
        )

        if shadow_data.get("status") == "SUCCESS":
            s_summ = shadow_data.get("summary", {})
            s_telem = shadow_data.get("telemetry", {})
            s_samp = shadow_data.get("sampling_metadata", {})
            s_prov = shadow_data.get("model_provenance", {})

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Sampled Frames</div><div class='metric-val'>{s_samp.get('sampled_frame_count', 0)}</div><div style='font-size: 0.72rem; color: #94a3b8;'>@ ~1 fps</div></div>", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Spatial Dispersion</div><div class='metric-val'>{s_summ.get('mean_spatial_dispersion', 0.0):.4f}</div><div style='font-size: 0.72rem; color: #94a3b8;'>RMS radial spread</div></div>", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Spatial Entropy</div><div class='metric-val'>{s_summ.get('mean_spatial_entropy_bits', 0.0):.2f} <span style='font-size: 0.9rem;'>bits</span></div><div style='font-size: 0.72rem; color: #94a3b8;'>density spread</div></div>", unsafe_allow_html=True)
            with sc4:
                st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Shadow Latency</div><div class='metric-val'>{s_telem.get('shadow_ms_per_frame', 0.0):.1f} <span style='font-size: 0.9rem;'>ms/f</span></div><div style='font-size: 0.72rem; color: #94a3b8;'>Total: {s_telem.get('shadow_total_wall_time', 0.0):.2f}s</div></div>", unsafe_allow_html=True)

            samples = shadow_data.get("samples", [])
            if samples:
                with st.expander("🗺️ Saliency Map Previews & Per-Frame Telemetry", expanded=False):
                    thumb_cols = st.columns(min(6, len(samples)))
                    for idx, sample in enumerate(samples[:6]):
                        with thumb_cols[idx]:
                            st.caption(f"⏱️ {sample.get('timestamp_sec', 0.0):.1f}s (F{sample.get('frame_idx')})")
                            thumb_raw = sample.get("saliency_map_thumbnail")
                            if thumb_raw:
                                thumb_arr = np.array(thumb_raw, dtype=np.float32)
                                thumb_norm = (thumb_arr / (thumb_arr.max() + 1e-9) * 255).astype(np.uint8)
                                thumb_color = cv2.applyColorMap(thumb_norm, cv2.COLORMAP_JET)
                                thumb_rgb = cv2.cvtColor(thumb_color, cv2.COLOR_BGR2RGB)
                                st.image(thumb_rgb, use_container_width=True)
                            st.markdown(f"<div style='font-size: 0.7rem; color: #94a3b8;'>Disp: {sample.get('saliency_spatial_dispersion', 0):.3f}</div>", unsafe_allow_html=True)
        elif shadow_data.get("status") == "FAILED":
            st.warning(f"⚠️ Shadow spatial analysis failed: `{shadow_data.get('error')}`. Primary ViralLens pipeline unaffected.")

    # Section 3: MULTI-AGENT DEEP DIVE TABS
    st.subheader("3. 👥 Multi-Agent Analytical Deep Dive")
    tab_att, tab_ocr, tab_hook, tab_beh, tab_emo, tab_edit, tab_comm, tab_strat = st.tabs([
        "Attention", "On-Screen Text", "Hook", "Behavior", "Emotion", "Editing", "Communication", "Strategy"
    ])

    with tab_att:
        st.markdown("### 👁️ Human Attention Agent")
        for f in human_att.get("findings", []):
            st.markdown(f"- 🔎 {f}")
        st.markdown("#### Empirical Evidence")
        for e in human_att.get("evidence", []):
            st.markdown(f"- 📊 `{e}`")
        st.markdown("#### Research Uncertainty")
        for u in human_att.get("uncertainties", []):
            st.caption(f"ℹ️ {u}")

    with tab_ocr:
        st.markdown("### 🔤 Specialist Video OCR & Typography Analysis")
        ocr_data = agent_results.get("ocr") or reel.get("ocr") or {}
        segments = ocr_data.get("segments", [])

        o_c1, o_c2, o_c3, o_c4, o_c5 = st.columns(5)
        o_c1.metric("Text Present", "Yes" if ocr_data.get("text_present") else "No")
        o_c2.metric("Segments", len(segments))
        o_c3.metric("First Text", f"{float(ocr_data.get('first_text_time', 0)):.2f}s" if ocr_data.get('first_text_time') is not None else "None")
        o_c4.metric("Total Duration", f"{float(ocr_data.get('total_text_duration', 0)):.2f}s")
        o_c5.metric("Avg Confidence", f"{float(ocr_data.get('ocr_confidence', 0)):.1f}%")

        if segments:
            st.markdown("#### Detected Text Segments")
            for chunk_start in range(0, len(segments), 3):
                chunk = segments[chunk_start : chunk_start + 3]
                cols = st.columns(len(chunk))
                for c_col, seg in zip(cols, chunk):
                    with c_col:
                        st.markdown(
                            f"<div class='glass-card' style='padding: 12px; margin-bottom: 10px; border-left: 3px solid #38bdf8;'>"
                            f"<div style='font-size: 0.75rem; color: #fbbf24; font-weight: 700;'>"
                            f"⏱️ {float(seg.get('start_time', 0)):.2f}s – {float(seg.get('end_time', 0)):.2f}s ({float(seg.get('duration', 0)):.2f}s)"
                            f"</div>"
                            f"<div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin: 6px 0;'>"
                            f"\"{seg.get('text', '')}\""
                            f"</div>"
                            f"<div style='font-size: 0.75rem; color: #94a3b8;'>"
                            f"Position: <b>{seg.get('position', 'unknown')}</b> | Conf: <b>{float(seg.get('confidence', 0)):.1f}%</b>"
                            f"</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            with st.expander("🔍 Full Aggregated On-Screen Text"):
                st.code(ocr_data.get("ocr_text", "") or "No persistent text", language="text")
        else:
            st.info("ℹ️ No persistent, substantive on-screen text was detected across sampled video frames.")

        diagnostics = ocr_data.get("diagnostics", {})
        if diagnostics:
            with st.expander("⚙️ OCR Technical Diagnostics"):
                st.json(diagnostics)

    with tab_hook:
        st.markdown("### 🎣 Hook Agent (First 1–3 Seconds)")
        hook_res = agent_results.get("hook", {})
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown(f"- **Hook Type:** `{hook_res.get('hook_type', 'N/A')}`")
            st.markdown(f"- **Attention Potential (0–3s):** `{hook_res.get('attention_score', 'N/A')}`")
            st.markdown(f"- **First Attention Event:** `{hook_res.get('first_attention_event', 'N/A')}s`")
        with h_col2:
            st.markdown(f"- **Confidence:** `{hook_res.get('confidence', 'N/A')}`")
            st.markdown(f"- **Recommendation:** {hook_res.get('recommendation', 'N/A')}")
        if hook_res.get("weaknesses"):
            st.warning("⚠️ Identified Hook Weaknesses: " + "; ".join(hook_res["weaknesses"]))

    with tab_beh:
        st.markdown("### 🧍 Human Behavior Agent")
        beh_res = agent_results.get("human_behavior", {})
        st.markdown(f"**Confidence:** `{beh_res.get('confidence', 'N/A')}` | **Direct Camera Engagement:** `{beh_res.get('direct_camera_engagement', 'N/A')}`")
        st.markdown("#### Strict Separation: Observation vs Interpretation")
        for b in beh_res.get("observable_behaviors", []):
            st.markdown(
                f"<div class='glass-card' style='padding: 12px; margin-bottom: 8px;'>"
                f"<div style='color: #38bdf8; font-size: 0.85rem;'><b>OBSERVATION:</b> {b.get('observation')}</div>"
                f"<div style='color: #94a3b8; font-size: 0.8rem; margin-top: 4px;'><b>INTERPRETATION:</b> {b.get('interpretation')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with tab_emo:
        st.markdown("### ❤️ Emotion Agent")
        emo_res = agent_results.get("emotion", {})
        st.markdown(f"- **Peak Expressive Intensity:** `{emo_res.get('emotion_intensity', 'N/A')}` at `{emo_res.get('peak_emotion_time', 'N/A')}s`")
        for f in emo_res.get("findings", []):
            st.markdown(f"- {f}")
        for e in emo_res.get("evidence", []):
            st.caption(f"Evidence: {e}")

    with tab_edit:
        st.markdown("### ✂️ Editing Agent")
        edit_res = agent_results.get("editing", {})
        ed1, ed2, ed3 = st.columns(3)
        ed1.metric("Cut Rate", f"{edit_res.get('cut_rate', 0)} cuts/s")
        ed2.metric("Mean Shot Duration", f"{edit_res.get('mean_shot_duration', 0)}s")
        ed3.metric("Pacing Style", str(edit_res.get('pacing', 'N/A')).title())
        for f in edit_res.get("findings", []):
            st.markdown(f"- {f}")
        for ev in edit_res.get("aligned_events", []):
            st.info(f"⏱️ {ev.get('relationship')}")

    with tab_comm:
        st.markdown("### 🗣️ Communication Agent")
        comm_res = agent_results.get("communication", {})
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Word Count", comm_res.get("word_count", 0))
        cm2.metric("Direct Address ('you')", "Yes" if comm_res.get("has_direct_address") else "No")
        cm3.metric("Style", str(comm_res.get("communication_style", "N/A")).replace("_", " ").title())
        for f in comm_res.get("findings", []):
            st.markdown(f"- {f}")
        if comm_res.get("transcript_excerpt"):
            st.code(comm_res["transcript_excerpt"], language="text")

    with tab_strat:
        st.markdown("### 🎯 Strategy Agent (Evidence-Backed Synthesis)")
        strat_res = agent_results.get("strategy", {})
        recs = strat_res.get("strategic_recommendations") or strat_res.get("action_items") or []
        for idx, item in enumerate(recs, 1):
            action = item.get("RECOMMENDATION") or item.get("actionable_recommendation") or item.get("action", "")
            obs = item.get("OBSERVATION") or item.get("observation", "")
            ev = item.get("EVIDENCE") or item.get("evidence", "")
            comp = item.get("COMPARISON") or item.get("comparison", "")
            conf = item.get("CONFIDENCE") or item.get("confidence", "")
            lim = item.get("LIMITATION") or item.get("limitation") or item.get("implication", "")

            st.markdown(
                f"<div class='glass-card' style='border-left: 4px solid #38bdf8; margin-bottom: 12px;'>"
                f"<div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc;'>Action {idx}: {action}</div>"
                f"<div style='font-size: 0.85rem; color: #94a3b8; margin-top: 6px;'><b>Observation:</b> {obs}</div>"
                f"<div style='font-size: 0.85rem; color: #34d399; margin-top: 2px;'><b>Empirical Evidence:</b> {ev}</div>"
                + (f"<div style='font-size: 0.82rem; color: #38bdf8; margin-top: 2px;'><b>Verified Competitor Comparison:</b> {comp}</div>" if comp else "")
                + (f"<div style='font-size: 0.8rem; color: #fbbf24; margin-top: 2px;'><b>Confidence:</b> {conf}</div>" if conf else "")
                + (f"<div style='font-size: 0.8rem; color: #cbd5e1; margin-top: 2px;'><b>Limitation:</b> {lim}</div>" if lim else "")
                + f"</div>",
                unsafe_allow_html=True,
            )

    # Section 4: DISCOVERY & QUERIES
    st.subheader("4. 🎯 Niche-Anchored Discovery & Provider Health")
    disc_health = str(diag.get("discovery_health") or "unknown").lower()
    query_results = diag.get("query_results", [])
    candidates_count = diag.get("candidates_returned", diag.get("raw_candidates", len(verified)))
    attempted_count = len(query_results) if query_results else len(queries)
    succ_count = len(diag.get("successful_queries", [])) or sum(1 for q in query_results if q.get("status") == "success")
    blk_count = len(diag.get("blocked_queries", [])) or sum(1 for q in query_results if q.get("status") == "blocked")

    dh_col1, dh_col2, dh_col3, dh_col4, dh_col5 = st.columns([1.5, 1, 1, 1, 1])
    with dh_col1:
        if disc_health == "healthy":
            st.markdown("<div class='metric-box'><div class='metric-lbl'>Discovery Health</div><div class='metric-val' style='color:#34d399; font-size:1.3rem;'>🟢 HEALTHY</div></div>", unsafe_allow_html=True)
        elif disc_health == "partial":
            st.markdown("<div class='metric-box'><div class='metric-lbl'>Discovery Health</div><div class='metric-val' style='color:#fbbf24; font-size:1.3rem;'>🟡 PARTIAL</div></div>", unsafe_allow_html=True)
        elif disc_health == "degraded":
            st.markdown("<div class='metric-box'><div class='metric-lbl'>Discovery Health</div><div class='metric-val' style='color:#f97316; font-size:1.3rem;'>🟠 DEGRADED</div></div>", unsafe_allow_html=True)
        elif disc_health == "failed":
            st.markdown("<div class='metric-box'><div class='metric-lbl'>Discovery Health</div><div class='metric-val' style='color:#f87171; font-size:1.3rem;'>🔴 FAILED</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Discovery Health</div><div class='metric-val' font-size:1.3rem;'>{disc_health.upper()}</div></div>", unsafe_allow_html=True)
    with dh_col2:
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Attempted</div><div class='metric-val'>{attempted_count}</div></div>", unsafe_allow_html=True)
    with dh_col3:
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Succeeded</div><div class='metric-val' style='color:#34d399;'>{succ_count}</div></div>", unsafe_allow_html=True)
    with dh_col4:
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Blocked</div><div class='metric-val' style='color:#f87171;'>{blk_count}</div></div>", unsafe_allow_html=True)
    with dh_col5:
        st.markdown(f"<div class='metric-box'><div class='metric-lbl'>Discovered</div><div class='metric-val' style='color:#38bdf8;'>{candidates_count}</div></div>", unsafe_allow_html=True)

    if disc_health == "failed":
        st.error("❌ Instagram discovery was blocked for all attempted queries. No verified competitors were generated.")
    elif disc_health in {"partial", "degraded"}:
        st.warning("⚠️ Some Instagram discovery queries were blocked by Instagram. Results below are based on successfully retrieved queries.")

    # Section 18: Compact Discovery Diagnostics & Fallback Notification
    if diag.get("fallback_triggered"):
        fb_ev = diag.get("fallback_event", {})
        p_from = str(fb_ev.get("primary_provider", "apify")).title()
        p_to = str(fb_ev.get("secondary_provider", "agent_reach")).title()
        trig_reason = fb_ev.get("trigger", "policy_trigger")
        st.warning(f"🔄 **Fallback:** {p_from} → {p_to} *(Trigger: `{trig_reason}`)*")

    active_provider = str(diag.get("provider") or getattr(settings, "DISCOVERY_PROVIDER", "apify")).title()
    compact_diagnostics_html = f"<b>Provider:</b> {active_provider} &nbsp;|&nbsp; <b>Health:</b> {disc_health}"
    if query_results:
        q_items = []
        for q_idx, q_state in enumerate(query_results, 1):
            q_st = str(q_state.get("status", "unknown")).lower()
            cand_c = q_state.get("results_count", 0)
            if q_st == "success":
                q_items.append(f"Q{q_idx} — success — {cand_c} candidates")
            elif q_st == "blocked":
                q_items.append(f"Q{q_idx} — blocked")
            elif q_st == "empty":
                q_items.append(f"Q{q_idx} — empty")
            else:
                q_items.append(f"Q{q_idx} — {q_st}")
        compact_diagnostics_html += "<br><span style='color: #94a3b8; font-size: 0.85rem;'>" + " &nbsp;•&nbsp; ".join(q_items) + "</span>"

    st.markdown(
        f"<div class='glass-card' style='padding: 12px 16px; margin-bottom: 14px; border-left: 3px solid #38bdf8;'>"
        f"{compact_diagnostics_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Competitive Context Card
    disc_profile = res.get("discovery_profile") or {}
    primary_ent = disc_profile.get("primary_entity") or reel.get("primary_entity") or ""
    niche_val = str(disc_profile.get("niche") or reel.get("niche") or "general")
    sub_niche_val = str(disc_profile.get("sub_niche") or "spider-man fan edits")
    edit_type_val = str(disc_profile.get("edit_type") or "action montage")
    sub_niche_conf = float(disc_profile.get("sub_niche_confidence") or 0.85)
    edit_type_conf = float(disc_profile.get("edit_type_confidence") or 0.90)

    st.markdown(
        f"<div class='glass-card' style='border-left: 4px solid #38bdf8; margin-bottom: 16px;'>"
        f"<div style='font-size: 0.75rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;'>🎯 Competitive Context</div>"
        f"<div style='display: flex; flex-wrap: wrap; gap: 24px; margin-top: 10px;'>"
        f"<div><span style='color: #94a3b8; font-size: 0.8rem;'>Entity:</span> <b style='color: #f8fafc;'>{primary_ent or 'None (Generic Topic)'}</b></div>"
        f"<div><span style='color: #94a3b8; font-size: 0.8rem;'>Niche:</span> <b style='color: #f8fafc;'>{niche_val.title()}</b></div>"
        f"<div><span style='color: #94a3b8; font-size: 0.8rem;'>Sub-Niche:</span> <b style='color: #34d399;'>{sub_niche_val.title()}</b> ({sub_niche_conf*100:.0f}% conf)</div>"
        f"<div><span style='color: #94a3b8; font-size: 0.8rem;'>Edit Type:</span> <b style='color: #fbbf24;'>{edit_type_val.title()}</b> ({edit_type_conf*100:.0f}% conf)</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Discovery Entity Diagnostics
    ent_type = disc_profile.get("entity_type", "unknown")
    ent_usable = disc_profile.get("entity_usable_for_discovery", False)

    ep_col1, ep_col2, ep_col3, ep_col4 = st.columns(4)
    ep_col1.metric("Anchor Entity", primary_ent if primary_ent else "None (Generic Topic)")
    ep_col2.metric("Entity Type", ent_type.replace("_", " ").title())
    ep_col3.metric("Discovery Usable", "✅ Usable" if ent_usable else "⛔ Filtered (Fallback)")
    ep_col4.metric("Topic / Niche", f"{disc_profile.get('topic', 'creative')} / {niche_val}")

    st.markdown("All queries strictly retain the niche anchor and entity topic (zero single-word searches):")
    q_cols = st.columns(len(queries)) if queries else [st.empty()]
    for idx, (col, q) in enumerate(zip(q_cols, queries), 1):
        with col:
            st.markdown(
                f"<div class='glass-card' style='padding: 12px; text-align: center;'>"
                f"<div class='badge badge-ranked'>{q.get('search_level', 'query')}</div>"
                f"<div style='font-family: JetBrains Mono; font-size: 0.85rem; margin-top: 8px; font-weight: 600;'>{q.get('query')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Section 5: COMPETITORS & VERIFICATION
    st.subheader("5. 🛡️ Verified Competitors & Performance Benchmarks")
    unique_creators = len(set(
        str(c.get("creator") or c.get("username") or "unknown").lower() for c in verified
    ))
    st.markdown(
        f"**Verified Relevant Reels:** `{len(verified)}` | "
        f"**Unique Creators:** `{unique_creators}` | "
        f"**Rejected (Unrelated):** `{len(rejected)}`"
    )

    # Competitor Clusters Display
    comp_clusters = res.get("competitor_clusters") or diag.get("competitor_clusters") or {}
    primary_cluster = comp_clusters.get("primary_cluster", {})
    sec_clusters = comp_clusters.get("secondary_clusters", [])

    if primary_cluster and primary_cluster.get("count", 0) > 0:
        st.markdown(
            f"#### 🏆 PRIMARY COMPETITOR CLUSTER: {primary_cluster.get('cluster_name', 'Action Montage').upper()} "
            f"({primary_cluster.get('count', 0)} verified reels, {primary_cluster.get('unique_creators', 0)} creators, "
            f"avg reach: {format_number(primary_cluster.get('avg_reach', 0))})"
        )
        p_cands = primary_cluster.get("competitors", [])[: min(4, len(primary_cluster.get("competitors", [])))]
        p_cols = st.columns(len(p_cands))
        for idx, (col, c) in enumerate(zip(p_cols, p_cands), 1):
            with col:
                cid = c.get("shortcode") or c.get("id") or f"c_{idx}"
                creator = c.get("creator") or c.get("username") or "creator"
                reach_val = format_number(c.get("reach_value") or c.get("views") or 0)
                perf_score = f"{float(c.get('performance_score', 0)):.2f}"
                rel_score = f"{float(c.get('video_relevance_score', 0)):.2f}"
                st.markdown(
                    f"<div class='glass-card' style='padding: 12px; border-top: 3px solid #34d399;'>"
                    f"<div style='font-size: 0.72rem; color: #34d399; font-weight: 700;'>PRIMARY CLUSTER #{idx}</div>"
                    f"<div style='font-weight: 700; color: #f8fafc; margin: 4px 0;'>@{creator}</div>"
                    f"<div style='font-size: 0.8rem; color: #38bdf8;'>Reach: <b>{reach_val}</b> ({c.get('reach_metric', 'views')})</div>"
                    f"<div style='font-size: 0.75rem; color: #94a3b8;'>Performance: <b>{perf_score}</b> | Relevance: <b>{rel_score}</b></div>"
                    f"<div style='font-size: 0.7rem; color: #cbd5e1; margin-top: 4px;'>ID: [{cid}]</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if sec_clusters:
        with st.expander(f"🌐 Secondary Competitor Clusters ({sum(c.get('count', 0) for c in sec_clusters)} reels across {len(sec_clusters)} clusters)"):
            for sc in sec_clusters:
                st.markdown(f"**{sc.get('cluster_name')}** — `{sc.get('count')} reels` (Avg Reach: `{format_number(sc.get('avg_reach', 0))}`)")
                for sc_cand in sc.get("competitors", [])[:3]:
                    st.caption(f"- @{sc_cand.get('creator')} [{sc_cand.get('shortcode') or sc_cand.get('id')}]: reach {format_number(sc_cand.get('reach_value') or sc_cand.get('views') or 0)} — {sc_cand.get('selection_reason', '')}")

    # Sorting options
    sort_option = st.selectbox(
        "Sort Competitors By:",
        [
            "Highest Reach (Default)",
            "Highest Engagement",
            "Highest Relevance",
            "Best Combined Performance",
            "Creator Diversity",
        ],
        index=0,
    )

    def _safe_reach(c: dict[str, Any]) -> int:
        for k in ("reach_value", "views", "plays", "view_count", "videoViewCount", "videoPlayCount"):
            v = c.get(k)
            if v is not None:
                try:
                    iv = int(v)
                    if iv >= 0:
                        return iv
                except (ValueError, TypeError):
                    pass
        return 0

    def _comp_sort_key(c: dict[str, Any]):
        reach_val = _safe_reach(c)
        likes_val = int(c.get("likes") or 0)
        rel_val = float(c.get("video_relevance_score") or 0.0)
        perf_val = float(c.get("performance_score") or 0.0)
        if "Engagement" in sort_option:
            return (likes_val, perf_val, reach_val)
        elif "Relevance" in sort_option:
            return (rel_val, perf_val, reach_val)
        elif "Combined" in sort_option:
            return (perf_val, reach_val, rel_val)
        elif "Diversity" in sort_option:
            creator_str = str(c.get("creator") or c.get("username") or "")
            return (creator_str, -reach_val)
        else:  # Highest Reach
            return (reach_val, perf_val, rel_val)

    display_verified = sorted(verified, key=_comp_sort_key, reverse=True if "Diversity" not in sort_option else False)

    # Dedicated Section: 🔥 Highest-Reach Verified Competitors
    if display_verified:
        st.markdown("#### 🔥 Highest-Reach Verified Competitors")
        top_slice = display_verified[: min(4, len(display_verified))]
        top_cols = st.columns(len(top_slice))
        for idx, (col, c) in enumerate(zip(top_cols, top_slice), 1):
            with col:
                cid = c.get("shortcode") or c.get("id") or f"c_{idx}"
                creator = c.get("creator") or c.get("username") or "creator"
                reach_val = format_number(_safe_reach(c))
                perf_score = f"{float(c.get('performance_score', 0)):.2f}"
                rel_score = f"{float(c.get('video_relevance_score', 0)):.2f}"
                st.markdown(
                    f"<div class='glass-card' style='padding: 12px; border-top: 3px solid #38bdf8;'>"
                    f"<div style='font-size: 0.72rem; color: #fbbf24; font-weight: 700;'>RANK #{idx} REACH</div>"
                    f"<div style='font-weight: 700; color: #f8fafc; margin: 4px 0;'>@{creator}</div>"
                    f"<div style='font-size: 0.8rem; color: #34d399;'>Reach: <b>{reach_val}</b> ({c.get('reach_metric', 'views')})</div>"
                    f"<div style='font-size: 0.75rem; color: #94a3b8;'>Performance: <b>{perf_score}</b> | Relevance: <b>{rel_score}</b></div>"
                    f"<div style='font-size: 0.7rem; color: #cbd5e1; margin-top: 4px;'>ID: [{cid}]</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("#### All Competitors & Verification Audit")
    for comp in (display_verified + rejected):
        cid = comp.get("shortcode") or comp.get("id") or "unknown"
        url = comp.get("url") or comp.get("source_url") or "#"
        creator = comp.get("creator") or comp.get("username") or "unknown"
        status = comp.get("status", "ANALYZED")

        badge_class = "badge-verified" if status == "VERIFIED" else ("badge-rejected" if status == "REJECTED" else "badge-failed")

        # Extract structured relevance matches
        rel_dims = comp.get("video_relevance_dimensions") or comp.get("relevance_evidence") or {}
        ent_m = f"{float(comp.get('entity_match', rel_dims.get('entity_similarity', 0))):.2f}"
        top_m = f"{float(comp.get('topic_match', rel_dims.get('topic_similarity', 0))):.2f}"
        sn_m = f"{float(comp.get('sub_niche_match', rel_dims.get('sub_niche_similarity', 0))):.2f}"
        et_m = f"{float(comp.get('edit_type_match', rel_dims.get('edit_type_similarity', 0))):.2f}"

        with st.container():
            st.markdown(
                f"<div class='glass-card'>"
                f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                f"<div><span class='badge {badge_class}'>{status}</span> <b><a href='{url}' target='_blank' style='color: #38bdf8; text-decoration: none;'>@{creator} — [{cid}]</a></b></div>"
                f"<div>Relevance: <b>{format_percentage(comp.get('video_relevance_score'))}</b> ({comp.get('evidence_label', 'Relevance')})</div>"
                f"</div>"
                f"<p style='font-size: 0.85rem; color: #94a3b8; margin: 8px 0 4px 0;'><b>Verification Reason:</b> {comp.get('video_relevance_reason', 'N/A')}</p>"
                f"<div style='display: flex; gap: 16px; font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px;'>"
                f"<span>Entity: <b>{ent_m}</b></span>"
                f"<span>Topic: <b>{top_m}</b></span>"
                f"<span>Sub-Niche: <b>{sn_m}</b></span>"
                f"<span>Edit-Type: <b>{et_m}</b></span>"
                f"</div>"
                f"<div style='display: flex; gap: 20px; font-size: 0.8rem; color: #cbd5e1;'>"
                f"<span>👀 Reach ({comp.get('reach_metric', 'views')}): <b>{format_number(comp.get('reach_value') or comp.get('views'))}</b></span>"
                f"<span>❤️ Likes: <b>{format_number(comp.get('likes'))}</b></span>"
                f"<span>⚡ Performance: <b>{float(comp.get('performance_score') or 0):.2f}</b></span>"
                f"<span>🎣 Hook Strength: <b>{format_percentage((comp.get('hook') or {}).get('strength'))}</b></span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Section 6: COMPETITOR INTELLIGENCE & RECURRING PATTERNS
    st.subheader("6. 🔍 Competitor Intelligence: Verified Recurring Patterns")
    pattern_agent_res = agent_results.get("pattern", {})
    comp_intel = agent_results.get("competitor_intelligence") or pattern_agent_res.get("competitor_intelligence") or {}

    if comp_intel and comp_intel.get("status") == "SUCCESS":
        sg_counts = comp_intel.get("subgroup_counts", {})
        st.markdown(
            f"<div style='display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;'>"
            f"<span class='badge' style='background: #1e293b; color: #38bdf8;'>🎯 Same Entity & Edit: <b>{sg_counts.get('same_entity_and_edit_type', 0)}</b></span>"
            f"<span class='badge' style='background: #1e293b; color: #a855f7;'>🎬 Same Edit Type: <b>{sg_counts.get('same_edit_type', 0)}</b></span>"
            f"<span class='badge' style='background: #1e293b; color: #34d399;'>📊 All Verified: <b>{sg_counts.get('all_verified', 0)}</b></span>"
            f"<span class='badge' style='background: #1e293b; color: #f59e0b;'>🔥 Highest Reach: <b>{sg_counts.get('highest_reach_same_edit_type', 0)}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        recurring = comp_intel.get("recurring_patterns", [])
        if recurring:
            card_cols = st.columns(min(len(recurring[:4]), 4))
            for col, p in zip(card_cols, recurring[:4]):
                with col:
                    t_val = p.get("target_value", "absent")
                    bg_badge = "#065f46" if t_val == "present" else "#7f1d1d"
                    badge_lbl = "Target: Aligned" if t_val == "present" else "Target: Differs"
                    st.markdown(
                        f"<div class='glass-card' style='padding: 12px; height: 100%;'>"
                        f"<div class='metric-lbl' style='font-size: 0.8rem;'>{p.get('pattern_name')}</div>"
                        f"<div class='metric-val' style='font-size: 1.4rem; color: #38bdf8;'>{p.get('numerator')}/{p.get('denominator')}</div>"
                        f"<div style='font-size: 0.72rem; color: #94a3b8;'>Subgroup: {p.get('subgroup')}</div>"
                        f"<div style='margin-top: 6px;'><span style='background: {bg_badge}; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 0.68rem;'>{badge_lbl}</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Gap analysis and findings
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        gaps = comp_intel.get("gap_analysis", [])
        if gaps:
            with st.expander("🔎 Target vs Competitor Gap Analysis & Creative Implications", expanded=False):
                for g in gaps:
                    st.markdown(f"- **{g.get('pattern_name')}**: {g.get('interpretation')}")
                creatives = comp_intel.get("creative_implications", [])
                if creatives:
                    st.markdown("#### Evidence-Backed Creative Recommendations")
                    for ci in creatives:
                        st.markdown(f"- **{ci.get('pattern_name')}**: {ci.get('recommendation')}")
                        st.caption(f"Evidence: {ci.get('evidence')} | Limitation: {ci.get('limitation')}")

    elif pattern_agent_res and pattern_agent_res.get("patterns"):
        p_cols = st.columns(len(pattern_agent_res["patterns"]))
        for col, p in zip(p_cols, pattern_agent_res["patterns"]):
            with col:
                st.markdown(
                    f"<div class='glass-card' style='text-align: center; padding: 14px;'>"
                    f"<div class='metric-lbl'>{p.get('pattern')}</div>"
                    f"<div class='metric-val' style='font-size: 1.5rem;'>{p.get('count')}</div>"
                    f"<div style='font-size: 0.72rem; color: #94a3b8; margin-top: 4px;'>Verified Competitors</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        for f in pattern_agent_res.get("findings", []):
            st.markdown(f"- {f}")

    # Section 7: RECOMMENDATION INTELLIGENCE: EVIDENCE-BACKED GUIDANCE
    st.subheader("7. 💡 Recommendation Intelligence: Evidence-Backed Guidance")
    rec_intel = (
        res.get("recommendation_intelligence")
        or agent_results.get("recommendation_intelligence")
        or (agent_results.get("strategy") or {}).get("recommendation_intelligence")
        or {}
    )

    if rec_intel and rec_intel.get("status") == "SUCCESS":
        recs = rec_intel.get("recommendations", [])
        categories = rec_intel.get("categories", {})
        counts_summary = {k: len(v) for k, v in categories.items() if v}

        pill_html = "<div style='display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap;'>"
        pill_colors = {
            "PRESERVE": ("rgba(16, 185, 129, 0.2)", "#34d399"),
            "CHANGE": ("rgba(245, 158, 11, 0.2)", "#fbbf24"),
            "TEST": ("rgba(168, 85, 247, 0.2)", "#c084fc"),
            "INVESTIGATE": ("rgba(59, 130, 246, 0.2)", "#60a5fa"),
            "AVOID_OVERINTERPRETATION": ("rgba(148, 163, 184, 0.2)", "#cbd5e1"),
            "NO_ACTIONABLE_RECOMMENDATION": ("rgba(239, 68, 68, 0.2)", "#f87171"),
        }
        for cat_name, count in counts_summary.items():
            bg_col, txt_col = pill_colors.get(cat_name, ("rgba(255,255,255,0.1)", "#fff"))
            pill_html += f"<span class='badge' style='background: {bg_col}; color: {txt_col}; border: 1px solid {txt_col}44;'>{cat_name}: <b>{count}</b></span>"
        pill_html += f"<span class='badge' style='background: #1e293b; color: #94a3b8;'>Cohort: <b>{rec_intel.get('comparison_group', 'N/A')}</b> (n={rec_intel.get('competitor_count', 0)})</span>"
        pill_html += "</div>"
        st.markdown(pill_html, unsafe_allow_html=True)

        for idx, rec in enumerate(recs, 1):
            cat = rec.get("category", "TEST")
            prio = str(rec.get("priority", "medium")).upper()
            title = rec.get("recommendation_id", f"REC_{idx}").replace("_", " ").title()

            prio_color = "#ef4444" if prio == "HIGH" else ("#f59e0b" if prio == "MEDIUM" else "#38bdf8")
            bg_col, txt_col = pill_colors.get(cat, ("rgba(255,255,255,0.1)", "#fff"))

            t_comp = rec.get("target_comparison", {})
            diff_text = t_comp.get("difference", "Target differs from observed peer cohort.")
            target_val = t_comp.get("target", "N/A")
            comp_val = t_comp.get("comparison_group", "N/A")

            card_html = f"""
            <div class='glass-card' style='border-left: 4px solid {txt_col}; margin-bottom: 16px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <div>
                        <span class='badge' style='background: {bg_col}; color: {txt_col};'>{cat}</span>
                        <span class='badge' style='background: rgba(255,255,255,0.06); color: {prio_color}; border: 1px solid {prio_color}66; margin-left: 6px;'>{prio} PRIORITY</span>
                        <span style='font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-left: 10px;'>{title}</span>
                    </div>
                    <div style='font-size: 0.75rem; color: #94a3b8;'>Strength: <b>{rec.get('strength', 'observed')}</b></div>
                </div>
                <div style='margin-top: 6px; font-size: 0.9rem;'><b>🔭 Observation:</b> <span style='color: #cbd5e1;'>{rec.get('observation', 'N/A')}</span></div>
                <div style='margin-top: 4px; font-size: 0.88rem;'><b>⚖️ Target vs Cohort:</b> <span style='color: #94a3b8;'>Target: [{target_val}] vs Cohort: [{comp_val}] — {diff_text}</span></div>
                <div style='margin-top: 4px; font-size: 0.88rem;'><b>💡 Creative Implication:</b> <span style='color: #cbd5e1;'>{rec.get('creative_implication', 'N/A')}</span></div>
                <div style='margin-top: 8px; padding: 10px; background: rgba(56, 189, 248, 0.08); border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.2);'>
                    <div style='font-size: 0.92rem; color: #38bdf8; font-weight: 600;'>🎯 Recommendation:</div>
                    <div style='font-size: 0.88rem; color: #e2e8f0; margin-top: 2px;'>{rec.get('recommendation', 'N/A')}</div>
                </div>
                <div style='margin-top: 8px; padding: 10px; background: rgba(168, 85, 247, 0.08); border-radius: 8px; border: 1px solid rgba(168, 85, 247, 0.2);'>
                    <div style='font-size: 0.92rem; color: #c084fc; font-weight: 600;'>🧪 Testable Action (Creator Experiment):</div>
                    <div style='font-size: 0.88rem; color: #e2e8f0; margin-top: 2px;'>{rec.get('testable_action', 'N/A')}</div>
                </div>
                <div style='margin-top: 8px; font-size: 0.78rem; color: #94a3b8; font-style: italic;'>
                    ⚠️ <b>Limitation:</b> {rec.get('limitation', 'Observational peer evidence only.')}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            with st.expander(f"🔬 Evidence Trace & Competitor Cohort for {title}", expanded=False):
                ev_list = rec.get("evidence", [])
                for ev in ev_list:
                    num_den_str = f" ({ev.get('numerator')}/{ev.get('denominator')})" if ev.get('denominator') is not None else ""
                    st.markdown(f"- **{ev.get('type', 'Signal')}:** {ev.get('value')}{num_den_str}")
                    st.caption(f"Source: {ev.get('source')} | Cohort: {ev.get('provenance', {}).get('comparison_group', 'N/A')}")
                    comp_ids = ev.get("competitor_ids", [])
                    if comp_ids:
                        st.caption(f"Supporting Competitors ({len(comp_ids)}): `{', '.join(comp_ids[:6])}`")
                prov = rec.get("provenance", {})
                if prov:
                    st.caption(f"Traceability: Target: `{prov.get('target_reel_id')}` | Features: `{', '.join(prov.get('source_features', []))}`")
    else:
        st.info("ℹ️ No actionable recommendations generated for this reel due to empty or unverified competitor baseline.")

    # Section 8: FINAL REPORT DOWNLOAD
    st.subheader("8. 📑 Final Audit Report")
    st.markdown(f"- **JSON Report:** `{res.get('report_json_path')}`")
    st.markdown(f"- **Markdown Report:** `{res.get('report_md_path')}`")

    if res.get("report_md_path") and os.path.exists(res.get("report_md_path")):
        with open(res.get("report_md_path"), "r", encoding="utf-8") as f:
            md_text = f.read()
        st.download_button(
            label="📥 Download Markdown Audit Report",
            data=md_text,
            file_name=os.path.basename(res.get("report_md_path")),
            mime="text/markdown",
            use_container_width=True,
        )

    # Section 9: PIPELINE PERFORMANCE & BOTTLENECK ANALYSIS
    st.subheader("9. ⚡ Pipeline Performance & Bottleneck Analysis")
    perf = res.get("performance_report")
    if perf:
        p_meta = perf.get("video_metadata", {})
        p_stages = perf.get("stages", {})
        p_bn = perf.get("primary_bottleneck", {})

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.markdown(
                f"<div class='metric-box'><div class='metric-lbl'>Total Pipeline Time</div><div class='metric-val'>{perf.get('total_duration_sec', 0.0):.2f}s</div></div>",
                unsafe_allow_html=True,
            )
        with p_col2:
            st.markdown(
                f"<div class='metric-box'><div class='metric-lbl'>Original File Size</div><div class='metric-val'>{p_meta.get('file_size_mb', 0.0):.2f} MB</div></div>",
                unsafe_allow_html=True,
            )
        with p_col3:
            st.markdown(
                f"<div class='metric-box'><div class='metric-lbl'>Analysis Payload Size</div><div class='metric-val'>{p_meta.get('analysis_size_mb', p_meta.get('file_size_mb', 0.0)):.2f} MB</div></div>",
                unsafe_allow_html=True,
            )
        with p_col4:
            ratio_pct = f"{p_meta.get('compression_ratio', 0.0) * 100:.1f}%" if p_meta.get('has_derivative') else "0.0%"
            st.markdown(
                f"<div class='metric-box'><div class='metric-lbl'>Payload Optimization</div><div class='metric-val'>{ratio_pct}</div></div>",
                unsafe_allow_html=True,
            )

        if p_bn and p_bn.get("stage"):
            st.info(
                f"ℹ️ **Primary Pipeline Bottleneck**: `{p_bn.get('stage')}` ({p_bn.get('duration_sec', 0.0):.2f}s, "
                f"{p_bn.get('percentage_of_total', 0.0):.1f}% of total runtime). "
                f"Local video OCR and attention inference executed concurrently to prevent serial blocking."
            )

        if p_stages:
            stage_rows = []
            for s_name, s_info in p_stages.items():
                stage_rows.append({
                    "Pipeline Stage": s_name,
                    "Duration (s)": round(s_info.get("duration_sec", 0.0), 2),
                    "% of Total": round(s_info.get("percentage_of_total", 0.0), 1),
                })
            stage_df = pd.DataFrame(stage_rows).sort_values(by="Duration (s)", ascending=False)
            st.dataframe(stage_df, use_container_width=True, hide_index=True)
