r"""
Real-Reel Validation Script for Phase 14 Attention Intelligence Integration.

Executes on verified real Reel MP4s:
1. E:\new viral\htdyr.mp4
2. C:\Users\vishn\AppData\Local\Temp\virallens_uploads\upload_jvxawbcy.mp4

Measures decoupled latencies:
A. Pure Attention Intelligence computation (pre-extracted signals in memory)
B. Serialization (JSON construction)
C. Visualization rendering (image generation)
D. Total integration overhead
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, r"E:\new viral")

from services.human_attention_service import HumanAttentionService
from services.shadow_spatial_service import ShadowSpatialService
from services.attention_intelligence import (
    AttentionIntelligenceService,
    render_attention_visualization,
    SCHEMA_VERSION,
)
from services.report_service import save_report
from config import settings

VIDEO_HTDYR = Path(r"E:\new viral\htdyr.mp4")
VIDEO_JVXAWBCY = Path(r"C:\Users\vishn\AppData\Local\Temp\virallens_uploads\upload_jvxawbcy.mp4")

OUTPUT_DIR = Path(r"E:\new viral\models\human_attention_experiments\attention_intelligence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_reel(video_path: Path, video_label: str) -> dict:
    print(f"\n==================================================")
    print(f"Validating Real Reel: {video_label}")
    print(f"Path: {video_path}")
    print(f"File Size: {video_path.stat().st_size:,} bytes")
    print(f"==================================================")

    # 1. Run frozen NEMAR temporal attention model
    print("[1/4] Running frozen NEMAR temporal attention...")
    t_nemar_start = time.perf_counter()
    nemar_service = HumanAttentionService()
    nemar_preds = nemar_service.predict_attention(video_path)
    t_nemar_sec = time.perf_counter() - t_nemar_start
    print(f"  NEMAR complete: {len(nemar_preds)} windows in {t_nemar_sec:.3f}s")

    # 2. Run frozen TinySalNet spatial model
    print("[2/4] Running frozen TinySalNet spatial saliency...")
    t_spatial_start = time.perf_counter()
    spatial_service = ShadowSpatialService()
    spatial_data = spatial_service.analyze_video(video_path, max_budget=15)
    t_spatial_sec = time.perf_counter() - t_spatial_start
    print(f"  TinySalNet complete: {len(spatial_data.get('samples', []))} frames in {t_spatial_sec:.3f}s (status: {spatial_data.get('status')})")

    # 3. Decoupled Metric Measurements
    print("[3/4] Measuring decoupled Attention Intelligence latencies...")

    # A. Pure Attention Intelligence computation (pre-extracted signals in memory)
    # Warm up / multi-run for reliable measurement
    durations_a = []
    for _ in range(5):
        t_a0 = time.perf_counter()
        ai = AttentionIntelligenceService.build_intelligence(
            nemar_predictions=nemar_preds,
            spatial_data=spatial_data,
            context={"niche": "creative", "primary_entity": "subject"},
            source_video_path=video_path,
        )
        durations_a.append((time.perf_counter() - t_a0) * 1000.0)
    latency_a_ms = round(float(sum(durations_a) / len(durations_a)), 3)

    # B. Serialization (JSON construction)
    durations_b = []
    for _ in range(5):
        t_b0 = time.perf_counter()
        _ = json.dumps(ai, ensure_ascii=False)
        durations_b.append((time.perf_counter() - t_b0) * 1000.0)
    latency_b_ms = round(float(sum(durations_b) / len(durations_b)), 3)

    # C. Visualization rendering (image generation)
    vis_out_path = settings.REPORTS_DIR / f"attention_vis_{video_label}.png"
    t_c0 = time.perf_counter()
    vis_path = render_attention_visualization(video_path, ai, vis_out_path)
    latency_c_ms = round((time.perf_counter() - t_c0) * 1000.0, 3)

    # D. Total integration overhead
    latency_d_ms = round(latency_a_ms + latency_b_ms + latency_c_ms, 3)

    print(f"  A. Pure Attention Intelligence: {latency_a_ms:.3f} ms (Target: < 50 ms -> {'PASS' if latency_a_ms < 50.0 else 'FAIL'})")
    print(f"  B. JSON Serialization:         {latency_b_ms:.3f} ms")
    print(f"  C. Visualization Rendering:    {latency_c_ms:.3f} ms")
    print(f"  D. Total Integration Overhead: {latency_d_ms:.3f} ms")

    # 4. Report Integration
    print("[4/4] Generating production audit report...")
    payload = {
        "target_video": str(video_path),
        "your_reel": {
            "niche": "creative",
            "topic": "visual edit",
            "primary_entity": "subject",
            "attention_intelligence": ai,
            "dhf1k_spatial_saliency_signal": spatial_data,
            "nemar_attention_signal": nemar_preds,
        },
        "attention_intelligence": ai,
        "dhf1k_spatial_saliency_signal": spatial_data,
        "nemar_attention_signal": nemar_preds,
        "attention_visualization_path": vis_path,
    }
    json_p, md_p = save_report(payload, base_name=f"phase14_{video_label}")
    print(f"  Saved JSON report: {json_p.name}")
    print(f"  Saved Markdown report: {md_p.name}")

    # Summarize extracted intelligence properties
    temporal_summary = {
        "opening_signal": ai["temporal"]["opening_signal"],
        "peak": ai["temporal"]["peak"],
        "peak_timestamp": ai["temporal"]["peak_timestamp"],
        "relative_peak_position": ai["temporal"]["relative_peak_position"],
        "persistence": ai["temporal"]["temporal_persistence"],
        "descriptors": ai["temporal"]["descriptors"],
    }

    spatial_summary = {
        "centroid": ai["spatial"]["centroid"],
        "peak_focus": ai["spatial"]["peak_focus"],
        "dispersion": ai["spatial"]["dispersion"],
        "entropy": ai["spatial"]["entropy"],
        "temporal_centroid_drift": ai["spatial"]["temporal_centroid_drift"],
        "focal_concentration": ai["spatial"]["focal_concentration"],
        "descriptors": ai["spatial"]["descriptors"],
    }

    focal_interpretation = ai["spatial"]["focal_competition"]
    events_summary = [
        {"type": e["event_type"], "start": e["timestamp_start"], "end": e["timestamp_end"], "conf": e["confidence"]}
        for e in ai["events"]
    ]

    return {
        "video_label": video_label,
        "video_path": str(video_path),
        "file_size_bytes": video_path.stat().st_size,
        "schema_version": SCHEMA_VERSION,
        "latencies": {
            "pure_attention_intelligence_computation_ms": latency_a_ms,
            "serialization_ms": latency_b_ms,
            "visualization_rendering_ms": latency_c_ms,
            "total_integration_overhead_ms": latency_d_ms,
            "meets_50ms_pure_criterion": latency_a_ms < 50.0,
        },
        "temporal_summary": temporal_summary,
        "spatial_summary": spatial_summary,
        "focal_interpretation": focal_interpretation,
        "events_count": len(ai["events"]),
        "events": events_summary,
        "creative_interpretation": ai["interpretation"],
        "provenance": ai["provenance"],
        "reports": {
            "json_path": str(json_p),
            "md_path": str(md_p),
            "visualization_path": vis_path,
        },
    }


def main():
    assert VIDEO_HTDYR.exists(), f"Benchmark video missing: {VIDEO_HTDYR}"
    assert VIDEO_JVXAWBCY.exists(), f"Benchmark video missing: {VIDEO_JVXAWBCY}"

    results = {}
    results["htdyr"] = validate_reel(VIDEO_HTDYR, "htdyr")
    results["upload_jvxawbcy"] = validate_reel(VIDEO_JVXAWBCY, "upload_jvxawbcy")

    summary_file = OUTPUT_DIR / "real_reel_validation_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Validation results written to: {summary_file}")


if __name__ == "__main__":
    main()
