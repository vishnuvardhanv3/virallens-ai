"""Pipeline Profiler & Performance Telemetry for ViralLens AI.

Instruments every stage of video ingestion, multi-modal analysis, discovery,
competitor processing, and report synthesis.
Emits structured console logs, provides timing reconciliation, and persists
machine-readable JSON performance reports.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from config.settings import REPORTS_DIR


class TimingCategory:
    """Standardized performance latency categories."""
    LOCAL_CPU = "LOCAL_CPU"
    LOCAL_IO = "LOCAL_IO"
    NETWORK = "NETWORK"
    TWELVE_LABS = "TWELVE_LABS"
    APIFY = "APIFY"
    COMPETITOR_DOWNLOAD = "COMPETITOR_DOWNLOAD"
    SLEEP_BACKOFF = "SLEEP_BACKOFF"
    OTHER_EXTERNAL = "OTHER_EXTERNAL"
    UNATTRIBUTED = "UNATTRIBUTED"

    ALL = [
        LOCAL_CPU,
        LOCAL_IO,
        NETWORK,
        TWELVE_LABS,
        APIFY,
        COMPETITOR_DOWNLOAD,
        SLEEP_BACKOFF,
        OTHER_EXTERNAL,
        UNATTRIBUTED,
    ]


def compute_interval_union(intervals: List[Tuple[float, float]]) -> float:
    """Compute total wall-clock coverage (measure of union) across possibly overlapping time intervals."""
    if not intervals:
        return 0.0
    valid = [(s, e) for s, e in intervals if e >= s and e > 0]
    if not valid:
        return 0.0
    sorted_intervals = sorted(valid, key=lambda x: x[0])
    merged = []
    curr_start, curr_end = sorted_intervals[0]
    for s, e in sorted_intervals[1:]:
        if s <= curr_end:
            curr_end = max(curr_end, e)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = s, e
    merged.append((curr_start, curr_end))
    return round(sum(e - s for s, e in merged), 4)


class StageNode:
    """Represents a single timed stage in the profiler hierarchy with call stats and interval tracking."""

    def __init__(
        self,
        name: str,
        category: str = TimingCategory.LOCAL_CPU,
        parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.category = category
        self.parent = parent
        self.children: List[str] = []
        self.inclusive_duration: float = 0.0
        self.exclusive_duration: float = 0.0
        self.start_perf: float = 0.0
        self.end_perf: float = 0.0
        self.intervals: List[Tuple[float, float]] = []
        self.durations: List[float] = []
        self.call_count: int = 0
        self.child_sum_duration: float = 0.0
        self.child_union_duration: float = 0.0
        self.overlap_duration: float = 0.0
        self.concurrency_level: int = 1
        self.metadata: Dict[str, Any] = metadata or {}
        self.completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        durs = self.durations if self.durations else ([self.inclusive_duration] if self.inclusive_duration > 0 else [])
        min_dur = round(min(durs), 3) if durs else 0.0
        max_dur = round(max(durs), 3) if durs else 0.0
        avg_dur = round(sum(durs) / max(1, len(durs)), 3) if durs else 0.0

        return {
            "name": self.name,
            "category": self.category,
            "parent": self.parent,
            "children": list(self.children),
            "wall_clock_duration": round(self.inclusive_duration, 3),
            "child_union_duration": round(self.child_union_duration, 3),
            "child_sum_duration": round(self.child_sum_duration, 3),
            "overlap_duration": round(self.overlap_duration, 3),
            "exclusive_duration": round(self.exclusive_duration, 3),
            "concurrency_level": self.concurrency_level,
            "inclusive_duration_sec": round(self.inclusive_duration, 3),
            "exclusive_duration_sec": round(self.exclusive_duration, 3),
            "wall_clock_duration_sec": round(self.inclusive_duration, 3),
            "call_count": max(1, self.call_count),
            "min_duration_sec": min_dur,
            "max_duration_sec": max_dur,
            "avg_duration_sec": avg_dur,
            "child_sum_duration_sec": round(self.child_sum_duration, 3),
            "child_union_duration_sec": round(self.child_union_duration, 3),
            "overlap_duration_sec": round(self.overlap_duration, 3),
            "metadata": dict(self.metadata),
        }


class PipelineProfiler:
    """Reusable high-precision timing and performance profiler with hierarchical stage tracking and reconciliation."""

    def __init__(
        self,
        target_video: Optional[str | Path] = None,
        reports_dir: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.target_video = Path(target_video).resolve() if target_video else None
        self.reports_dir = Path(reports_dir or output_dir or REPORTS_DIR / "performance")
        self.start_perf = time.perf_counter()
        self.end_perf: Optional[float] = None
        self._first_stage_started: bool = False
        self._lock = threading.Lock()

        # Flat timings dictionary for backward compatibility (inclusive durations in seconds)
        self.timings: Dict[str, float] = {}

        # Hierarchical stage tracking
        self.stages: Dict[str, StageNode] = {}
        self.root_stages: List[str] = []
        self._stage_stack: List[str] = []

        # Standard operational counters
        self.metrics: Dict[str, Any] = {
            "candidate_count": 0,
            "verified_competitor_count": 0,
            "download_count": 0,
            "twelvelabs_upload_count": 0,
            "twelvelabs_analysis_count": 0,
            "ocr_processing_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retries": 0,
            "failed_operations": 0,
        }
        self.metadata: Dict[str, Any] = {}
        self._active_stages: Dict[str, float] = {}

        if self.target_video and self.target_video.exists():
            self._probe_initial_metadata()

    def start(self) -> None:
        """Start or explicitly reset the monotonic wall-clock timer."""
        with self._lock:
            self.start_perf = time.perf_counter()
            self.end_perf = None
            self._first_stage_started = True

    def _probe_initial_metadata(self) -> None:
        """Capture basic file size and metadata from target video."""
        if not self.target_video or not self.target_video.exists():
            return
        size_bytes = self.target_video.stat().st_size
        self.metadata["file_size_mb"] = round(size_bytes / (1024 * 1024), 2)
        self.metadata["file_size_bytes"] = size_bytes
        self.metadata["filename"] = self.target_video.name
        self.metadata["target_path"] = str(self.target_video)
        self.metadata["resolution"] = "unknown"
        self.metadata["fps"] = 0.0
        self.metadata["duration_sec"] = 0.0

        try:
            import cv2
            cap = cv2.VideoCapture(str(self.target_video))
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur = round(fc / fps, 2) if fps > 0 else 0.0
                cap.release()
                self.metadata["width"] = w
                self.metadata["height"] = h
                self.metadata["resolution"] = f"{w}x{h}"
                self.metadata["fps"] = round(fps, 1)
                self.metadata["frame_count"] = fc
                self.metadata["duration_sec"] = dur
        except Exception:
            pass

    def set_video_metadata(
        self,
        duration_sec: float,
        width: int,
        height: int,
        fps: float,
        frame_count: Optional[int] = None,
        bitrate_kbps: Optional[float] = None,
        codec: Optional[str] = None,
    ) -> None:
        """Update profiler with probed video stream metadata."""
        with self._lock:
            self.metadata["duration_sec"] = round(float(duration_sec), 2)
            self.metadata["width"] = int(width)
            self.metadata["height"] = int(height)
            self.metadata["resolution"] = f"{int(width)}x{int(height)}"
            self.metadata["fps"] = round(float(fps), 1)
            if frame_count is not None:
                self.metadata["frame_count"] = int(frame_count)
            if bitrate_kbps is not None:
                self.metadata["bitrate_kbps"] = round(float(bitrate_kbps), 1)
            if codec:
                self.metadata["codec"] = str(codec)

    def record_metric(self, key: str, value: Any) -> None:
        """Record diagnostic count or telemetry value."""
        with self._lock:
            self.metrics[key] = value

    def increment_metric(self, key: str, amount: int = 1) -> None:
        """Atomically increment a diagnostic metric counter."""
        with self._lock:
            self.metrics[key] = self.metrics.get(key, 0) + amount

    @contextmanager
    def profile_stage(
        self,
        stage_name: str,
        category: str = TimingCategory.LOCAL_CPU,
        parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        concurrency_level: int = 1,
    ) -> Generator[StageNode, None, None]:
        """Context manager measuring execution duration with parent-child hierarchy and concurrency tracking."""
        with self._lock:
            effective_parent = parent
            if effective_parent is None and self._stage_stack:
                effective_parent = self._stage_stack[-1]

            node = self.stages.get(stage_name)
            if node is None:
                node = StageNode(
                    name=stage_name,
                    category=category,
                    parent=effective_parent,
                    metadata=metadata,
                )
                self.stages[stage_name] = node
                if effective_parent is None:
                    if stage_name not in self.root_stages:
                        self.root_stages.append(stage_name)
                else:
                    p_node = self.stages.get(effective_parent)
                    if p_node and stage_name not in p_node.children:
                        p_node.children.append(stage_name)
            else:
                if category != TimingCategory.LOCAL_CPU:
                    node.category = category
                if metadata:
                    node.metadata.update(metadata)
                if effective_parent and node.parent is None:
                    node.parent = effective_parent
                    p_node = self.stages.get(effective_parent)
                    if p_node and stage_name not in p_node.children:
                        p_node.children.append(stage_name)

            if concurrency_level > 1:
                node.concurrency_level = max(node.concurrency_level, concurrency_level)

            t_start = time.perf_counter()
            node.start_perf = t_start
            node.call_count += 1
            if not self._first_stage_started:
                self._first_stage_started = True
                self.start_perf = node.start_perf

            self._stage_stack.append(stage_name)
            self._active_stages[stage_name] = t_start

        try:
            yield node
        finally:
            t_end = time.perf_counter()
            elapsed = t_end - t_start
            with self._lock:
                node.end_perf = t_end
                node.durations.append(round(elapsed, 4))
                node.intervals.append((t_start, t_end))
                if node.completed and node.inclusive_duration > 0:
                    node.inclusive_duration = round(node.inclusive_duration + elapsed, 3)
                else:
                    node.inclusive_duration = round(elapsed, 3)
                node.completed = True

                self.timings[stage_name] = node.inclusive_duration
                self._active_stages.pop(stage_name, None)
                if self._stage_stack and self._stage_stack[-1] == stage_name:
                    self._stage_stack.pop()

    @contextmanager
    def time_stage(
        self,
        stage_name: str,
        category: str = TimingCategory.LOCAL_CPU,
        parent: Optional[str] = None,
        concurrency_level: int = 1,
    ) -> Generator[StageNode, None, None]:
        """Context manager to measure elapsed execution time (alias for profile_stage)."""
        with self.profile_stage(stage_name, category=category, parent=parent, concurrency_level=concurrency_level) as node:
            yield node

    def record_stage_direct(
        self,
        stage_name: str,
        duration: float,
        category: str = TimingCategory.LOCAL_CPU,
        parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """Directly record a known duration for a child stage (e.g. from telemetry)."""
        with self._lock:
            effective_parent = parent
            # If parent alias is used (e.g. 'apify' when stage_f is active), link to active parent if suitable
            if effective_parent and effective_parent not in self.stages:
                for active_p in ("stage_f_instagram_discovery", "stage_g_competitor_download_and_analysis"):
                    if active_p in self.stages:
                        effective_parent = active_p
                        break

            node = self.stages.get(stage_name)
            if node is None:
                node = StageNode(
                    name=stage_name,
                    category=category,
                    parent=effective_parent,
                    metadata=metadata,
                )
                self.stages[stage_name] = node
                if effective_parent:
                    p_node = self.stages.get(effective_parent)
                    if p_node and stage_name not in p_node.children:
                        p_node.children.append(stage_name)
                elif stage_name not in self.root_stages:
                    self.root_stages.append(stage_name)
            else:
                if category != TimingCategory.LOCAL_CPU:
                    node.category = category
                if metadata:
                    node.metadata.update(metadata)
                if effective_parent and node.parent is None:
                    node.parent = effective_parent
                    p_node = self.stages.get(effective_parent)
                    if p_node and stage_name not in p_node.children:
                        p_node.children.append(stage_name)

            dur = round(float(duration), 3)
            node.call_count += 1
            node.durations.append(dur)
            node.inclusive_duration = dur
            node.exclusive_duration = dur
            node.completed = True

            now = time.perf_counter()
            s_time = start_time if start_time is not None else (now - dur)
            e_time = end_time if end_time is not None else now
            node.start_perf = s_time
            node.end_perf = e_time
            node.intervals.append((s_time, e_time))

            self.timings[stage_name] = round(float(duration), 2)

    def start_stage(self, stage_name: str) -> None:
        """Manually mark the start of an async or multi-step stage."""
        with self._lock:
            self._active_stages[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        """Manually mark the end of an async or multi-step stage."""
        with self._lock:
            t0 = self._active_stages.pop(stage_name, None)
            if t0 is None:
                return 0.0
            t_now = time.perf_counter()
            elapsed = round(t_now - t0, 3)
            self.timings[stage_name] = round(elapsed, 2)
            if stage_name in self.stages:
                node = self.stages[stage_name]
                node.inclusive_duration = elapsed
                node.end_perf = t_now
                node.intervals.append((t0, t_now))
                node.completed = True
            return elapsed

    def get_elapsed_total(self) -> float:
        """Calculate total wall-clock time from profiler initialization using monotonic clock."""
        t_end = self.end_perf if self.end_perf is not None else time.perf_counter()
        return round(t_end - self.start_perf, 2)

    def stop(self) -> float:
        """Mark final completion of the profiler."""
        with self._lock:
            if self.end_perf is None:
                self.end_perf = time.perf_counter()
            return self.get_elapsed_total()

    def _compute_exclusive_durations(self) -> None:
        """Compute exclusive duration and concurrent child interval union for every stage."""
        for name, node in self.stages.items():
            if not node.children:
                node.exclusive_duration = node.inclusive_duration
                node.child_sum_duration = 0.0
                node.child_union_duration = 0.0
                node.overlap_duration = 0.0
            else:
                child_nodes = [self.stages[c] for c in node.children if c in self.stages]
                child_sum = sum(c.inclusive_duration for c in child_nodes)

                child_intervals = []
                for c in child_nodes:
                    if c.intervals:
                        child_intervals.extend(c.intervals)
                    elif c.start_perf > 0 and c.end_perf >= c.start_perf:
                        child_intervals.append((c.start_perf, c.end_perf))
                    elif c.inclusive_duration > 0 and node.start_perf > 0:
                        child_intervals.append((node.start_perf, node.start_perf + c.inclusive_duration))

                if child_intervals:
                    child_union = compute_interval_union(child_intervals)
                    if node.inclusive_duration > 0:
                        child_union = min(node.inclusive_duration, child_union)
                else:
                    child_union = min(node.inclusive_duration, child_sum)

                overlap = max(0.0, round(child_sum - child_union, 3))
                node.child_sum_duration = round(child_sum, 3)
                node.child_union_duration = round(child_union, 3)
                node.overlap_duration = round(overlap, 3)

                # Sequential: child_union == child_sum (overlap == 0)
                # Concurrent: parent.inclusive ≈ child_union + parent.exclusive
                node.exclusive_duration = max(0.0, round(node.inclusive_duration - child_union, 3))

    def reconcile_timing(self) -> Dict[str, Any]:
        """Verify that sum of recorded exclusive stage durations reconciles with total wall-clock runtime."""
        self._compute_exclusive_durations()
        total_wall_clock = self.get_elapsed_total()

        recorded_exclusive = round(
            sum(node.exclusive_duration for node in self.stages.values()), 3
        )

        unattributed = max(0.0, round(total_wall_clock - recorded_exclusive, 3))
        unattributed_pct = round(
            (unattributed / max(0.01, total_wall_clock)) * 100, 2
        )

        passed = bool(unattributed_pct <= 5.0)

        # Category breakdown
        category_breakdown: Dict[str, float] = {c: 0.0 for c in TimingCategory.ALL}
        for node in self.stages.values():
            cat = node.category if node.category in category_breakdown else TimingCategory.OTHER_EXTERNAL
            category_breakdown[cat] = round(category_breakdown[cat] + node.exclusive_duration, 3)

        category_breakdown[TimingCategory.UNATTRIBUTED] = unattributed

        # Competitor processing concurrency breakdown
        comp_summary = {}
        for s_name, node in self.stages.items():
            if "competitor" in s_name:
                comp_cids = set()
                for c in node.children:
                    if "competitor_" in c:
                        parts = c.split("_")
                        if len(parts) >= 4:
                            comp_cids.add(f"{parts[1]}_{parts[2]}")
                        elif len(parts) >= 2:
                            comp_cids.add(parts[1])
                        else:
                            comp_cids.add(c)
                num_comps = len(comp_cids) if comp_cids else len(node.children)
                comp_summary = {
                    "stage": s_name,
                    "number_of_competitors": num_comps,
                    "num_competitors": num_comps,
                    "max_concurrency": node.concurrency_level,
                    "wall_clock_duration": node.inclusive_duration,
                    "child_union_duration": node.child_union_duration,
                    "child_sum_duration": node.child_sum_duration,
                    "overlap_duration": node.overlap_duration,
                    "exclusive_duration": node.exclusive_duration,
                    "parent_wall_time": node.inclusive_duration,
                    "child_sum_duration_sec": node.child_sum_duration,
                    "child_union_duration_sec": node.child_union_duration,
                    "overlap_duration_sec": node.overlap_duration,
                    "parent_wall_time_sec": node.inclusive_duration,
                }
                break

        # Validate per-stage timing reconciliation rule
        stage_reconciliations = {}
        for s_name, node in self.stages.items():
            if node.children:
                is_concurrent = bool(node.concurrency_level > 1 or node.overlap_duration > 0.05)
                expected_inclusive = round(node.child_union_duration + node.exclusive_duration, 3)
                diff = abs(round(node.inclusive_duration - expected_inclusive, 3))
                stage_reconciliations[s_name] = {
                    "is_concurrent": is_concurrent,
                    "concurrency_level": node.concurrency_level,
                    "wall_clock_duration": node.inclusive_duration,
                    "child_sum_duration": node.child_sum_duration,
                    "child_union_duration": node.child_union_duration,
                    "overlap_duration": node.overlap_duration,
                    "exclusive_duration": node.exclusive_duration,
                    "reconciled": bool(diff <= 0.05),
                }

        return {
            "total_wall_clock": total_wall_clock,
            "recorded_exclusive_time": recorded_exclusive,
            "unattributed_time": unattributed,
            "unattributed_percent": unattributed_pct,
            "audit_passed": passed,
            "categories": category_breakdown,
            "competitor_concurrency_breakdown": comp_summary,
            "stage_reconciliations": stage_reconciliations,
        }

    def compute_bottleneck(self) -> str:
        """Identify which stage consumed the highest fraction of runtime."""
        self._compute_exclusive_durations()
        if not self.stages:
            filtered = {k: v for k, v in self.timings.items() if k not in {"total", "overall"}}
            if not filtered:
                return "none"
            return max(filtered.items(), key=lambda x: x[1])[0]

        candidates = {}
        for name, node in self.stages.items():
            dur = node.inclusive_duration if node.parent is None else node.exclusive_duration
            candidates[name] = dur

        if not candidates:
            return "none"
        return max(candidates.items(), key=lambda x: x[1])[0]

    def get_top_bottlenecks(self, top_n: int = 3, root_only: bool = True) -> List[Dict[str, Any]]:
        """Return top N bottleneck stages sorted descending by duration."""
        self._compute_exclusive_durations()
        total_sec = self.get_elapsed_total()
        recorded_total = sum(node.exclusive_duration for node in self.stages.values())
        base_total = recorded_total if recorded_total > 0 else max(0.01, total_sec)

        candidate_nodes = [
            node for node in self.stages.values()
            if (not root_only or node.parent is None)
        ]
        if not candidate_nodes:
            candidate_nodes = list(self.stages.values())

        items = []
        for node in candidate_nodes:
            dur = node.inclusive_duration if node.parent is None else node.exclusive_duration
            pct = round((dur / max(0.01, base_total)) * 100, 1)
            items.append({
                "stage": node.name,
                "category": node.category,
                "duration_sec": round(dur, 2),
                "percentage_of_total": pct,
                "is_root": node.parent is None,
            })

        items.sort(key=lambda x: x["duration_sec"], reverse=True)
        return items[:top_n]

    def to_dict(self) -> Dict[str, Any]:
        """Construct machine-readable JSON performance dictionary."""
        reconciliation = self.reconcile_timing()
        total_sec = reconciliation["total_wall_clock"]
        recorded_total = reconciliation["recorded_exclusive_time"]
        base_total = recorded_total if recorded_total > 0 else max(0.01, total_sec)
        top_bottlenecks = self.get_top_bottlenecks(top_n=3, root_only=True)
        primary_b = top_bottlenecks[0] if top_bottlenecks else {
            "stage": "none", "duration_sec": 0.0, "percentage_of_total": 0.0, "category": TimingCategory.LOCAL_CPU
        }

        stages_breakdown = {}
        exclusive_stage_time = {}
        for s_name, s_node in self.stages.items():
            exclusive_stage_time[s_name] = round(s_node.exclusive_duration, 3)
            stages_breakdown[s_name] = {
                "name": s_name,
                "wall_clock_duration": round(s_node.inclusive_duration, 3),
                "wall_clock_duration_sec": round(s_node.inclusive_duration, 3),
                "duration_sec": round(s_node.inclusive_duration, 2),
                "inclusive_duration_sec": round(s_node.inclusive_duration, 3),
                "exclusive_duration": round(s_node.exclusive_duration, 3),
                "exclusive_duration_sec": round(s_node.exclusive_duration, 2),
                "child_sum_duration": round(s_node.child_sum_duration, 3),
                "child_union_duration": round(s_node.child_union_duration, 3),
                "overlap_duration": round(s_node.overlap_duration, 3),
                "concurrency_level": s_node.concurrency_level,
                "category": s_node.category,
                "parent": s_node.parent,
                "children": list(s_node.children),
                "percentage_of_total": round((s_node.inclusive_duration / max(0.01, base_total)) * 100, 1),
                "call_count": max(1, s_node.call_count),
                "avg_duration_sec": round(sum(s_node.durations) / max(1, len(s_node.durations)), 3) if s_node.durations else round(s_node.inclusive_duration, 3),
                "min_duration_sec": round(min(s_node.durations), 3) if s_node.durations else round(s_node.inclusive_duration, 3),
                "max_duration_sec": round(max(s_node.durations), 3) if s_node.durations else round(s_node.inclusive_duration, 3),
                "child_sum_duration_sec": round(s_node.child_sum_duration, 3),
                "child_union_duration_sec": round(s_node.child_union_duration, 3),
                "overlap_duration_sec": round(s_node.overlap_duration, 3),
                "metadata": dict(s_node.metadata),
            }
        for s_name, s_dur in self.timings.items():
            if s_name not in stages_breakdown:
                exclusive_stage_time[s_name] = round(s_dur, 3)
                stages_breakdown[s_name] = {
                    "duration_sec": s_dur,
                    "exclusive_duration_sec": s_dur,
                    "category": TimingCategory.LOCAL_CPU,
                    "parent": None,
                    "children": [],
                    "percentage_of_total": round((s_dur / max(0.01, base_total)) * 100, 1),
                    "call_count": 1,
                    "avg_duration_sec": s_dur,
                    "min_duration_sec": s_dur,
                    "max_duration_sec": s_dur,
                    "child_sum_duration_sec": 0.0,
                    "child_union_duration_sec": 0.0,
                    "overlap_duration_sec": 0.0,
                    "concurrency_level": 1,
                    "metadata": {},
                }

        nested_stage_details = {}
        for s_name, s_node in self.stages.items():
            if s_node.children:
                nested_stage_details[s_name] = {
                    "inclusive_duration_sec": round(s_node.inclusive_duration, 3),
                    "exclusive_duration_sec": round(s_node.exclusive_duration, 3),
                    "child_sum_duration_sec": round(s_node.child_sum_duration, 3),
                    "child_union_duration_sec": round(s_node.child_union_duration, 3),
                    "overlap_duration_sec": round(s_node.overlap_duration, 3),
                    "category": s_node.category,
                    "children": {
                        c: {
                            "inclusive_duration_sec": round(self.stages[c].inclusive_duration, 3),
                            "exclusive_duration_sec": round(self.stages[c].exclusive_duration, 3),
                            "category": self.stages[c].category,
                        }
                        for c in s_node.children
                        if c in self.stages
                    },
                }

        return {
            "target_video": str(self.target_video) if self.target_video else "unknown",
            "file_size_mb": self.metadata.get("file_size_mb", 0.0),
            "duration_sec": self.metadata.get("duration_sec", 0.0),
            "resolution": self.metadata.get("resolution", "unknown"),
            "fps": self.metadata.get("fps", 0.0),
            "frame_count": self.metadata.get("frame_count", 0),
            "codec": self.metadata.get("codec", "unknown"),
            "bitrate_kbps": self.metadata.get("bitrate_kbps", 0.0),
            "video_metadata": dict(self.metadata),
            "timings": dict(self.timings),
            "stages": stages_breakdown,
            "exclusive_stage_time": exclusive_stage_time,
            "nested_stage_details": nested_stage_details,
            "wall_clock_total": total_sec,
            "reconciliation": reconciliation,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "total_sec": total_sec,
            "total_duration_sec": total_sec,
            "bottleneck": primary_b["stage"],
            "primary_bottleneck": primary_b,
            "top_bottlenecks": top_bottlenecks,
            "categories": reconciliation["categories"],
            "timestamp": datetime.now().isoformat(),
        }

    def format_user_performance_report(self) -> str:
        """Format exact structured timing report matching user specifications."""
        reconciliation = self.reconcile_timing()
        total_sec = reconciliation["total_wall_clock"]

        def _stage_dur(name: str) -> float:
            if name in self.stages:
                return self.stages[name].inclusive_duration
            if name in self.timings:
                return self.timings[name]
            for s_name, node in self.stages.items():
                if name in s_name:
                    return node.inclusive_duration
            for t_name, dur in self.timings.items():
                if name in t_name:
                    return dur
            return 0.0

        if "competitor_download" in self.stages:
            comp_downloads = self.stages["competitor_download"].inclusive_duration
        else:
            comp_downloads = sum(
                node.inclusive_duration for name, node in self.stages.items()
                if "download" in name and "competitor" in name and name != "competitors_total"
            )

        if "competitor_analysis" in self.stages:
            comp_analyses = self.stages["competitor_analysis"].inclusive_duration
        else:
            comp_analyses = sum(
                node.inclusive_duration for name, node in self.stages.items()
                if ("twelvelabs" in name or "attention" in name) and "competitor" in name and name != "competitors_total"
            )

        if "relevance" in self.stages:
            comp_relevance = self.stages["relevance"].inclusive_duration
        else:
            comp_relevance = sum(
                node.inclusive_duration for name, node in self.stages.items()
                if "relevance" in name and "competitor" in name and name != "competitors_total"
            )

        lines = [
            "[PERFORMANCE]",
            f"file_size_mb={self.metadata.get('file_size_mb', 0.0)}",
            f"duration_sec={self.metadata.get('duration_sec', 0.0)}",
            f"resolution={self.metadata.get('resolution', 'unknown')}",
            f"fps={self.metadata.get('fps', 0.0)}",
            "",
            "[WALL CLOCK]",
            f"total={total_sec:.2f}s",
            "",
            "[BREAKDOWN]",
            f"asset_preparation={_stage_dur('asset_preparation'):.2f}s",
            f"attention={_stage_dur('attention'):.2f}s",
            f"hook={_stage_dur('hook'):.2f}s",
            f"behavior={_stage_dur('behavior'):.2f}s",
            f"emotion={_stage_dur('emotion'):.2f}s",
            f"editing={_stage_dur('editing'):.2f}s",
            f"communication={_stage_dur('communication'):.2f}s",
            f"twelvelabs={_stage_dur('twelvelabs'):.2f}s",
            f"discovery_profile={_stage_dur('discovery_profile'):.2f}s",
            f"apify={_stage_dur('apify'):.2f}s",
            f"competitor_download={comp_downloads:.2f}s",
            f"competitor_analysis={comp_analyses:.2f}s",
            f"relevance={comp_relevance:.2f}s",
            f"pattern={_stage_dur('pattern'):.2f}s",
            f"strategy={_stage_dur('strategy'):.2f}s",
            f"report={_stage_dur('report'):.2f}s",
            "",
            "[APIFY]",
        ]

        for i in range(1, 6):
            dur = _stage_dur(f"apify_query_{i}")
            if dur == 0.0:
                dur = _stage_dur(f"query_{i}")
            lines.append(f"query_{i}={dur:.2f}s")

        lines.extend([
            "",
            "[RECONCILIATION]",
            f"recorded={reconciliation['recorded_exclusive_time']:.2f}s",
            f"unattributed={reconciliation['unattributed_time']:.2f}s",
            f"unattributed_percent={reconciliation['unattributed_percent']:.2f}%",
        ])

        comp_info = reconciliation.get("competitor_concurrency_breakdown", {})
        if comp_info:
            lines.extend([
                "",
                "[COMPETITOR PROCESSING CONCURRENCY]",
                f"number of competitors: {comp_info.get('number_of_competitors', comp_info.get('num_competitors', 0))}",
                f"max concurrency: {comp_info.get('max_concurrency', 1)}",
                f"child_sum_duration: {comp_info.get('child_sum_duration_sec', comp_info.get('child_sum_duration', 0.0)):.2f}s",
                f"child_union_duration: {comp_info.get('child_union_duration_sec', comp_info.get('child_union_duration', 0.0)):.2f}s",
                f"overlap_duration: {comp_info.get('overlap_duration_sec', comp_info.get('overlap_duration', 0.0)):.2f}s",
                f"parent wall time: {comp_info.get('parent_wall_time_sec', comp_info.get('parent_wall_time', 0.0)):.2f}s",
            ])

        lines.extend([
            "",
            "[TOP BOTTLENECKS]",
        ])

        top_b = self.get_top_bottlenecks(top_n=3)
        for idx, b in enumerate(top_b, 1):
            lines.append(f"{idx}. {b['stage']} ({b['category']}) - {b['duration_sec']:.2f}s ({b['percentage_of_total']}%)")

        return "\n".join(lines)

    def summary_log(self) -> str:
        """Return structured human-readable performance summary string."""
        data = self.to_dict()
        reconciliation = data["reconciliation"]
        lines = [
            "=" * 60,
            "[Performance]",
            f"target={data['target_video']}",
            f"size_mb={data['file_size_mb']}",
            f"duration_sec={data['duration_sec']}",
            f"resolution={data['resolution']}",
            f"fps={data['fps']}",
        ]
        if data.get("codec") and data["codec"] != "unknown":
            lines.append(f"codec={data['codec']}")
        lines.append("")
        lines.append("[Timing by Category]")
        for cat, dur in reconciliation["categories"].items():
            if dur > 0.0:
                pct = round((dur / max(0.01, data['total_sec'])) * 100, 1)
                lines.append(f"  {cat:<25} {dur:>6.2f}s ({pct:>5.1f}%)")

        lines.append("")
        lines.append("[Top Stages]")
        top_stages = self.get_top_bottlenecks(top_n=6)
        for s in top_stages:
            lines.append(f"  {s['stage']:<35} {s['duration_sec']:>6.2f}s ({s['percentage_of_total']:>5.1f}%) [{s['category']}]")

        comp_info = reconciliation.get("competitor_concurrency_breakdown", {})
        if comp_info:
            lines.append("")
            lines.append("[Competitor Processing Concurrency]")
            lines.append(f"  number of competitors: {comp_info.get('number_of_competitors', comp_info.get('num_competitors', 0))}")
            lines.append(f"  max concurrency:       {comp_info.get('max_concurrency', 1)}")
            lines.append(f"  child_sum_duration:    {comp_info.get('child_sum_duration_sec', comp_info.get('child_sum_duration', 0.0)):.2f}s")
            lines.append(f"  child_union_duration:  {comp_info.get('child_union_duration_sec', comp_info.get('child_union_duration', 0.0)):.2f}s")
            lines.append(f"  overlap_duration:      {comp_info.get('overlap_duration_sec', comp_info.get('overlap_duration', 0.0)):.2f}s")
            lines.append(f"  parent wall time:      {comp_info.get('parent_wall_time_sec', comp_info.get('parent_wall_time', 0.0)):.2f}s")

        lines.append(f"  TOTAL{' ' * 31} {data['total_sec']:>6.2f}s (100.0%)")
        lines.append(f"bottleneck={data['bottleneck']} ({data['primary_bottleneck']['duration_sec']:.2f}s)")
        lines.append(f"reconciliation: recorded={reconciliation['recorded_exclusive_time']:.2f}s, unattributed={reconciliation['unattributed_time']:.2f}s ({reconciliation['unattributed_percent']:.1f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print structured human-readable performance summary to stdout."""
        print("\n" + self.summary_log() + "\n")

    def save_report(self, output_dir: Optional[Path | str] = None) -> Dict[str, Any]:
        """Persist machine-readable performance report in reports/performance/ and return data."""
        target_dir = Path(output_dir) if output_dir else self.reports_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        vid_stem = self.target_video.stem if self.target_video else "video"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = target_dir / f"perf_{vid_stem}_{timestamp}.json"

        report_data = self.to_dict()
        report_data["report_path"] = str(report_path)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_data
