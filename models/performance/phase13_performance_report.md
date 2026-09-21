# Phase 13 — ViralLens End-to-End Performance Hardening & Timing Reconciliation Report

**Date:** 2026-09-15  
**Project Root:** `E:\new viral`  
**Dataset Constraints:** `E:\v1.0.0` (READ-ONLY, untouched)  
**Safety & Compliance:** Strict performance engineering. No models added/retrained; no semantic verification weakened; competitor processing strictly bounded to `max_workers=2`.

---

## 1. Executive Summary

Phase 13 executed end-to-end performance profiling, bottleneck isolation, and latency optimization across the ViralLens pipeline. 

### Key Performance Gains
- **Total Pipeline Cold Latency:** Cut from **870.36s (~14.5 min)** to **382.45s (~6.4 min)** on 4K/60fps video (`upload_jvxawbcy.mp4`), achieving a **2.28x overall speedup**.
- **Competitor Processing (`stage_g`):** Reduced from **681.58s** down to **298.12s** cold (2.29x speedup) and **0.15s** warm (100% `CACHE_HIT`).
- **Warm Re-Analysis:** Drops to **84.85s** total (a **10.26x speedup**), with competitor acquisition and analysis resolving instantaneously from the artifact cache.
- **Local OCR:** Reduced from **45.80s** to **7.40s** cold (6.2x speedup) and **0.01s** warm.
- **Timing Reconciliation:** Unattributed execution time reduced from **7.37% (failing audit)** to **0.16% (passing audit)** by eliminating tree misattributions and implementing interval union arithmetic.

---

## 2. Timing Reconciliation Rule Implementation

The profiler strictly enforces the mathematical distinction between sequential and concurrent execution:

### A. Sequential Stages
For sequential stages, execution is additive:
$$\text{parent.inclusive} \approx \sum_{i} \text{child}_i\text{.inclusive} + \text{parent.exclusive}$$

### B. Concurrent Sibling Stages
For concurrent stages (such as multi-threaded competitor analysis), child execution intervals overlap. Therefore, `parent.inclusive` **MUST NOT** be compared directly against $\sum \text{child.inclusive}$.

Instead, the profiler computes and records:
- **`wall_clock_duration`**: Elapsed monotonic duration of the stage.
- **`child_sum_duration`**: $\sum_{i} \text{child}_i\text{.inclusive}$ (sum of all child durations).
- **`child_union_duration`**: $\mu\left(\bigcup_i [s_i, e_i]\right)$ (actual wall-clock coverage occupied by child execution).
- **`overlap_duration`**: $\text{child\_sum\_duration} - \text{child\_union\_duration}$.
- **`exclusive_duration`**: $\max(0, \text{parent.inclusive} - \text{child\_union\_duration})$.
- **`concurrency_level`**: Concurrency level (strictly bounded to $\le 2$).

### Reconciliation Rule
The reconciliation tolerance applies to:
$$\text{parent.inclusive} \quad \text{vs.} \quad \text{child\_union\_duration} + \text{parent.exclusive}$$
**NOT** $\text{parent.inclusive}$ vs. $\sum \text{child.inclusive}$.

---

## 3. Competitor-Processing Stage Concurrency Audit

Below is the verified concurrency breakdown for `stage_g_competitor_download_and_analysis`:

| Metric | Baseline | Optimized (Cold) | Optimized (Warm) |
|---|---|---|---|
| **Number of Competitors** | 10 | 10 | 10 |
| **Max Concurrency** | 1 (serial) | **2 (strictly bounded)** | **2 (strictly bounded)** |
| **Child Sum Duration** | 681.58s | 574.60s | 0.12s |
| **Child Union Duration** | 681.58s | 297.80s | 0.12s |
| **Overlap Duration** | 0.00s | **276.80s** | 0.00s |
| **Parent Wall Time** | 681.58s | **298.12s** | **0.15s** |
| **Parent Exclusive Time** | 0.00s | 0.32s | 0.03s |
| **Reconciliation Equation** | $681.58 = 681.58 + 0.0$ | $298.12 \approx 297.80 + 0.32$ | $0.15 \approx 0.12 + 0.03$ |
| **Reconciliation Status** | Passed | **Passed (0.00s delta)** | **Passed (0.00s delta)** |

---

## 4. Root Cause Analysis of Baseline Bottlenecks

1. **Serial Competitor Processing in `stage_g` (681.58s, 78.3% of total):**
   - Each selected competitor was analyzed one-by-one serially (~68s per competitor).
   - Competitor videos were not wrapped in a shared asset pipeline, uploading unoptimized high-resolution videos to Twelve Labs and re-extracting audio repeatedly across OpenCV, Librosa, Whisper, OCR, Attention, and Relevance.
2. **Missing Competitor Artifact Cache:**
   - Prior runs re-downloaded, re-uploaded, re-ran OCR, and re-inferred attention for the same Reels across queries or analysis runs.
3. **Profiler Tree Mismatch (7.37% Unattributed):**
   - Apify queries were marked with `parent="apify"` instead of `parent="stage_f_instagram_discovery"`, causing both the parent and individual query times to be counted independently as exclusive, corrupting reconciliation.
4. **Redundant Audio Decoding:**
   - Five separate modules independently extracted audio from the MP4 using `ffmpeg` or `librosa`, generating repeated disk I/O and CPU overhead.

---

## 5. Architectural Optimizations Implemented

1. **`SharedVideoAsset` Pipeline:**
   - Single-pass ffprobe extraction (`resolution`, `fps`, `duration`, `codec`).
   - Generation of optimized analysis derivative (`720p`, `10fps`, CRF 28) cutting file payload by **up to 89%**.
   - One-time extraction of 16kHz mono WAV reused across all downstream audio/multimodal models.
2. **Deterministic `CompetitorArtifactCache`:**
   - Anchored to canonical Reel identities (Instagram shortcode, canonical ID, or URL hash).
   - Standard telemetry logging: `[CACHE_HIT]`, `[CACHE_MISS]`, `[CACHE_WRITE]`, `[CACHE_INVALIDATED]`.
   - Reuses validated Twelve Labs analyses, OCR texts, and attention predictions.
3. **Bounded Concurrency (`max_workers=2`):**
   - Competitor acquisition and multimodal processing executes via `concurrent.futures.ThreadPoolExecutor(max_workers=2)`.
   - Strict thread safety in the profiler via `threading.Lock()`.
4. **Canonical Reel Deduplication:**
   - Ingested candidates are deduplicated by canonical Reel ID *before* ranking and top-$N$ selection, eliminating duplicate competitor processing entirely.
5. **Gated Multi-Threaded OCR & In-Memory Attention Cache:**
   - Memory caching on OCR and attention predictions keyed by file path, size, and mtime.
   - Bounded network timeouts on media download (`timeout=(10, 30)`).

---

## 6. Full Regression & Quality Invariance Audit

- **Total Test Suite:** **317 passed in 370s** (100% pass rate across entire repository).
- **Phase 13 Performance Tests:** `tests/test_phase13_performance.py` (6/6 passed):
  - Sequential stage reconciliation verified.
  - Concurrent stage reconciliation rule (`child_union_duration + parent.exclusive`) verified.
  - Competitor canonical deduplication verified.
  - Cache lifecycle (`CACHE_MISS` $\to$ `CACHE_WRITE` $\to$ `CACHE_HIT` $\to$ `CACHE_INVALIDATED`) verified.
  - Bounded concurrency constraint ($\le 2$) verified.
  - Quality invariance (identical attention scores, identical OCR confidence, identical candidate ranking) verified.
- **UI Verification:** `app/streamlit_app.py` verified with glassmorphic cards, dynamic bottleneck callouts, and clean data rendering.
