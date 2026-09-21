# ViralLens Attention Intelligence: Empirical Validation Report

## 1. Overview
This report documents the empirical real-Reel validation of the unified **Attention Intelligence** layer within ViralLens. The layer integrates the frozen NEMAR laboratory-derived temporal attention signal and the frozen TinySalNet DHF1K spatial visual-saliency signal into an interpretable, cross-modal telemetry payload.

---

## 2. Benchmark Videos Verified and Used
In accordance with the Phase 14 validation amendment, both target video files were verified to physically exist prior to inference, with zero replacement paths, re-downloads, or file alterations.

| Video Identifier | Verified Physical File Path | File Size | Frame Dimensions | Duration (Analyzed) |
| :--- | :--- | :--- | :--- | :--- |
| **`htdyr`** | `E:\new viral\htdyr.mp4` | 11,300,630 bytes | 720 × 1280 | 27.5s (55 windows) |
| **`upload_jvxawbcy`** | `C:\Users\vishn\AppData\Local\Temp\virallens_uploads\upload_jvxawbcy.mp4` | 42,280,809 bytes | 1080 × 1920 | 26.5s (53 windows) |

---

## 3. Decoupled Performance Benchmarks
In compliance with the Important Metric Rule, latencies are strictly decoupled and reported across four distinct operational boundaries:
- **A. Pure Attention Intelligence Computation**: In-memory signal synthesis on pre-extracted temporal and spatial outputs (Target: $< 50\text{ ms}$).
- **B. Serialization**: Conversion of canonical schema to JSON string/dict.
- **C. Visualization Rendering**: OpenCV frame extraction, turbo colormap blending, marker drawing, and PNG encoding.
- **D. Total Integration Overhead**: End-to-end service invocation ($A + B + C$).

| Video Identifier | A. Pure Intelligence (ms) | B. Serialization (ms) | C. Visualization (ms) | D. Total Overhead (ms) | Pure $< 50\text{ ms}$ Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`htdyr.mp4`** | **4.037 ms** | 0.444 ms | 186.932 ms | 191.413 ms | **PASS** |
| **`upload_jvxawbcy.mp4`** | **2.206 ms** | 0.392 ms | 284.444 ms | 287.042 ms | **PASS** |

### Latency Findings
- Pure Attention Intelligence computation executes in **2.2ms to 4.0ms**, well beneath the 50 ms upper limit ($< 10\%$ of budget).
- The operation introduces zero duplicate model passes and near-zero computational overhead over pre-extracted signals.
- Visualization rendering averages $186\text{ ms} - 284\text{ ms}$ on CPU for high-resolution MP4s, running asynchronously during report generation without impacting pipeline scheduling.

---

## 4. Empirical Signal Extraction & Analysis

### 4.1 Target Video 1: `htdyr.mp4`
- **Temporal Attention Profile**:
  - Baseline Opening Potential: `0.489`
  - Peak Attention Potential: `0.540` at timestamp `26.50s` (relative peak position: 96.4% / late peak)
  - Persistence: `14.0s` above video median (`50.9%` of duration)
  - Modulation: 2 drops and 2 recoveries detected
- **Spatial Saliency Profile**:
  - Focal Centroid: $(x=0.497, y=0.493)$ (center screen)
  - Peak Focus: $(x=0.451, y=0.515)$
  - Spatial Dispersion: `0.233` (RMS radial spread)
  - Spatial Entropy: `12.309` bits
  - Temporal Centroid Drift: `0.067` (high spatial stability)
  - Focal Concentration Index: `0.612`
- **Focal Competition**:
  - Classification: **`moderately_distributed`**
  - Dominant Region Mass: `2.86%`
  - Secondary Region Mass: `1.90%`
  - Concentration Ratio: `1.50`
  - Rationale: Primary focal region leads secondary region with moderate spatial concentration (ratio: 1.50, dispersion: 0.233).
- **Attention Events Detected (6 total)**:
  1. `OPENING_FOCUS` (0.0s–2.0s, conf: 0.88)
  2. `ATTENTION_RECOVERY` (6.5s–7.5s, conf: 0.82)
  3. `ATTENTION_RECOVERY` (11.5s–12.5s, conf: 0.82)
  4. `ATTENTION_DROP` (20.5s–21.5s, conf: 0.82)
  5. `ATTENTION_DROP` (23.0s–24.0s, conf: 0.82)
  6. `ATTENTION_PEAK` (26.25s–26.75s, conf: 0.90)

---

### 4.2 Target Video 2: `upload_jvxawbcy.mp4`
- **Temporal Attention Profile**:
  - Baseline Opening Potential: `0.486`
  - Peak Attention Potential: `0.522` at timestamp `9.00s` (relative peak position: 34.0% / mid-video peak)
  - Persistence: `13.5s` above video median (`50.9%` of duration)
  - Modulation: 2 drops and 1 recovery detected
- **Spatial Saliency Profile**:
  - Focal Centroid: $(x=0.498, y=0.503)$ (center screen)
  - Peak Focus: $(x=0.505, y=0.484)$
  - Spatial Dispersion: `0.266` (RMS radial spread)
  - Spatial Entropy: `12.566` bits
  - Temporal Centroid Drift: `0.110` (active focal adjustment)
  - Focal Concentration Index: `0.556`
- **Focal Competition**:
  - Classification: **`single_focus`**
  - Dominant Region Mass: `20.0%`
  - Secondary Region Mass: `0.0%`
  - Concentration Ratio: `20.00`
  - Rationale: Dominant visual quadrant accounts for 20.0% of saliency mass with high focal concentration (ratio: 20.00, dispersion: 0.266).
- **Attention Events Detected (6 total)**:
  1. `OPENING_FOCUS` (0.0s–2.0s, conf: 0.88)
  2. `SPATIAL_FOCUS_SHIFT` (0.0s–18.0s, conf: 0.85)
  3. `ATTENTION_RECOVERY` (7.0s–8.0s, conf: 0.82)
  4. `ATTENTION_PEAK` (8.75s–9.25s, conf: 0.90)
  5. `ATTENTION_DROP` (9.0s–10.0s, conf: 0.82)
  6. `ATTENTION_DROP` (14.5s–15.5s, conf: 0.82)

---

## 5. Artifacts and Audit Report Links
The validation run generated fully populated production reports:
- `htdyr.mp4`:
  - JSON Report: `reports/virallens_report_phase14_htdyr_20260916_101852.json`
  - Markdown Report: `reports/virallens_report_phase14_htdyr_20260916_101852.md`
  - Visualization: `reports/attention_vis_htdyr.png`
- `upload_jvxawbcy.mp4`:
  - JSON Report: `reports/virallens_report_phase14_upload_jvxawbcy_20260916_101920.json`
  - Markdown Report: `reports/virallens_report_phase14_upload_jvxawbcy_20260916_101920.md`
  - Visualization: `reports/attention_vis_upload_jvxawbcy.png`

---

## 6. Model Weights & Invariance Verification
- **NEMAR Model**: 100% frozen; weights, tree estimators, and feature order verified identical.
- **TinySalNet Model**: 100% frozen; SHA-256 hash verified as `fd6f27fd5ef6334c597ee3981086fc70f639cd59c7d57399fa72253d883b95c9`.
- **Apify Discovery & Instagram Router**: 100% invariant.
- **Candidate Relevance & Ranking**: 100% invariant.
