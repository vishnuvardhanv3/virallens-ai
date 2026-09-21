# DHF1K Dataset Audit and Canonical Schema Mapping Report

**Project Root**: `E:\new viral`  
**Dataset Root**: `E:\viral_attention_data\DHF1K`  
**Audit Date**: September 10, 2026  
**Audit Scope**: Read-only dataset structure verification, schema inspection, feature extraction feasibility, split verification, and integration strategy.  
**Compliance Guarantee**: No training was initiated. No production models or code files in `E:\new viral\models\human_attention\` or `E:\v1.0.0\` were modified or overwritten.

---

## 1. Archive Provenance & Exact Archive Locations Found

All three original RAR archives were located on the `E:\` drive and remain completely intact, preserved in their original download locations:

| Archive Name | Exact Location | File Size (Bytes) | Size (GB / MB) | Status |
| :--- | :--- | :--- | :--- | :--- |
| `video.rar` | `E:\video.rar` | 4,032,563,381 | ~3.75 GB | Intact & Unmodified |
| `exportdata_train.rar` | `E:\exportdata_train.rar` | 758,878,229 | ~723.7 MB | Intact & Unmodified |
| `annotation.rar` | `E:\annotation.rar` | 3,888,600,981 | ~3.62 GB | Intact & Unmodified |

---

## 2. Exact Discovered Dataset Structure & Statistics

Extraction destination root: `E:\viral_attention_data\DHF1K`

```
E:\viral_attention_data\DHF1K\
├── video/                               [1,000 files, 3.83 GB]
│   ├── 001.AVI ... 1000.AVI
│
├── annotation/                          [1,219,939 files, 2,801 folders, 3.90 GB]
│   ├── 0001/ ... 0700/
│   │   ├── {last_frame}.png             (RGB 24-bit keyframe, e.g., 0450.png)
│   │   ├── maps/                        (Continuous saliency density maps, 8-bit PNG)
│   │   │   ├── 0001.png ... {N}.png
│   │   ├── fixation/                    (Binary fixation maps, 8-bit PNG)
│   │   │   ├── 0001.png ... {N}.png
│   │   │   └── maps/                    (MATLAB v7 .mat binary fixation matrices)
│   │   │       ├── 0001.mat ... {N}.mat
│
└── exportdata_train/                    [11,900 files, 18 folders, 2.90 GB]
    ├── P01/ ... P17/
    │   ├── Pxx_Trail001.txt ... Pxx_Trail700.txt
```

### Dataset File Counts & Physical Sizes
- **Video Files**: `1,000` files (`4,117,756,894` bytes ~ 3.83 GB)
- **Annotation Files**: `1,219,939` files (`4,184,287,481` bytes ~ 3.90 GB)
  - PNG files: `813,526`
  - MAT files: `406,413`
- **Gaze / Export Files**: `11,900` files (`3,117,276,242` bytes ~ 2.90 GB)
- **Total Files**: `1,232,839` files
- **Total Disk Space**: `11,419,320,617` bytes (~10.64 GB)

---

## 3. Video Properties & Audio Stream Analysis

Audit of representative video clips (`001.AVI`, `002.AVI`, `100.AVI`, `199.AVI`, `499.AVI`, `599.AVI`, `600.AVI`, `699.AVI`, `700.AVI`, `999.AVI`) using OpenCV and FFprobe:

- **Naming Convention**: `{index:03d}.AVI` ranging from `001.AVI` to `1000.AVI`.
- **Container**: AVI (`.AVI`).
- **Video Codec**: `FMP4` (ISO MPEG-4 Part 2 / OpenDivX / XviD).
- **Resolution**: `640 x 360` pixels (standard 16:9 widescreen format, 360p).
- **Frame Rate (FPS)**: Exactly `30.0` FPS across all inspected sequences.
- **Duration**: Ranges from ~5.8s to ~42.1s (mean: ~17.2s; frame counts range from 174 to 1,263 frames).
- **Audio Stream**: **Present**. Inspection via `ffprobe` verified that the AVI files contain both `video` and `audio` streams (`['video', 'audio']`). FFmpeg successfully extracts 16 kHz mono WAV audio without errors.

---

## 4. Annotation Semantics & MAT File Structure

Each annotated sequence folder (`0001` through `0700`) corresponds 1:1 to video files `001.AVI` through `700.AVI`.

1. **`maps/{frame:04d}.png`** (Saliency Map):
   - **Format**: 8-bit single-channel grayscale PNG.
   - **Dimensions**: `360 x 640` pixels.
   - **Range**: Continuous pixel values `[0, 255]`.
   - **Semantics**: Empirical human visual attention density map obtained by accumulating fixation locations across 17 observers and applying Gaussian spatial smoothing (standard deviation matching foveal vision ~1° of visual angle).

2. **`fixation/{frame:04d}.png`** (Fixation Map):
   - **Format**: 8-bit single-channel PNG.
   - **Dimensions**: `360 x 640` pixels.
   - **Range**: Binary values `{0, 255}`.
   - **Semantics**: Consensus fixation points across all 17 participants for that specific video frame (255 = gaze fixation point, 0 = non-fixated background).

3. **`fixation/maps/{frame:04d}.mat`** (MAT File Structure):
   - **Format**: MATLAB 5.0 (v7 format), readable via `scipy.io.loadmat`.
   - **Variables**: Single variable `'I'`.
   - **Shape**: `(360, 640)` ndarray of `uint8`.
   - **Values**: Exact binary matrix `{0, 1}`.
   - **Verification**: Verified via `check_mat_fix.py` that `mat_data['I'] == (fixation_png > 0)`. The MAT files provide the identical ground truth fixation points in standard MATLAB binary matrix format.

4. **Sequence Root Image `{last_frame}.png`**:
   - **Format**: 24-bit 3-channel RGB PNG located directly under `annotation/{seq}/`.
   - **Semantics**: The extracted final RGB frame of each video clip.

---

## 5. Raw Gaze Export Data Schema (`exportdata_train`)

Located at `E:\viral_attention_data\DHF1K\exportdata_train\`:
- **Participants**: Exactly `17` observers labeled `P01` to `P17`.
- **Trials**: Exactly `700` trials per participant (`Pxx_Trail001.txt` to `Pxx_Trail700.txt`), totaling `17 × 700 = 11,900` files.
- **Delimiter**: Tab-separated (`\t`).
- **Sampling Frequency**: **250 Hz** (median timestamp interval $\Delta t = 3,997 \,\mu\text{s} \approx 4.0 \,\text{ms}$). Recorded using an SMI RED 250Hz eye tracker.

### Column Specification

| Column Index | Column Name | Data Type | Physical Meaning / Units | Sample Value |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `Time` | `int64` | System clock timestamp in microseconds | `107977452442` |
| 1 | `Type` | `string` | Recording event type identifier | `SMP` (Sample) |
| 2 | `Trial` | `int` | Sequence / Video trial number (1 to 700) | `1` |
| 3 | `L Dia X [px]` | `float` | Left eye pupil horizontal diameter in pixels | `16.22` |
| 4 | `L Dia Y [px]` | `float` | Left eye pupil vertical diameter in pixels | `16.22` |
| 5 | `L POR X [px]` | `float` | Left Point of Regard (POR) X screen coordinate in px | `1107.82` |
| 6 | `L POR Y [px]` | `float` | Left Point of Regard (POR) Y screen coordinate in px | `417.99` |
| 7 | `L Event Info` | `string` | Oculomotor classifier: `Fixation`, `Saccade`, `Blink`, `-` | `Fixation` |

---

## 6. Official DHF1K Split Verification (Evidence-Based)

From empirical inspection of the file hierarchy and trial numbers:

| Split Name | Video Range | Video Count | Annotation Present | Gaze Export Present | Official DHF1K Benchmark Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | `001.AVI` – `600.AVI` | 600 | **Yes** (`0001`–`0600`) | **Yes** (`Trail001`–`Trail600`) | Training split |
| **Validation** | `601.AVI` – `700.AVI` | 100 | **Yes** (`0601`–`0700`) | **Yes** (`Trail601`–`Trail700`) | Validation split |
| **Test** | `701.AVI` – `1000.AVI` | 300 | **No** (none > 700) | **No** (none > 700) | Held-out benchmark test set |

**Empirical Evidence**:
1. All 1,000 AVI files (`001.AVI` to `1000.AVI`) are present in `video/`.
2. Folders in `annotation/` stop abruptly at `0700`. No sequence folders `0701`–`1000` exist.
3. Every participant directory in `exportdata_train/` has exactly 700 files (`Trail001.txt` to `Trail700.txt`). No trials > 700 exist.
4. This strictly mirrors the official Wang et al. (CVPR 2018) distribution policy: ground truth for videos 701–1000 is intentionally kept private on the automated evaluation server to prevent benchmark gaming.

---

## 7. Canonical ViralLens Attention Data Schema Mapping

Mapping DHF1K observations into the canonical ViralLens schema (`dataset_source`, `video_id`, `participant_id`, `frame_index`, `timestamp`, `gaze_x`, `gaze_y`, `fixation`, `saliency_map`, `attention_target_type`, `split`):

### A. Raw Gaze Observation Record (from `exportdata_train`)
```json
{
  "dataset_source": "DHF1K",
  "video_id": "DHF1K_001",
  "participant_id": "P01",
  "frame_index": 6,
  "timestamp": 0.1999,
  "gaze_x": 948.05,
  "gaze_y": 349.92,
  "fixation": 1,
  "saliency_map": null,
  "attention_target_type": "raw_gaze_point",
  "split": "train"
}
```

### B. Frame-Level Consensus Annotation Record (from `annotation`)
```json
{
  "dataset_source": "DHF1K",
  "video_id": "DHF1K_001",
  "participant_id": "aggregate_17_observers",
  "frame_index": 1,
  "timestamp": 0.0,
  "gaze_x": null,
  "gaze_y": null,
  "fixation": "E:\viral_attention_data\DHF1K\annotation\0001\fixation\0001.png",
  "saliency_map": "E:\viral_attention_data\DHF1K\annotation\0001\maps\0001.png",
  "attention_target_type": "spatial_saliency_density",
  "split": "train"
}
```

---

## 8. 14 Production Feature Compatibility Verification

We executed `VideoFeatureExtractor(window_sec=0.5)` on sample DHF1K videos. All 14 features were computed successfully and yielded valid numerical distributions:

| Feature Name | Status | Generation Mechanism | Sample Range on DHF1K |
| :--- | :--- | :--- | :--- |
| `motion_magnitude` | Compatible | Optical flow / frame diff magnitude across 0.5s window | 0.0255 – 0.0587 |
| `motion_std` | Compatible | Temporal standard deviation of motion magnitude | 0.0036 – 0.0125 |
| `scene_cut_rate` | Compatible | HSV histogram correlation drop below threshold per window | 0.0 – 2.0 cuts/s |
| `time_since_cut` | Compatible | Elapsed seconds since preceding detected cut | 0.25 – 42.0s |
| `brightness_mean` | Compatible | Mean luminance Y-channel in 0.5s window | 0.3880 – 0.4121 |
| `brightness_std` | Compatible | Standard deviation of luminance in 0.5s window | 0.0006 – 0.0037 |
| `contrast` | Compatible | Standard deviation of pixel intensities across frames | 0.2147 – 0.2294 |
| `visual_complexity` | Compatible | Variance of Laplacian edges in 0.5s window | 2.8480 – 3.3611 |
| `face_presence` | Compatible | Haar cascade face detector ratio across window frames | 0.0 – 0.25 |
| `text_presence` | Compatible | Sobel horizontal gradient energy in caption band | 0.2205 – 0.3196 |
| `audio_rms` | Compatible | FFmpeg audio extraction -> 16kHz mono RMS | 0.1184 – 0.1577 |
| `audio_silence_ratio`| Compatible | Fraction of audio window samples below 0.015 | 0.0705 – 0.1361 |
| `audio_spectral_centroid`| Compatible | FFT spectral center of mass in audio window | 1,213 – 1,727 Hz |
| `audio_speech_presence`| Compatible | Spectral energy in human vocal band (300-3400 Hz) | 0.5504 – 0.7380 |

---

## 9. Comparison: DHF1K vs. NEMAR Production Interface

| Dimension | NEMAR (Current Production Model) | DHF1K (Audited Dataset) |
| :--- | :--- | :--- |
| **Video Corpus** | 5 educational lecture videos (~23.5 min total) | 1,000 real-world clips (web, action, nature, sports) |
| **Video Diversity** | Homogeneous (single presenter, slides, classroom) | Extremely heterogeneous (diverse cameras, lighting, motion) |
| **Eye-Tracking Frequency** | 128 Hz | 250 Hz |
| **Participants** | ~30 participants | 17 observers per sequence |
| **Cognitive States** | Attentive (`ses-01`) vs Distracted (`ses-02`) | Free-viewing attentive (no distracted cohort) |
| **Spatial Saliency Maps** | Derived synthetically via Gaussian kernel | Precomputed ground truth maps (`maps/*.png`) |
| **Sample Count (0.5s Windows)**| ~2,800 windows | ~12,000 annotated windows (Videos 1-700) |

---

## 10. Training Targets Identification

DHF1K provides three distinct levels of attention supervisory signals:
1. **Spatial Saliency Target**: Ground truth 2D continuous density maps (`maps/*.png`). Suitable for training deep spatial visual attention networks (e.g. UNet, ViT, Video Transformer).
2. **Binary Fixation Consensus Target**: Point maps (`fixation/*.png` and `fixation/maps/*.mat`). Suitable for NSS (Normalized Scanpath Saliency) loss and Information Gain.
3. **Temporal Attention Concentration Target**: Aggregating `exportdata_train` gaze coordinates per 0.5s window computes an exact analog of NEMAR's `human_visual_attention` score:
   $$\text{Attention Concentration} = 1.0 - \min(1.0, 2.0 \times \text{Gaze Dispersion})$$
   with pupil dilation (`L Dia X/Y`) providing physiological arousal telemetry.

---

## 11. Integration Strategy Assessment & Recommendation

### Options Evaluated:
- **Option A (Train independently)**: Train a dedicated DHF1K attention model on the 600 training and 100 validation videos.
- **Option B (Combine with NEMAR)**: Pool NEMAR and DHF1K windows directly.
- **Option C (Use only for pretraining)**: Pretrain a visual attention feature extractor on DHF1K, fine-tune on NEMAR/Reels.
- **Option D (Use for validation / cross-dataset testing)**: Evaluate the existing NEMAR production model zero-shot on DHF1K.

### Evidence-Based Phased Recommendation: `D -> A -> C/B`
1. **Phase 1 (Immediate - Cross-Dataset Validation / Option D)**:
   Run zero-shot inference with the existing production model on DHF1K's 100 validation videos (`601.AVI`–`700.AVI`) to benchmark generalizability and establish a baseline MAE/RMSE on real-world video without modifying any production files.
2. **Phase 2 (Independent Training / Option A)**:
   Train an independent regressor (`model_dhf1k.pkl`) on the 600 DHF1K training videos into a separate directory (`models/human_attention_dhf1k/`). This ensures zero risk to the existing `models/human_attention/model.pkl`.
3. **Phase 3 (Ensemble / Option C)**:
   Combine NEMAR (specialized in long-form cognitive fatigue and attentive vs distracted state) with DHF1K (specialized in dynamic real-world visual editing and multi-observer gaze concentration).

---

## 12. Safety and Integrity Verification

- `E:\new viral\models\human_attention\model.pkl`: **Untouched (SHA/timestamps unchanged)**.
- `E:\v1.0.0`: **Untouched**.
- `E:\viral_attention_data\DHF1K` source files: **Untouched (no files modified, moved, or deleted)**.
- `E:\ViralLens_AI_NEW` / `E:\ViralLens_AI_FIXED`: **Not accessed**.
- **Model Training**: **Not started**.
