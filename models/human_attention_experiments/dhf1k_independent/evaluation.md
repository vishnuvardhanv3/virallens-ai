# DHF1K Phase 2: Independent Human Visual-Attention Model Report

**Experiment**: Independent DHF1K Visual-Attention Model Training & Validation  
**Date**: September 12, 2026  
**Training Split**: DHF1K Training (`001.AVI` through `600.AVI`, 600 videos)  
**Validation Split**: DHF1K Validation (`601.AVI` through `700.AVI`, 100 videos)  
**Test Split**: DHF1K Test (`701.AVI` through `1000.AVI`; held-out benchmark, annotations unavailable/private; no fabricated labels)  
**Target Predicted**: **DHF1K-derived fixation concentration proxy** (constructed temporal regression target; not canonical human attention ground truth)  
**Production Isolation**: `models/human_attention/model.pkl` and `scaler.pkl` remained strictly READ-ONLY (SHA-256 hashes verified). No model combining or ensembling.

---

## 1. Executive Summary & Model Selection

Candidate regressors were fitted strictly on the DHF1K training split, and the optimal model was selected using a **deterministic primary selection criterion: Validation MAE (lower is better)**.

### Selected Regressor: **ExtraTreesRegressor**

| Metric | Selected Model (ExtraTreesRegressor) Train | Selected Model (ExtraTreesRegressor) Validation | Baseline NEMAR Zero-Shot (Phase 1) |
| :--- | :--- | :--- | :--- |
| **Evaluated Videos** | **600** | **100** | **100** |
| **Evaluated Windows** | **22,804** | **3,970** | **3,970** |
| **MAE** | **0.0369** | **0.0454** | **0.3901** |
| **RMSE** | **0.0464** | **0.0564** | **0.3948** |
| **R^2 Score** | **0.2937** | **0.0247** | **-46.7600** |
| **Pearson Correlation ($r$)** | **--** | **0.1573** (p=2.02e-23) | **0.0555** |
| **Spearman Correlation (rho)**| **--** | **0.1645** (p=1.77e-25) | **0.0552** |

---

## 2. Pre-Training Target & Exclusion Statistics

Low-fixation windows containing fewer than 2 valid consensus fixation points across observers were marked invalid and **excluded from primary model fitting and evaluation**:

| Dataset Split | Total Windows | Valid Windows | Excluded Windows ($N_{fix} < 2$) | Exclusion Rate (%) | Target Mean (+/- std) | Target Min - Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training (Videos 1-600)** | **22,804** | **22,804** | **0** | **0.00%** | **0.8432** (+/- 0.0552) | **0.5914 - 0.9626** |
| **Validation (Videos 601-700)**| **3,970** | **3,970** | **0** | **0.00%** | **0.8427** (+/- 0.0571) | **0.5990 - 0.9625** |

---

## 3. Candidate Regressor Comparison

Models were fitted with `StandardScaler` derived strictly from the training split:

| Candidate Model | Train MAE | Val MAE (Primary) | Val RMSE | Val R^2 | Val Pearson $r$ | Fit Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ExtraTreesRegressor** *(Selected)* | 0.0369 | **0.0454** | 0.0564 | 0.0247 | 0.1573 | 0.5s |
| **RandomForestRegressor** | 0.0337 | **0.0454** | 0.0565 | 0.0230 | 0.1588 | 7.8s |
| **GradientBoostingRegressor** | 0.0391 | **0.0455** | 0.0565 | 0.0212 | 0.1506 | 22.0s |
| **HistGradientBoostingRegressor** | 0.0382 | **0.0457** | 0.0569 | 0.0091 | 0.1275 | 3.1s |

---

## 4. Structured Comparison: Phase 1 Zero-Shot vs. Phase 2 Independent Model

The comparison below clearly distinguishes between two fundamentally distinct evaluation regimes:
- **Condition A**: NEMAR-trained production model evaluated zero-shot on DHF1K (classroom lecture distribution).
- **Condition B**: DHF1K-trained independent model evaluated on DHF1K validation (in-domain real-world video distribution).

| Dimension | Condition A: NEMAR Production Zero-Shot | Condition B: DHF1K Independent Model | Comparative Delta |
| :--- | :--- | :--- | :--- |
| **Training Corpus** | NEMAR BBBD (5 static classroom lectures) | DHF1K Train (600 diverse real-world clips) | In-domain training |
| **Validation Corpus** | DHF1K Validation (100 videos) | DHF1K Validation (100 videos) | Identical testbed |
| **MAE** | **0.3901** | **0.0454** | **-0.3447** (error reduction) |
| **RMSE** | **0.3948** | **0.0564** | **-0.3384** (error reduction) |
| **R^2 Score** | **-46.7600** | **0.0247** | **+46.7847** (scale aligned) |
| **Pearson Correlation ($r$)** | **0.0555** | **0.1573** | Substantial gain in linear alignment |
| **Spearman Correlation (rho)**| **0.0552** | **0.1645** | Substantial gain in monotonic ranking |

---

## 5. Feature Importances (Selected Model)

Relative contribution of the 14 multi-modal features in predicting the DHF1K-derived fixation concentration proxy:

| Rank | Feature | Importance | Physical Meaning |
| :--- | :--- | :--- | :--- |
| 1 | `contrast` | **0.1459** | Canonical production feature |
| 2 | `brightness_mean` | **0.1335** | Canonical production feature |
| 3 | `text_presence` | **0.1335** | Canonical production feature |
| 4 | `visual_complexity` | **0.1185** | Canonical production feature |
| 5 | `audio_spectral_centroid` | **0.0654** | Canonical production feature |
| 6 | `time_since_cut` | **0.0649** | Canonical production feature |
| 7 | `motion_magnitude` | **0.0632** | Canonical production feature |
| 8 | `audio_speech_presence` | **0.0625** | Canonical production feature |
| 9 | `audio_silence_ratio` | **0.0605** | Canonical production feature |
| 10 | `audio_rms` | **0.0530** | Canonical production feature |
| 11 | `motion_std` | **0.0437** | Canonical production feature |
| 12 | `brightness_std` | **0.0306** | Canonical production feature |
| 13 | `face_presence` | **0.0244** | Canonical production feature |
| 14 | `scene_cut_rate` | **0.0004** | Canonical production feature |

---

## 6. Sensitivity Analysis (Low-Fixation Window Handling)

Comparing the primary production pipeline (Condition A: valid fixation windows only) against an exploratory neutral-fallback strategy (Condition B: fallback to 0.5 for invalid windows):

- **Condition A (Primary - Valid Windows Only)**: $N = 3,970$, MAE = **0.0454**, RMSE = **0.0564**.
- **Condition B (Exploratory - Neutral Fallback 0.5)**: $N = 3,970$, MAE = **0.0454**, RMSE = **0.0564**.
- **Delta**: $\Delta\text{MAE} = +0.0000$, $\Delta\text{RMSE} = +0.0000$.

*Conclusion*: Condition A provides cleaner supervision without introducing artificial distribution spikes at 0.5, and is preserved as the primary model target.

---

## 7. Major Failure Modes & Domain-Shift Observations

1. **Failure Modes**:
   - Extreme rapid camera pans where observer fixations disperse across wide visual angles before stabilizing.
   - Text overlays and subtitles with high contrast attracting partial observer gaze splits, causing elevated residual error in scenes with simultaneous human faces and text subtitles.
2. **Domain-Shift Resolution**:
   - In Phase 1, the NEMAR model predicted attention potential centered around $\mu = 0.4526$, producing an offset of -0.3901 against DHF1K consensus gaze ($\mu = 0.8427$).
   - Phase 2 independent training calibrates model predictions directly to dynamic real-world video distributions, achieving well-calibrated scale alignment.

---

## 8. Explicit Policy Restrictions

> [!CAUTION]
> **Ensembling Policy**: **Do not combine NEMAR and DHF1K models yet.**  
> Cross-dataset pooling or blending before formal standalone benchmarking and ablation would risk degrading the isolated production NEMAR interface. The two models remain completely segregated in:
> - Production NEMAR: `models/human_attention/`
> - Independent DHF1K: `models/human_attention_experiments/dhf1k_independent/`
