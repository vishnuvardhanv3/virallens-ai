# DHF1K Phase 1: Zero-Shot Cross-Dataset Validation Report

**Experiment**: Zero-Shot Cross-Dataset Validation of Production NEMAR Human Attention Model on DHF1K  
**Date**: September 20, 2026  
**Evaluated Split**: DHF1K Validation (`601.AVI` through `700.AVI`, 100 videos)  
**Production Model Evaluated**: `E:\new viral\models\human_attention\model.pkl` (Trained on NEMAR BBBD Experiment 1)  
**Safety Compliance**: Zero modifications made to production models, `v1.0.0`, or DHF1K source datasets. All artifacts stored in `models/human_attention_experiments/dhf1k_zero_shot/`.

---

## 1. Executive Summary & Core Results

> [!WARNING]
> **Transfer Status**: **Weak / Not practically established.**  
> Zero-shot cross-dataset evaluation demonstrates that while inference executes successfully and yields measurable positive correlations ($r \approx 0.0555$, $\rho \approx 0.0552$), the association is extremely weak and accounts for less than 0.31% of the target variance. Phase 1 confirms substantial domain and target mismatch between seated lecture stimuli and dynamic real-world video. It does **not** establish useful predictive transfer.

| Metric | Target B (Fixation-Derived Concentration) | Target A1 (Saliency-Density Peak) | Target A2 (Saliency-Density Entropy) |
| :--- | :--- | :--- | :--- |
| **Evaluated Samples (0.5s Windows)** | **3,970** | **3,970** | **3,970** |
| **Evaluated Videos** | **100 / 100 (100%)** | **100 / 100 (100%)** | **100 / 100 (100%)** |
| **MAE** | **0.3901** | **0.5474** | **0.2163** |
| **RMSE** | **0.3948** | **0.5479** | **0.2187** |
| **Pearson Correlation ($r$)** | **0.0555** (p=4.69e-04) | **nan** (constant GT) | **0.0487** (p=2.16e-03) |
| **Spearman Correlation (rho)** | **0.0552** (p=4.98e-04) | **nan** (constant GT) | **0.0427** (p=7.07e-03) |
| **R^2 Score** | **-46.7600** | **0.0000** | **-94.3370** |

---

## 2. Target Construction Methodology

To evaluate laboratory-derived human visual-attention signals rigorously without conflating dataset semantics, we evaluated two distinct target representations independently.

> [!IMPORTANT]
> DHF1K targets are NOT assumed or claimed to be mathematically equivalent to the NEMAR attention target. Comparisons between them represent a cross-dataset target-definition comparison reflecting different experimental protocols.

### Target B: Fixation-Derived Attention Concentration
- **Source**: `annotation/XXXX/fixation/*.png` (consensus fixation points across 17 observers).
- **Mathematical Definition**:
  For all consensus fixation coordinates $(x, y)$ in normalized $[0, 1]$ screen space within a 0.5s window:
  $$c_x = \frac{1}{N}\sum x, \quad c_y = \frac{1}{N}\sum y$$
  $$\text{dispersion} = \text{std}\left(\sqrt{(x - c_x)^2 + (y - c_y)^2}\right)$$
  $$\text{Target B} = \max(0.0, 1.0 - \min(1.0, 2.0 \times \text{dispersion}))$$
- **Characteristics**: Ground truth mean = **0.8427** (std = 0.0571). Quantifies spatial tightness of multi-observer consensus gaze.

### Target A: Saliency-Density Concentration
- **Target A1 (Peak Saliency Concentration)**: $\text{Peak} = \max_{(x,y)} M(x, y) / 255.0$. Mean = **1.0000** (std = 0.0000; effectively constant across frames, rendering correlation undefined).
- **Target A2 (Spatial Entropy Concentration)**: $\text{Concentration} = 1.0 - \left(\frac{H(M)}{\log_2(360 \times 640)}\right)$ on continuous density maps $M$. Mean = **0.2363**.

---

## 3. Distribution & Calibration Comparison

| Distribution Statistic | Model Predictions | Ground Truth Target B (Fixation) | Ground Truth Target A1 (Peak) | Ground Truth Target A2 (Entropy) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | **0.4526** | **0.8427** | **1.0000** | **0.2363** |
| **Std Dev** | **0.0239** | **0.0571** | **0.0000** | **0.0224** |
| **Median** | **0.4561** | **0.8486** | **1.0000** | **0.2347** |
| **Min - Max** | **0.3645 - 0.5259** | **0.5990 - 0.9625** | **1.0000 - 1.0000** | **0.1613 - 0.3108** |

**Cross-Dataset Calibration Analysis & Predictive Agreement**:
- **Severe Negative $R^2$**: $R^2 = -46.76$ for Target B and $R^2 = -94.34$ for Target A2 indicate very poor predictive agreement. The model performs significantly worse than predicting a constant mean ground-truth value.
- **Scale and Calibration Mismatch**: The NEMAR production model was trained on classroom lectures where single-observer gaze dispersion against broad whiteboard slides produced attention potential centered at mean $\approx 0.45$. In contrast, on real-world dynamic video clips in DHF1K, 17 observers fixate tightly on prominent focal subjects (actors, vehicles, moving objects), yielding higher consensus fixation concentration (mean $= 0.8427$).
- **Weak Association**: While the linear and rank correlations with Target B are positive ($r \approx 0.0555$, $\rho \approx 0.0552$), the association is too weak to provide practical predictive utility across datasets without retraining.

---

## 4. Feature Domain Shift Analysis

Comparing feature distributions between the NEMAR training set and DHF1K validation videos reveals substantial domain discrepancy:

| Feature | NEMAR Training Mean (+/- std) | DHF1K Validation Mean (+/- std) | Z-Score Shift | Scientific Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `brightness_mean` | 0.7436 (+/- 0.1872) | 0.4486 (+/- 0.1367) | **-1.58** | Severe domain shift (different visual/acoustic regime) |
| `audio_silence_ratio` | 0.1817 (+/- 0.1237) | 0.3311 (+/- 0.3011) | **+1.21** | Severe domain shift (different visual/acoustic regime) |
| `time_since_cut` | 123.9291 (+/- 120.2728) | 11.0950 (+/- 7.9861) | **-0.94** | Moderate domain shift |
| `contrast` | 0.1664 (+/- 0.0864) | 0.2430 (+/- 0.0545) | **+0.89** | Moderate domain shift |
| `motion_magnitude` | 0.0100 (+/- 0.0188) | 0.0259 (+/- 0.0238) | **+0.85** | Moderate domain shift |
| `text_presence` | 0.0908 (+/- 0.0773) | 0.1488 (+/- 0.0780) | **+0.75** | Moderate domain shift |
| `audio_rms` | 0.1285 (+/- 0.0477) | 0.1030 (+/- 0.0856) | **-0.53** | Moderate domain shift |
| `audio_spectral_centroid` | 2067.4686 (+/- 727.6832) | 1791.0934 (+/- 811.5252) | **-0.38** | Well-aligned distribution |
| `audio_speech_presence` | 0.5035 (+/- 0.1243) | 0.5451 (+/- 0.2042) | **+0.33** | Well-aligned distribution |
| `brightness_std` | 0.0113 (+/- 0.0314) | 0.0062 (+/- 0.0112) | **-0.16** | Well-aligned distribution |
| `visual_complexity` | 1.4219 (+/- 1.8506) | 1.6310 (+/- 1.1384) | **+0.11** | Well-aligned distribution |
| `scene_cut_rate` | 0.0267 (+/- 0.2918) | 0.0111 (+/- 0.2196) | **-0.05** | Well-aligned distribution |
| `motion_std` | 0.0101 (+/- 0.0246) | 0.0110 (+/- 0.0130) | **+0.04** | Well-aligned distribution |
| `face_presence` | 0.0260 (+/- 0.1550) | 0.0313 (+/- 0.1516) | **+0.03** | Well-aligned distribution |

---

## 5. Critical Scientific Limitations & Evidence-Based Recommendations

### Why $p < 0.001$ Does NOT Establish Practical Predictive Value for $r \approx 0.05$:
1. **Sample Size Artifact**: In statistical inference, the standard error of a correlation coefficient scales inversely with sample size:
   $$\text{SE}_r \approx \frac{1}{\sqrt{N - 1}} = \frac{1}{\sqrt{3970 - 1}} \approx 0.0159$$
   With $N = 3,970$ observations, the statistical test has tremendous power ($t \approx 3.50$), rejecting the null hypothesis $H_0: r = 0$ with $p = 4.69 \times 10^-4$.
2. **Negligible Effect Size**: The coefficient of determination is:
   $$r^2 = (0.0555)^2 \approx 0.0031 \quad (0.31\% \text{ of variance explained})$$
   Over 99.69% of the variance in DHF1K fixation concentration is unaccounted for by the zero-shot model predictions. Statistical significance establishes that the association is unlikely to be pure random sampling noise, but it does **not** imply practical predictive utility.
3. **Severe Predictive Invalidation ($R^2 < 0$)**: Target B $R^2 = -46.76$ and Target A2 $R^2 = -94.34$ prove that the zero-shot regressor cannot be deployed as an off-the-shelf predictor for real-world dynamic video gaze.
4. **Target A1 Degeneracy**: The peak saliency density is saturated at 1.0 across all frames, rendering correlation mathematically undefined.

### Summary of What Phase 1 Demonstrates:
- **Successful zero-shot execution**: Pipeline, feature extraction, and inference run reproducibly without failure.
- **Measurable but extremely weak cross-dataset association**: $r = 0.0555$, $\rho = 0.0552$.
- **Substantial domain/target mismatch**: Structural divergence between laboratory educational slides and dynamic web video.
- **No useful generalization established**: Zero-shot transfer does not work at an acceptable level.

### Clear Evidence-Based Recommendations:
1. **Proceed to Phase 2 (Independent DHF1K Training First)**:
   Train a dedicated regressor specifically on DHF1K's 600 training videos (`models/human_attention_dhf1k/model_dhf1k.pkl`). This will allow model weights to learn the specific motion, cut frequency, and visual complexity dynamics of real-world video.
2. **Explicit Restriction on Model Ensembling**:
   **Do not combine NEMAR and DHF1K models yet.** Cross-dataset pooling or ensembling before establishing independent DHF1K model baseline performance and understanding domain-specific weighting would degrade production model reliability.
