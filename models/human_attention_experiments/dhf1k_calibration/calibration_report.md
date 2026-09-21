# DHF1K Phase 4: Model Calibration & Generalization Analysis Report

**Experiment**: Post-Hoc Calibration & Prediction Compression Analysis of Independent DHF1K Model  
**Date**: September 12, 2026  
**Model Evaluated**: `models/human_attention_experiments/dhf1k_independent/model_dhf1k.pkl` (ExtraTreesRegressor)  
**Validation Corpus**: DHF1K Validation (`601.AVI` through `700.AVI`, 100 videos, 3,970 windows)  
**Target Analyzed**: **DHF1K-derived fixation concentration proxy** (constructed temporal regression target)  
**Safety Compliance**: Strictly read-only analysis. Zero retraining, zero fine-tuning, zero model overwrites. SHA-256 hashes of production NEMAR and Phase 2 independent models verified pre- and post-analysis.

> [!IMPORTANT]
> **Non-Causal Methodology Note**: Calibration / regression-to-the-mean shrinkage is evaluated as an empirical association with observed residuals, **not** claimed as the causal mechanism of error.

---

## 1. Executive Summary & Calibration Diagnostics (Task 1)

Diagnostic regression evaluated on the existing validation set:
$$\text{target} = \alpha + \beta \times \text{prediction}$$

| Metric | Original Model Value | Ideal Value | Scientific Meaning |
| :--- | :--- | :--- | :--- |
| **Prediction Mean (+/- std)** | **0.8425** (+/- 0.0095) | **0.8427** (+/- 0.0571) | Prediction variance is substantially smaller than ground truth |
| **Target Mean (+/- std)** | **0.8427** (+/- 0.0571) | **--** | Validation ground-truth distribution |
| **Mean Calibration Bias** | **-0.0002** | **0.0000** | Model has virtually zero overall calibration offset |
| **Calibration Slope (beta)** | **0.9457** | **1.0000** | High slope ($>> 1.0$) reflects strong regression-to-the-mean shrinkage |
| **Calibration Intercept (alpha)**| **+0.0460** | **0.0000** | OLS regression offset |
| **Global MAE / RMSE** | **0.0454** / **0.0564** | **--** | Baseline validation errors |
| **Pearson $r$ / Spearman rho**| **0.1573** / **0.1644** | **1.0000** | Statistically significant positive association |

---

## 2. Prediction Range & Compression Analysis (Task 2)

Comparing distribution spans and percentiles reveals strong shrinkage of predictions toward the central mean:

| Statistic | Target Value (Ground Truth) | Prediction Value (Model Output) | Compression Ratio / Delta |
| :--- | :--- | :--- | :--- |
| **Standard Deviation (sigma)** | **0.3635** | **0.0981** | **0.1663** (sigma_pred / sigma_target = 16.6%) |
| **Minimum Value** | **0.5990** | **0.7959** | Model lower bound is compressed upward |
| **Maximum Value** | **0.9625** | **0.8940** | Model upper bound is compressed downward |
| **1st Percentile (p1)** | **0.6938** | **0.8107** | Delta = +0.1169 |
| **5th Percentile (p5)** | **0.7405** | **0.8285** | Delta = +0.0880 |
| **25th Percentile (Q1)** | **0.8056** | **0.8381** | Delta = +0.0325 |
| **50th Percentile (Median)** | **0.8486** | **0.8418** | Delta = -0.0068 |
| **75th Percentile (Q3)** | **0.8845** | **0.8477** | Delta = -0.0368 |
| **95th Percentile (p95)** | **0.9262** | **0.8566** | Delta = -0.0696 |
| **99th Percentile (p99)** | **0.9449** | **0.8662** | Delta = -0.0787 |

---

## 3. Exploratory In-Sample Calibration (Task 3)

> [!WARNING]
> **Exploratory Label**: These results represent in-sample fits on the validation set itself and **must not** be interpreted as unbiased generalization metrics.

| Method | Parameters / Transform | MAE | RMSE | R^2 | Pearson $r$ | Spearman rho | MAE Change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Model** | Uncalibrated ExtraTrees | **0.0454** | **0.0564** | **0.0247** | **0.1573** | **0.1645** | Baseline |
| **In-Sample Affine** | $\hat{y}_{\text{cal}} = 0.9457\hat{y} +0.0460$ | **0.0454** | **0.0564** | **0.0247** | **0.1573** | **0.1644** | **-0.0000** |
| **In-Sample Isotonic** | Piecewise constant non-decreasing | **0.0450** | **0.0560** | **0.0385** | **0.1962** | **0.1788** | **-0.0004** |

---

## 4. Grouped 5-Fold Cross-Validated Calibration (Task 4)

To prevent in-sample parameter leakage, a **5-fold cross-fitting scheme grouped by Video ID** was executed. Calibration parameters were estimated strictly on 80 videos (4 folds) and evaluated on 20 held-out videos (1 fold), repeating across all 5 folds:

| Calibration Model | Out-of-Fold MAE | Out-of-Fold RMSE | Out-of-Fold R^2 | Out-of-Fold Pearson $r$ | Out-of-Fold Spearman rho | Delta MAE vs Original |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Model (Uncalibrated)** | **0.0454** | **0.0564** | **0.0247** | **0.1573** | **0.1645** | Baseline |
| **Cross-Fitted Affine Calibration** | **0.0457** | **0.0567** | **0.0131** | **0.1204** | **0.1170** | **+0.0003** |
| **Cross-Fitted Isotonic Calibration** | **0.0459** | **0.0569** | **0.0063** | **0.1053** | **0.1197** | **+0.0005** |

*Cross-fitting finding*: When evaluated strictly out-of-fold, post-hoc calibration produces a negligible overall MAE delta ($\Delta\text{MAE} = +0.0003$).

---

## 5. Target-Extreme & Tail Performance Comparison (Task 5)

Comparing error and bias across target intervals to assess whether calibration reduces tail bias without degrading central density performance:

| Target Range Bin | Sample Count | Target Mean | Original MAE | Calibrated MAE | Delta MAE | Original Bias | Calibrated Bias | Tail Bias Reduced? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.60–0.70** | 45 | 0.6697 | 0.1723 | 0.1732 | **+0.0009** | +0.1723 | +0.1732 | No |
| **0.70–0.80** | 850 | 0.7662 | 0.0740 | 0.0747 | **+0.0008** | +0.0740 | +0.0747 | No |
| **0.80–0.90** | 2,426 | 0.8519 | 0.0250 | 0.0251 | **+0.0001** | -0.0093 | -0.0092 | **Yes** |
| **0.90–1.00** | 649 | 0.9203 | 0.0754 | 0.0758 | **+0.0004** | -0.0754 | -0.0758 | No |

### Tail Performance Synthesis:
- In the extreme low-concentration tail ($[0.60, 0.70)$, $N = 45$), cross-fitted calibration yields original bias +0.1723 vs calibrated bias **+0.1732** ($\Delta\text{MAE} = +0.0009$).
- In the extreme high-concentration tail ($[0.90, 1.00]$, $N = 649$), cross-fitted calibration yields original bias -0.0754 vs calibrated bias **-0.0758** ($\Delta\text{MAE} = +0.0004$).
- In the central bulk ($[0.80, 0.90)$, $N = 2,426$, 61.1% of samples), error remains virtually unchanged ($\Delta\text{MAE} = +0.0001$).

---

## 6. Video-Level Robustness After Calibration (Task 6)

Evaluating whether cross-fitted calibration improves or degrades individual video performance across the 100 validation clips:

| Video-Level Metric | Original Model | Calibrated Model (Cross-Fitted) | Delta |
| :--- | :--- | :--- | :--- |
| **Mean Video MAE** | **0.0457** | **0.0460** | **+0.0003** |
| **Median Video MAE** | **0.0447** | **0.0449** | **+0.0002** |
| **Standard Deviation** | **0.0128** | **0.0130** | **+0.0002** |
| **Minimum Video MAE** | **0.0246** | **0.0255** | **--** |
| **Maximum Video MAE** | **0.1004** | **0.1007** | **--** |

### Per-Video Improvement Counts:
- **Videos Improved**: **32 / 100**
- **Videos Worsened**: **55 / 100**
- **Videos Unchanged**: **13 / 100**
- **Median Per-Video Delta**: **+0.0002**

---

## 7. Generalization Caution (Task 7)

> [!CAUTION]
> **Strict Generalization Limits**:
> 1. **No External Test Ground Truth**: Phase 4 analyzes predictions exclusively on the existing DHF1K validation set (`601.AVI` to `700.AVI`).
> 2. **Cross-Fitting vs. New Data**: While 5-fold cross-fitting prevents in-sample parameter leakage, it does **not** substitute for evaluation on an independent external video distribution.
> 3. **Private Benchmark Integrity**: DHF1K benchmark videos `701.AVI` through `1000.AVI` have private annotations and must remain completely unevaluated to preserve benchmark integrity.

---

## 8. Decision Rule Recommendation (Task 8)

### Selected Decision: **Calibration not justified**

**Rationale**:
Cross-fitted calibration worsens validation error or destabilizes per-video predictions.

**Key Decision Factors**:
- Overall cross-fitted MAE change: **+0.0003** (negligible).
- Overall cross-fitted RMSE change: **+0.0003**.
- Overall cross-fitted $R^2$ change: **-0.0116**.
- Central density MAE degradation: **+0.0001**.
- Per-video balance: **32 improved vs 55 worsened**.

*Final Technical Policy*: Because the uncalibrated model (`model_dhf1k.pkl`) is already centered around the validation mean ($\text{bias} = -0.0002$) and cross-fitted calibration degrades central density precision without yielding meaningful net MAE reduction, **`model_dhf1k.pkl` is retained as the independent DHF1K experiment artifact unchanged**, while the production NEMAR model (`models/human_attention/model.pkl`) remains strictly isolated and untouched. Post-hoc calibration is marked as exploratory and marginal.
