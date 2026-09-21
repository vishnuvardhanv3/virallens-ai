# DHF1K Phase 5: Feature Representation Upgrade Report

**Experiment**: Feature Representation Upgrade & Model Capacity Analysis for Human Visual-Attention Signal  
**Date**: September 12, 2026  
**Dataset Source**: DHF1K (`E:\\viral_attention_data\\DHF1K`)  
**Target Variable**: **DHF1K-derived fixation concentration proxy** (constructed temporal regression target)  
**Safety & Isolation**: Production NEMAR model (`models/human_attention/`) and Phase 2 independent model (`models/human_attention_experiments/dhf1k_independent/`) verified unmodified.

---

## 1. Executive Summary & Required Comparison

| Representation / Model | Feature Count | Val MAE | Val RMSE | Val R² | Pearson $r$ | Spearman $\rho$ | Absolute $\Delta\text{MAE}$ | Relative $\Delta\text{MAE}$ (%) | $\Delta R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **exp_a_baseline** (ExtraTreesRegressor) | 14 | **0.0454** | 0.0564 | +0.0247 | 0.1573 | 0.1645 | +0.0000 | +0.00% | +0.0000 |
| **exp_b_temporal_dynamics** (ExtraTreesRegressor) | 27 | **0.0454** | 0.0565 | +0.0225 | 0.1504 | 0.1512 | +0.0000 | +0.00% | -0.0022 |
| **exp_c1_causal_context** (ExtraTreesRegressor) | 28 | **0.0454** | 0.0564 | +0.0241 | 0.1572 | 0.1440 | +0.0000 | +0.00% | -0.0006 |
| **exp_c2_bidirectional_context** (ExtraTreesRegressor) | 42 | **0.0454** | 0.0564 | +0.0256 | 0.1616 | 0.1570 | +0.0000 | +0.00% | +0.0009 |
| **exp_d_ExtraTreesRegressor** (ExtraTreesRegressor) | 42 | **0.0454** | 0.0564 | +0.0256 | 0.1616 | 0.1570 | +0.0000 | +0.00% | +0.0009 |
| **exp_d_RandomForestRegressor** (RandomForestRegressor) | 42 | **0.0454** | 0.0564 | +0.0255 | 0.1631 | 0.1632 | +0.0000 | +0.00% | +0.0008 |
| **exp_d_HistGradientBoostingRegressor** (HistGradientBoostingRegressor) | 42 | **0.0457** | 0.0567 | +0.0139 | 0.1376 | 0.1321 | +0.0003 | +0.66% | -0.0108 |
| **exp_d_GradientBoostingRegressor** (GradientBoostingRegressor) | 42 | **0.0458** | 0.0569 | +0.0087 | 0.1215 | 0.1293 | +0.0004 | +0.88% | -0.0160 |

---

## 2. Decision-Quality & Video-Level Diagnostics

### Per-Video MAE Delta Distribution ($\Delta = \text{MAE}_{\text{new}} - \text{MAE}_{\text{base}}$):
- **Mean Delta**: **-0.0000**
- **Median Delta**: **-0.0001**
- **Standard Deviation**: **0.0013**
- **25th Percentile ($Q_1$)**: **-0.0008**
- **75th Percentile ($Q_3$)**: **+0.0005**

### Video-Level Improvement Breakdown (100 Validation Videos):
- **Improved Videos**: **49 / 100** (49.0%)
- **Worsened Videos**: **42 / 100** (42.0%)
- **Unchanged Videos**: **9 / 100** (9.0%)

### Deterministic Paired Bootstrap 95% Confidence Interval ($B = 2,000$, seed=42):
- **95% Bootstrap CI**: **[-0.0003, +0.0002]**
- **Bootstrap Mean ($\pm$ SD)**: **-0.0000** ($\pm$ 0.0001)
- **Diagnostic Finding**: 95% bootstrap CI for mean video-level MAE difference is [-0.0003, +0.0002]. Interval contains zero, indicating difference is not statistically distinguishable from zero at alpha=0.05.

---

## 3. Robustness Stratification

### Target Concentration Bins:
| Target Range Bin | Sample Count | Target Mean | Base MAE | Upgrade MAE | $\Delta\text{MAE}$ | Base Bias | Upgrade Bias |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.60–0.70** | 45 | 0.6697 | 0.1723 | 0.1723 | **-0.0000** | +0.1723 | +0.1723 |
| **0.70–0.80** | 850 | 0.7662 | 0.0740 | 0.0733 | **-0.0007** | +0.0740 | +0.0733 |
| **0.80–0.90** | 2,426 | 0.8519 | 0.0250 | 0.0253 | **+0.0003** | -0.0093 | -0.0098 |
| **0.90–1.00** | 649 | 0.9203 | 0.0754 | 0.0755 | **+0.0001** | -0.0754 | -0.0755 |

### Temporal Playback Quartiles:
| Quartile Interval | Sample Count | Target Mean | Base MAE | Upgrade MAE | $\Delta\text{MAE}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1 (0.00–0.25)** | 1,007 | 0.8404 | 0.0471 | 0.0471 | **-0.0000** |
| **Q2 (0.25–0.50)** | 955 | 0.8414 | 0.0447 | 0.0443 | **-0.0004** |
| **Q3 (0.50–0.75)** | 984 | 0.8412 | 0.0447 | 0.0450 | **+0.0003** |
| **Q4 (0.75–1.00)** | 1,024 | 0.8475 | 0.0450 | 0.0452 | **+0.0002** |

---

## 4. Feature Importance Concentration (Top 10 Features)

| Rank | Feature Name | Impurity Importance (Gini) | Cumulative Share |
| :--- | :--- | :--- | :--- |
| 1 | `contrast` | **0.0468** | 0.0468 |
| 2 | `contrast_lead1` | **0.0457** | 0.0925 |
| 3 | `text_presence` | **0.0446** | 0.1371 |
| 4 | `brightness_mean_lead1` | **0.0441** | 0.1811 |
| 5 | `contrast_lag1` | **0.0433** | 0.2244 |
| 6 | `text_presence_lag1` | **0.0417** | 0.2662 |
| 7 | `brightness_mean_lag1` | **0.0411** | 0.3073 |
| 8 | `brightness_mean` | **0.0409** | 0.3482 |
| 9 | `text_presence_lead1` | **0.0384** | 0.3866 |
| 10 | `visual_complexity` | **0.0376** | 0.4242 |

- **Top 3 Features Share**: **13.7%**
- **Top 5 Features Share**: **22.4%**
- **Top 10 Features Share**: **42.4%**

> [!NOTE]
> **Non-Causal Note**: Feature importance metrics reflect Gini impurity reductions within the decision tree ensemble and describe predictive utility in this tabular space; they do **not** represent causal visual mechanisms.

---

## 5. Decision Rule & Final Recommendation

### Final Decision: **Representation upgrade marginal / Phase 2 model retained**

**Rationale**:  
The representation upgrade yields a marginal change over the Phase 2 baseline (Delta MAE = +0.0000, relative = +0.00%, Delta R^2 = +0.0009). The 95% bootstrap confidence interval [-0.0003, +0.0002] contains zero, confirming the difference is not statistically distinguishable from zero. Per-video balance is 49 improved vs 42 worsened.

**Technical Recommendation**:  
Retain Phase 2 independent model (models/human_attention_experiments/dhf1k_independent/model_dhf1k.pkl) unchanged.
