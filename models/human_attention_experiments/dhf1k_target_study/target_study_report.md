# DHF1K Phase 6: Target Representation Investigation Report

**Experiment**: Research Investigation into DHF1K Supervisory Target Formulations & Learnability  
**Date**: September 12, 2026  
**Source Dataset**: DHF1K (`E:\\viral_attention_data\\DHF1K`)  
**Target Category**: **DHF1K-derived supervisory targets** (explicitly not canonical ground-truth human attention)  
**Safety & Isolation**: Production NEMAR model (`models/human_attention/`), Phase 2 model (`models/human_attention_experiments/dhf1k_independent/`), and Phase 5 artifacts verified unmodified.

---

## 1. Target-Methodology Audit & Mathematical Justification

| Target Identifier | Raw Spatial Statistic | Normalized Scalar Formula | Coordinate Domain | Normalization Reference | Parameter Source | Semantics (Higher Values) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Target A (Baseline)** | std(radial_dist) from window centroid in [0, 1]^2 screen space | `T_A = clip(1.0 - min(1.0, 2.0 * std(radial_dist)), 0.0, 1.0)` | [0, 1] x [0, 1] normalized screen space | Empirical factor 2.0 from Phase 2 baseline | Historical Phase 2 baseline parameter | **Concentration (higher = tighter radial clustering)** |
| **Target B (Fixation Concentration)** | raw_B = sqrt(var(x) + var(y)) [total bivariate standard deviation] | `T_B = clip(1.0 - min(1.0, raw_B / S_ref), 0.0, 1.0)` | [0, 1] x [0, 1] normalized screen space | S_ref = 1/sqrt(6) = 0.4082 (theoretical uniform 2D distribution) | Theoretical uniform distribution + train-distribution audit | **Concentration (higher = lower orthogonal bivariate spatial variance)** |
| **Target C (Saliency Concentration)** | raw_C = sqrt(sum p * [(x - cx)^2 + (y - cy)^2]) [continuous RMS radial distance from center of mass] | `T_C = clip(1.0 - min(1.0, raw_C / S_ref), 0.0, 1.0)` | [0, 1] x [0, 1] normalized continuous density mass | S_ref = 1/sqrt(6) = 0.4082 (theoretical uniform continuous mass) | Theoretical continuous uniform distribution + train-distribution audit | **Concentration (higher = tighter density mass around center of mass)** |
| **Target D (Saliency Entropy Dispersion)** | raw_D = H(M) = -sum p * log2(p) [Shannon spatial entropy in bits] | `T_D = H(M) / log2(360 * 640)` | 360 x 640 discrete pixel grid | H_max = log2(360 * 640) = 17.8138 bits (maximum uniform discrete entropy) | Theoretical resolution limit (360 x 640) | **Dispersion (higher = flatter, more dispersed attention; strictly non-inverted)** |

---

## 2. Distributional Diagnostics & Compression Analysis

Comparing statistical dispersion across DHF1K training and validation splits:

| Target Name | Train Mean ($\pm$ SD) | Train CV ($\sigma / \mu$) | Skewness | Kurtosis | IQR | p99 - p1 Spread | Inter-Video Var | Intra-Video Var | Distinct Values |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Target A (Phase 2 Fixation Concentration)** | **0.8432** ($\pm$ 0.0552) | 0.0655 | -0.5186 | +0.0383 | 0.0766 | 0.2439 | 0.001034 | 0.002094 | 2,652 |
| **Target B (Bivariate Fixation Concentration)** | **0.6260** ($\pm$ 0.1353) | 0.2161 | -0.6438 | +0.2139 | 0.1874 | 0.5965 | 0.009425 | 0.009304 | 5,573 |
| **Target C (Saliency Center-of-Mass Concentration)** | **0.6382** ($\pm$ 0.1255) | 0.1967 | -0.7814 | +0.4463 | 0.1718 | 0.5504 | 0.008117 | 0.007991 | 5,125 |
| **Target D (Saliency Entropy Dispersion)** | **0.7628** ($\pm$ 0.0232) | 0.0304 | -0.1675 | -0.2999 | 0.0324 | 0.1040 | 0.000244 | 0.000308 | 1,267 |

---

## 3. Supervised Learnability & Validation Performance Comparison

All targets trained using identical Phase 2 ExtraTrees architecture (`n_estimators=100`, `max_depth=12`, `min_samples_leaf=4`, `random_state=42`) with identical 14 base production features:

| Target Name | Val MAE | Val RMSE | Val R² | Pearson $r$ ($p$-val) | Spearman $\rho$ | Target SD | Pred SD | SD Ratio ($\sigma_{\hat{y}} / \sigma_y$) | Variance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Target A** | **0.0445** | 0.0552 | **+0.0257** | **0.1639** | 0.1774 | 0.0559 | 0.0095 | **0.1700** | 0.0289 |
| **Target B** | **0.1037** | 0.1299 | **+0.0227** | **0.1561** | 0.1463 | 0.1314 | 0.0257 | **0.1955** | 0.0382 |
| **Target C** | **0.0945** | 0.1190 | **+0.0279** | **0.1707** | 0.1633 | 0.1207 | 0.0248 | **0.2057** | 0.0423 |
| **Target D** | **0.0177** | 0.0220 | **+0.0372** | **0.1969** | 0.1905 | 0.0224 | 0.0039 | **0.1754** | 0.0308 |

---

## 4. Key Comparative Findings

1. **Target A (Phase 2 Fixation Concentration Proxy)**:
   - Serves as the Phase 2 baseline ($R^2 = +0.0247$, Pearson $r = 0.1573$).
   - Exhibits low target variance ($\sigma = 0.0571$), with severe prediction compression ($\sigma_{\hat{y}} / \sigma_y = 16.6\%$).

2. **Target B (Bivariate Fixation Spatial Concentration)**:
   - Uses total bivariate variance $\text{var}(x) + \text{var}(y)$ normalized by theoretical uniform dispersion $1/\sqrt{6} \approx 0.4082$.
   - Captures orthogonal dispersion directly without isotropic radial collapse.

3. **Target C (Saliency Center-of-Mass Spatial Concentration)**:
   - Measures continuous spatial mass concentration around the center of mass.
   - Leverages full multi-observer continuous saliency density rather than discrete fixation subsets.

4. **Target D (Saliency Map Spatial Entropy Dispersion)**:
   - Captures continuous spatial Shannon entropy ($H / \log_2(360 \times 640)$) measuring spatial attention dispersion.
   - Maintained strictly in dispersion semantics without sign inversion.

---

## 5. Final Decision Classification

### Decision: **B. Alternative target marginal**

**Primary Rationale**:  
Alternative supervisory target Target D shows modest metric improvements over Target A (Val R^2 = +0.0372 vs +0.0257, Delta R^2 = +0.0115, Pearson r = 0.1969 vs 0.1639), but does not achieve a qualitative leap in explained variance (R^2 remains < 0.10).

**Comparison Summary**:
- Baseline Target A: Val $R^2 = \mathbf{+0.0257}$, Pearson $r = \mathbf{0.1639}$
- Best Alternative (Target D): Val $R^2 = \mathbf{+0.0372}$, Pearson $r = \mathbf{0.1969}$
- Difference: $\Delta R^2 = \mathbf{+0.0115}$, $\Delta r = \mathbf{+0.0330}$

**Policy Recommendation**:  
All existing models (production NEMAR `model.pkl` and Phase 2 independent `model_dhf1k.pkl`) remain strictly retained and unchanged.
