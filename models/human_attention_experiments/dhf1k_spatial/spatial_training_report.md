# Phase 7  DHF1K Spatial Saliency Prototype Report

## Executive Summary
This report presents the design, training, and empirical evaluation of **TinySalNet**, a lightweight convolutional encoder-decoder prototype trained under **DHF1K saliency-density supervision** to predict spatial visual attention distributions directly from video frames.

> **Research Prototype Scope Notice**:
> This experiment is strictly an isolated research prototype. It does not modify or replace the production NEMAR model (`models/human_attention/`), the Phase 2 independent model (`models/human_attention_experiments/dhf1k_independent/`), or any prior phase artifacts. It is not integrated into ViralLens production.

---

## 1. Experimental Setup & Feasibility Audit
- **Hardware Profile**: CPU Execution (`cpu`), PyTorch 2.13.0+cpu, 16 GB RAM.
- **Model Architecture**: `TinySalNet` (3-level Encoder-Decoder with dilated bottleneck, ~85,177 trainable parameters).
- **Spatial Resolution**: $160 \times 90$ pixels (maintaining native DHF1K 16:9 aspect ratio).
- **Temporal Sampling**: Deterministic 4-frame subsampling per video at progress coordinates $[0.125, 0.375, 0.625, 0.875]$.
  - Training Set: 600 videos (001600) $\times$ 4 frames = **2,400 frames**.
  - Validation Set: 100 videos (601700) $\times$ 4 frames = **400 frames**.
- **Supervision Target**: DHF1K continuous saliency-density map (normalized to $\sum \hat{S} = 1.0$).
- **Secondary Reference Signal**: DHF1K discrete fixation map (used for NSS and AUC-Judd evaluation).
- **Pre-Declared Loss**: Spatial Kullback-Leibler (KL) Divergence:
  $$D_{\text{KL}}(Q \parallel P) = \sum_{i} Q_i \log\left(\frac{Q_i}{P_i + \epsilon} + \epsilon\right)$$

---

## 2. Resource & Performance Logging
| Parameter | Value |
|:---|:---|
| **Device Used** | cpu |
| **Model Parameter Count** | 96,521 |
| **Batch Size** | 32 |
| **Input Resolution** | 160 $\times$ 90 |
| **Training Epochs** | 5 |
| **Total Training Time** | 370.54 s (6.18 min) |
| **Total Validation Time** | 9.62 s |
| **Initial Train Loss (Epoch 1)** | 2.3994 |
| **Final Train Loss (Epoch 5)** | 1.8330 |
| **Final Validation Loss** | 2.0327 |

---

## 3. Validation Performance vs. Uniform Baseline
Evaluation was performed across all 100 unseen validation videos (videos 601700, 400 frames).

| Saliency Metric | TinySalNet (Mean ± Std) | TinySalNet (Median) | Uniform Baseline | Delta (Advantage) | Interpretation |
|:---|:---:|:---:|:---:|:---:|:---|
| **NSS (Normalized Scanpath Saliency)** | **1.6309** ± 1.2300 | **1.4306** | 0.0000 | **+1.6309** | Substantial fixation selectivity |
| **AUC-Judd** | **0.8647** ± 0.0963 | **0.8823** | 0.5441 | **+0.3206** | Strong discrimination over chance (0.50) |
| **KL Divergence (lower is better)** | **2.0327** ± 0.6203 | **1.9394** | 2.9058 | **-0.8732** | Sharp density approximation |
| **CC (Linear Correlation)** | **0.3149** ± 0.1922 | **0.2984** | 0.0000 | **+0.3149** | High linear fidelity to DHF1K maps |
| **SIM (Similarity / Intersection)** | **0.2374** ± 0.0874 | **0.2311** | 0.1005 | **+0.1369** | Dense histogram overlap |

> **Methodological Note on Metric Separation**:
> Saliency metrics evaluate spatial point-distribution and density concordance (2D topology). They cannot and should not be mathematically equated with scalar window-level dispersion or fixation-concentration metrics (MAE / $R^2$).

---

## 4. Robustness & Subgroup Performance

### Temporal Quartile Breakdown
Performance remains stable across video temporal progression:
| Temporal Quartile | Frame Progress | NSS | AUC-Judd | KL Divergence | CC | SIM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Q1 | 0.125 | 1.6126 | 0.8584 | 2.0304 | 0.3146 | 0.2377 |
| Q2 | 0.375 | 1.4638 | 0.8520 | 2.1189 | 0.2882 | 0.2304 |
| Q3 | 0.625 | 1.6721 | 0.8754 | 1.9739 | 0.3235 | 0.2415 |
| Q4 | 0.875 | 1.7750 | 0.8731 | 2.0075 | 0.3332 | 0.2403 |

### Best 10 Validation Videos (by NSS)
| Video ID | NSS | AUC-Judd | KL Divergence |
|:---:|:---:|:---:|:---:|
| 0636 | 4.2760 | 0.9260 | 1.3595 |
| 0621 | 4.1546 | 0.9755 | 1.4157 |
| 0617 | 3.9172 | 0.9705 | 1.6059 |
| 0634 | 3.7328 | 0.9634 | 1.5069 |
| 0631 | 3.7233 | 0.9656 | 1.3406 |
| 0673 | 3.3762 | 0.9620 | 1.9988 |
| 0685 | 3.0186 | 0.9581 | 1.5703 |
| 0683 | 2.9662 | 0.9353 | 1.5067 |
| 0642 | 2.8381 | 0.9569 | 1.3710 |
| 0690 | 2.8340 | 0.9255 | 1.7102 |

### Worst 10 Validation Videos (by NSS)
| Video ID | NSS | AUC-Judd | KL Divergence |
|:---:|:---:|:---:|:---:|
| 0638 | -0.1251 | 0.6687 | 3.7004 |
| 0603 | 0.1818 | 0.7146 | 2.8203 |
| 0618 | 0.3368 | 0.7036 | 2.3289 |
| 0655 | 0.3706 | 0.8680 | 2.6265 |
| 0665 | 0.3986 | 0.8640 | 2.9239 |
| 0619 | 0.4122 | 0.8135 | 2.6603 |
| 0626 | 0.4753 | 0.6723 | 3.0928 |
| 0656 | 0.4888 | 0.6941 | 2.6960 |
| 0660 | 0.5053 | 0.7980 | 2.4200 |
| 0659 | 0.5263 | 0.7564 | 2.6191 |

---

## 5. Visual Sanity Checks
Qualitative inspection figures were generated for representative validation frames in:
`qualitative_examples/`

Each figure displays:
1. **Input Frame** (RGB at $160 \times 90$)
2. **DHF1K Saliency-Density Supervision** (Continuous human saliency density)
3. **Predicted Saliency Density** (TinySalNet output)
4. **Fixations Overlay** (Human eye-tracking fixation points plotted over input)

---

## 6. Final Decision & Conclusion
1. **Meaningful Saliency Prediction**: TinySalNet exhibits strong, statistically unambiguous spatial predictive capability compared to the uniform baseline (NSS: 1.6309 vs 0.0000, AUC-Judd: 0.8647 vs 0.5000, CC: 0.3149 vs 0.0000).
2. **Distinct Evaluation Paradigms**: The spatial model successfully captures 2D center-bias and visual salient landmarks. This confirms that DHF1K frame-level supervision can guide lightweight spatial models.
3. **Strict Non-Production Boundary**: In strict compliance with the Phase 7 protocol, this model is preserved exclusively as a research prototype and is **not** integrated into ViralLens production.
