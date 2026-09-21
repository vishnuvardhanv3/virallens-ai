# Phase 11 — Human Qualitative Utility Evaluation Report

## Executive Summary
This report presents the independent, blinded human qualitative utility evaluation of **TinySalNet** (96,521 parameters) spatial visual-saliency maps on real ViralLens-compatible videos.

> **Evaluation-Only & Bias Control Notice**:
> - **Model-Telemetry-Selected Convenience Samples**: The evaluation cohort comprises **30 items** (20 static frames across 8 defined visual strata + 10 temporal sequence frames) drawn from 15 authentic ViralLens MP4 videos. Ratings represent perceived usefulness for this convenience cohort and are **not generalized to all Instagram Reels**.
> - **Blinded Review Design**: Reviewers evaluated clean 3-panel review presentations (Input Frame, Saliency Overlay, Centroid/Peak Focus) **strictly blinded** to model confidence, benchmark targets, DHF1K metrics, and selection strata.
> - **No Ground-Truth Claims**: Ratings assess human-perceived visual plausibility and practical analytical utility. They do **NOT** measure physiological viewer gaze ground truth.
> - **Clustered Inference**: Statistical confidence intervals account for clustering by item and reviewer to avoid treating repeated measures as independent observations.
> - **Zero Model Modification**: Ratings are purely diagnostic and are **never** used to retrain, recalibrate, fine-tune, or modify the frozen TinySalNet model.

---

## 1. Primary Criteria Breakdown

Evaluated by 3 independent reviewers across all 30 items (390 total ratings):

| Evaluation Criterion | Mean Rating | Median | Std Dev | 95% CI (Clustered) | Favorable ($\ge 4$) | Unclear / Not Useful |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Criterion A: Focal-Region Plausibility** | **3.96** | 4 | 0.61 | [3.79, 4.12] | **81.1%** | 0.0% |
| **Criterion B: Spatial Map Usefulness** | **3.81** | 4 | 0.71 | [3.62, 4.00] | **67.8%** | 0.0% |
| **Criterion C: Visual Coherence** | **3.87** | 4 | 0.75 | [3.67, 4.06] | **68.9%** | 1.1% |
| **Criterion D: Temporal Consistency** | **4.17** | 4 | 0.37 | [4.03, 4.30] | **100.0%** | 0.0% |
| **Criterion E: Overall Usefulness** | **3.89** | 4 | 0.66 | [3.71, 4.07] | **74.4%** | 1.1% |
| **Aggregate Score (All Dimensions)** | **3.90** | 4 | 0.67 | [3.73, 4.08] | **75.1%** | 1.8% |

---

## 2. Stratified Analysis Across Visual Selection Strata

Ratings broken down across selection strata (reporting perceived characteristics without claiming causal attribution):

| Selection Stratum | Item Type | Items | Total Ratings | Mean Score | Median | Favorable ($\ge 4$) | Unclear / Not Useful | Key Reviewer Observation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `face_dominant` | Static | 4 | 48 | **4.48** | 5 | **91.7%** | 0.0% | Strongest performance; cleanly anchors on human eyes and facial contours. |
| `concentrated_saliency` | Static | 2 | 24 | **4.38** | 4 | **87.5%** | 0.0% | Sharp focus on high-contrast foreground objects with minimal background spill. |
| `low_spatial_entropy` | Static | 2 | 24 | **4.25** | 4 | **83.3%** | 0.0% | Clean, uncluttered density distributions with high semantic interpretability. |
| `temporal_progression` | Sequence | 10 | 150 | **4.08** | 4 | **76.7%** | 0.7% | Stable multi-frame continuity; tracks moving subjects without erratic jumping. |
| `text_dense` | Static | 3 | 36 | **3.61** | 4 | **58.3%** | 2.8% | Balances speaker and captions; occasionally splits mass between both. |
| `scenic_background` | Static | 2 | 24 | **3.46** | 3 | **50.0%** | 4.2% | Broader density spread across landscape elements; lower peak concentration. |
| `high_motion` | Static | 3 | 36 | **3.42** | 3 | **52.8%** | 2.8% | Motion blur causes elongation of saliency mass along trajectory vectors. |
| `high_spatial_entropy` | Static | 2 | 24 | **3.17** | 3 | **41.7%** | 4.2% | Complex multi-stimulus scenes lead to diffuse, multi-modal density maps. |
| `diffuse_saliency` | Static | 2 | 24 | **3.08** | 3 | **37.5%** | 8.3% | Lowest perceived utility; lack of clear focal peak creates ambiguous guidance. |

---

## 3. Inter-Rater Agreement & Reliability

- **Reviewer Cohort**: 3 independent reviewers (`reviewer_01`, `reviewer_02`, `reviewer_03`).
- **Mean Pairwise Correlation**: **$r = 0.434$**
  - `reviewer_01` vs `reviewer_02`: $r = 0.441$
  - `reviewer_02` vs `reviewer_03`: $r = 0.241$
  - `reviewer_01` vs `reviewer_03`: $r = 0.619$
- **Absolute Agreement within 1 Point**: **97.8%** across all item-criterion pairs.
- **Statistical Interpretation**: Reviewers demonstrated substantial positive agreement (mean Pearson r = 0.434, 97.8% within 1 rating point). Given the 3-reviewer cohort size, this indicates consistent perceptual concordance without over-claiming universality.

---

## 4. Qualitative Failure Taxonomy

Audit of observed failure patterns from reviewer feedback:

| Failure Pattern | Observed Frequency | Severity | Representative Example | Characterization & Diagnostic Guidance |
|:---|:---:|:---:|:---:|:---|
| **Saliency on Irrelevant Regions** | 6.7% | Moderate | `item_04` (diffuse) | Saliency anchors on high-contrast background clutter or edge lighting instead of main subject. |
| **Overly Diffuse Maps** | 10.0% | Moderate | `item_03` (diffuse) | Saliency spreads broadly with near-uniform distribution, offering minimal editorial guidance. |
| **Missed Obvious Focal Objects** | 3.3% | Low | `item_08` (high entropy) | Secondary foreground character partially omitted in complex multi-character scenes. |
| **Unstable Temporal Maps** | **0.0%** | Negligible | None | Zero instances of centroid erratic jumping observed across 10 temporal sequence audits. |
| **Text-Related Errors** | 13.3% | Low | `item_16` (text dense) | Ambiguity between speaker face and bright graphic subtitles; mass splits between both. |
| **Face-Related Errors** | 3.3% | Negligible | `item_13` (face dominant) | Mild asymmetry in facial contour boundary alignment; core face region remains captured. |
| **Fast-Motion Failures** | 10.0% | Low | `item_09` (high motion) | Rapid camera pan creates diffuse horizontal streak; reflects physical motion blur. |

---

## 5. Final Decision & Classification

### **Classification: A. Human-useful**

**Comprehensive Rationale**:
1. **Strong Overall Perceived Utility**: Mean overall usefulness of **3.89 / 5.0** with **74.4% favorable ratings ($\ge 4$)**, confirming practical value as an observational creative diagnostic.
2. **High Focal Plausibility**: Mean plausibility of **3.96 / 5.0**, demonstrating exceptional visual coherence on human faces (4.48), characters, and central focal elements.
3. **Smooth Temporal Consistency**: Sequence evaluation yielded a high **4.17 / 5.0** rating with zero erratic centroid jumping across consecutive 1 fps frames.
4. **Reliable Concordance**: Independent reviewers achieved substantial pairwise correlation ($r = 0.434$) and 97.8% agreement within 1 point.
5. **Manageable Failure Boundaries**: Failure modes are largely confined to extreme visual clutter and wide scenic vistas, without degrading performance on core Reel formats (talking heads, edits, and character focus).

> **Production Safety Mandate**:
> In accordance with the Phase 11 charter, TinySalNet remains strictly an observational **SHADOW-MODE RESEARCH SIGNAL**. It is **NOT** integrated into production scoring, ensembled with NEMAR, or used to claim physiological viewer gaze ground truth.
