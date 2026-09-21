# Phase 11 — Human Qualitative Utility Evaluation Protocol

## 1. Objective & Scope
This protocol governs the blind qualitative evaluation of **TinySalNet** spatial visual-saliency predictions on real ViralLens-compatible Instagram Reel MP4 videos.

> **Evaluation-Only Mandate**:
> - Perceived usefulness and visual plausibility are assessed qualitatively.
> - **Zero Ground-Truth Gaze Claims**: These ratings assess human-perceived utility only; they do NOT measure physiological eye-tracking gaze fixations.
> - **Blinded Review**: Reviewers see only the input frame, predicted saliency heatmap overlay, and centroid/peak visualization. No model scores, confidence values, DHF1K metrics, or selection strata are disclosed.

---

## 2. Evaluation Cohort Description
The cohort consists of **model-telemetry-selected convenience samples** (20 static frames across 8 defined visual strata + 10 temporal sequence frames) drawn from 15 authentic ViralLens-compatible videos. Ratings represent perceived usefulness for this specific sample and are not generalized to all Instagram Reels.

---

## 3. Rating Criteria & Rubric (1–5 Likert Scale)

Reviewers evaluate each item across 5 standardized dimensions:

### Criterion A: Focal-Region Plausibility (1–5)
- **1 (Very Implausible)**: Focuses exclusively on empty background or uninformative corner artifacts.
- **2 (Poor Plausibility)**: Major focal object missed; minor secondary details highlighted.
- **3 (Moderate Plausibility)**: Highlights plausible visual content with noticeable diffuse spillover.
- **4 (Good Plausibility)**: Accurately identifies primary visual subject with minor edge bleed.
- **5 (Highly Plausible)**: Precisely concentrates on the most salient visual element in the scene.

### Criterion B: Spatial Map Usefulness (1–5)
- **1 (Useless)**: Density distribution provides zero practical insight for video analysis.
- **2 (Low Utility)**: Map is either completely flat or arbitrarily fragmented.
- **3 (Moderate Utility)**: Broad visual focus is discernible and usable with caveats.
- **4 (Useful)**: Clear spatial guidance on viewer focal density.
- **5 (Highly Useful)**: Clean, actionable spatial density representation suitable for creative diagnostics.

### Criterion C: Visual Coherence with the Frame (1–5)
- **1 (No Coherence)**: Heatmap boundaries bear no relationship to objects in the frame.
- **2 (Weak Coherence)**: Shapes are distorted or arbitrarily offset from physical objects.
- **3 (Moderate Coherence)**: Centers on subjects but lacks boundary conformity.
- **4 (Good Coherence)**: Saliency boundaries conform closely to key subject contours.
- **5 (Excellent Coherence)**: Crisp semantic alignment with visual subject boundaries.

### Criterion D: Temporal Consistency (1–5, Temporal Sequences Only)
- **1 (Severely Erratic)**: Saliency jumps randomly across the frame between consecutive frames.
- **2 (Poor Continuity)**: Noticeable flickering and abrupt centroid shifts without camera motion.
- **3 (Acceptable Continuity)**: Moderate stability with minor temporal jitter.
- **4 (Smooth Continuity)**: Smooth, physically plausible transitions tracking movement.
- **5 (Seamless Consistency)**: Rock-solid temporal progression perfectly reflecting motion.

### Criterion E: Overall Usefulness for Video Analysis (1–5)
- **1 (Unusable)**: Provides negative or misleading utility.
- **2 (Marginal)**: Minimal additive value over raw video frames.
- **3 (Acceptable)**: Moderately helpful as a supplemental observational signal.
- **4 (Valuable)**: Clear practical utility for scene composition and visual pacing audit.
- **5 (Exceptional)**: High-impact analytical asset for creative intelligence.

*Special Option*: Reviewers may explicitly mark **"unclear / not useful"** when a frame cannot be meaningfully interpreted.
