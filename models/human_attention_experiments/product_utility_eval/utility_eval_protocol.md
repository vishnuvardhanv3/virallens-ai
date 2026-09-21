# Phase 12: Controlled Product-Utility Evaluation Protocol

## 1. Experimental Design & Objective
This protocol governs the controlled within-subject crossover product-utility evaluation of the frozen **`TinySalNet`** spatial visual-saliency model in ViralLens.

### Primary Research Question
*Does an otherwise identical ViralLens analysis report become more useful for a human reviewer when the experimental TinySalNet spatial saliency visualization is shown?*

### Experimental Invariance Mandate
- **Strict Observational Shadow Mode**: TinySalNet operates strictly as an observational, decoupled signal on CPU in `eval()` mode.
- **Zero Production Coupling**: Zero score blending, zero ensembling, and zero modification of production NEMAR visual attention scoring, Apify discovery queries, candidate ranking, pattern discovery, or recommendation generation.
- **SHA-256 Hashes Verified**:
  - TinySalNet (`spatial_model.pt`): `fd6f27fd5ef6334c597ee3981086fc70f639cd59c7d57399fa72253d883b95c9`
  - NEMAR (`model.pkl`): `38b8c952d0806a2fbed52624a497b3bdb1b69554bc81b247e64770e70c26fa89`
  - NEMAR (`scaler.pkl`): `438102936820e4f1729dd643078cb3d545bc2fe369b89c95c05cd0e58ce81f58`
  - NEMAR (`feature_schema.json`): `bde5b68e4256d7635733c7d33ca9d18db5118d39e81ce37aa5419c1bde46e411`

---

## 2. Matched Conditions
- **Condition A (Control)**: Standard ViralLens analysis report (Reel metadata, NEMAR baseline/peak attention, chronological events, OCR, hook/pacing analysis) **without** TinySalNet visualization.
- **Condition B (Experimental)**: Identical underlying ViralLens analysis report **plus** Section 2b: *"Experimental DHF1K Spatial Saliency — Shadow Mode"* (predicted spatial heatmap overlay, mass centroid, peak focus point, spatial dispersion, and entropy).

---

## 3. Within-Subject Crossover Design & Counterbalancing
- **Cohort**: 20 matched items drawn from the authentic Phase 10 video cohort across 7 content niches (`comedy`, `entertainment`, `anime`, `superhero`, `music`, `text_dense`, `high_motion`).
- **Counterbalancing**:
  - Deterministic Latin square / ABBA partition (fixed random seed = 42).
  - Each reviewer evaluates exactly 10 items in Condition A and 10 items in Condition B (50/50 balance).
  - Across the cohort, each item is evaluated under both Condition A and Condition B across reviewers.
  - No video is presented back-to-back in both conditions to eliminate memory and fatigue biases.

---

## 4. Four Concrete Video-Analysis Tasks
1. **Task A: Primary Focal Region Identification**
   - *Task*: Identify where the viewer's visual focus is primarily concentrated in the key scene (quadrant / dominant actor / text).
   - *Outcome*: Selected region; sub-second completion time (seconds).
2. **Task B: Most Important Visual Moment Detection**
   - *Task*: Pinpoint the sampled timestamp with the highest visual impact / focus density.
   - *Outcome*: Selected timestamp; decision confidence (1–5 Likert).
3. **Task C: Focal Competition Assessment**
   - *Task*: Determine whether the frame exhibits a single dominant focal subject vs. multiple competing visual stimuli (clutter/subtitles/background).
   - *Outcome*: Binary classification (`single_dominant` vs. `competing_stimuli`); perceived usefulness (1–5 Likert).
4. **Task D: Overall Creative Analysis Assessment**
   - *Task*: Formulate a holistic visual composition recommendation for the Reel.
   - *Outcome*: Decision confidence (1–5), perceived usefulness (1–5), need for additional inspection (`True`/`False`), optional feedback comment.

---

## 5. Statistical Inference & Reporting
- **Primary Outcome**: Task completion time difference ($\Delta t = t_B - t_A$).
- **Secondary Outcomes**: Decision confidence delta ($\Delta \text{conf}$), perceived usefulness delta ($\Delta \text{use}$), additional inspection rate delta ($\Delta \text{insp}$).
- **Inference Methods**:
  - Clustered bootstrap 95% confidence intervals (clustered at item and reviewer levels, 2,000 resamples).
  - Wilcoxon signed-rank test and paired Student's $t$-test.
  - Cohen's $d_z$ effect size for paired samples.
  - Stratified analysis across 7 content niches.
  - 8-category qualitative failure taxonomy.

---

## 6. Scientific Disclaimers & Participant Anonymity
- **Subjective Utility Only**: Reviewer ratings assess human decision-support utility only; they do NOT measure physiological eye-tracking gaze fixations on Instagram Reels.
- **Blinded Presentation**: Reviewers see only presentation panels; no model training loss, DHF1K benchmark scores, or selection strata are disclosed.
- **Reviewer Anonymity**: Reviewers are anonymized as `reviewer_01`, `reviewer_02`, etc.
