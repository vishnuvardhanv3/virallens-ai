# Phase 14 — Attention Intelligence Integration

## 1. Objective
Transform the already validated, frozen NEMAR temporal visual-attention signal and TinySalNet/DHF1K spatial visual-saliency signal into an interpretable, cross-modal "Attention Intelligence" layer within ViralLens. This phase represents pure product integration without adding models, retraining weights, altering competitor discovery, or making ungrounded causal claims regarding viewer retention or scroll-stopping.

---

## 2. Existing Signals
Before modifications, the codebase contained two independent model signals:
1. **NEMAR Temporal Signal**: Random Forest regressor (100 estimators, 14 multimodal features including motion magnitude, scene cut rate, face presence, text presence, visual complexity, and audio features) predicting temporal visual-attention potential across 0.5s windows.
2. **TinySalNet Spatial Saliency Signal**: Deep 2D convolutional model (96,521 parameters) operating on 160×90 frames on CPU in `eval()` mode, producing normalized 2D density maps with dispersion, entropy, centroid $(c_x, c_y)$, peak $(p_x, p_y)$, and inter-frame shift.
3. **Contextual Signals**: Twelve Labs multimodal video subjects, OpenCV technical metrics, and Whisper/Tesseract OCR text presence and timestamps.

---

## 3. Canonical Schema
Created the canonical internal schema in `services/attention_intelligence.py` with full provenance preservation:
- `schema_version`: `"1.0.0"`
- `status`: `"SUCCESS"` | `"PARTIAL"` | `"FAILED"`
- `temporal`: source (`NEMAR`), signal type (`laboratory_derived_attention_signal`), trajectory, baseline, peak, peak timestamp, relative peak position, opening signal, temporal persistence, drops, recoveries, and qualitative relative descriptors.
- `spatial`: source (`TinySalNet_DHF1K`), signal type (`predicted_visual_saliency`), centroid, peak focus, dispersion, entropy, temporal centroid drift, focal concentration, and principled focal competition analysis.
- `context`: Twelve Labs visual subjects, OCR typography, motion magnitude, scene cuts, face presence, and visual complexity.
- `relationships`: cross-modal spatial-temporal alignment, focal competition, text-saliency interaction, face-saliency interaction, and scene-transition alignment.
- `events`: discrete chronological attention events (`OPENING_FOCUS`, `ATTENTION_PEAK`, `ATTENTION_DROP`, `ATTENTION_RECOVERY`, `SPATIAL_FOCUS_SHIFT`, `SCENE_TRANSITION_ALIGNMENT`, `FOCAL_COMPETITION`, `TEXT_SALIENCY_INTERACTION`, `FACE_SALIENCY_ALIGNMENT`).
- `interpretation`: evidence-backed observations strictly using non-causal language.
- `limitations`: explicit scientific boundaries.
- `provenance`: model architectures, hashes, timestamps, and window/frame counts.

---

## 4. Temporal Intelligence
Derived interpretable temporal properties from NEMAR predictions without inventing arbitrary thresholds:
- **Baseline**: Opening 2.0s average attention potential.
- **Peak & Relative Position**: Highest predicted window score and normalized timestamp $t_{peak} / t_{total}$.
- **Temporal Persistence**: Duration and fraction of video where attention potential exceeds the video median.
- **Attention Modulation**: Empirical drop and recovery detection where consecutive window delta exceeds $1.2 \times \sigma$ ($> 0.04$).
- **Relative Descriptors**: Formulated strictly with non-causal comparative language (e.g., "opening attention signal is elevated relative to the video median", "prominent relative peak occurs mid-video at 9.00s").

---

## 5. Spatial Intelligence
Derived interpretable spatial visual-saliency properties from TinySalNet:
- **Focal Centroid**: Center of mass coordinates $(c_x, c_y)$.
- **Peak Focus**: Location of maximal saliency intensity $(p_x, p_y)$.
- **Spatial Dispersion**: Raw RMS radial distance from centroid (preserved without artificial scaling constants).
- **Spatial Entropy**: Shannon entropy in bits quantifying density spread.
- **Temporal Centroid Drift**: Frame-to-frame Euclidean distance tracking spatial stability.
- **Focal Concentration**: Inverted dispersion index $1.0 - (\text{Dispersion} / 0.6)$ bounded in $[0, 1]$.

---

## 6. Spatial × Temporal Relationships
Synthesized cross-modal dynamics connecting temporal attention potential with spatial visual saliency:
1. **Stable Focal Attention**: Elevated temporal attention coincides with concentrated, stable spatial saliency.
2. **Moving Focal Attention**: Elevated temporal attention coincides with active spatial centroid displacement.
3. **Diffuse Unanchored Opening**: Moderate opening temporal attention coincides with diffuse spatial saliency.
4. **Text-Saliency Interaction**: Evaluates whether OCR text coincides with lower-third, upper banner, or central saliency concentration.
5. **Face-Saliency Interaction**: Evaluates whether human facial presence aligns with dominant spatial saliency.
6. **Scene-Transition Alignment**: Quantifies inter-frame saliency shift across cut transitions.

---

## 7. Focal Competition
Implemented principled focal competition analysis derived from the empirical density distribution:
- **Quadrant Mass Distribution**: Evaluates dominant quadrant mass ($M_1$), secondary quadrant mass ($M_2$), and concentration ratio $R = M_1 / \max(0.01, M_2)$.
- **Structured Categories**:
  - `single_focus`: $R \ge 2.2$ and Dispersion $< 0.28$.
  - `moderately_distributed`: $R \ge 1.5$ and Dispersion $< 0.35$.
  - `multi_focus`: $M_2 \ge 0.28$ and Dispersion $\ge 0.32$.
  - `diffuse`: Uniform or unanchored spread.
- Reports clear empirical rationales detailing why a scene is classified into a given state.

---

## 8. Attention Events
Created structured attention events with complete provenance, evidence, and non-causal interpretations:
- Every event records `event_type`, `timestamp_start`, `timestamp_end`, `evidence` dictionary, `source_signals` array, grounded `confidence` ($0.50 - 0.90$), and an `interpretation` string.
- Confidence strictly reflects signal clarity and feature alignment; zero fabricated statistical certainty (no 95%+ claims without empirical ground truth).

---

## 9. Pattern Agent Integration
Extended `PatternAgent` in `agents/pattern_agent.py`:
- Added pattern indicators: `concentrated_opening_spatial_focus` and `stable_focal_attention`.
- Strictly preserved subgroup hierarchy and denominator accounting:
  - `same_entity_and_edit_type` (primary subgroup)
  - `highest_reach_same_edit_type`
  - `all_verified` (broader market context)
- Clear target-vs-competitor distinction without extrapolating population-wide Instagram behavior.

---

## 10. Recommendation Integration
Extended `StrategyAgent` in `agents/strategy_agent.py`:
- Formulates evidence-grounded strategic action items citing Attention Intelligence.
- Preserves the structured schema: `observation`, `evidence`, `comparison`, `recommendation`, `confidence`, and `limitation`.
- Employs non-causal language (e.g., "Consider keeping the primary subject visually dominant during the opening segment to establish an unmistakable focal anchor; model-derived spatial saliency measures visual conspicuity under laboratory video-viewing conditions and does not measure individual viewer scroll-stop decisions").

---

## 11. Visualization
Implemented `render_attention_visualization`:
- Selects a representative frame (at peak attention timestamp or opening window).
- Applies a soft perceptual heatmap overlay (`COLORMAP_TURBO`, alpha=0.32) that does not obscure underlying video action or facial expressions.
- Annotates Centroid (green circle) and Peak Focus (crosshair).
- Embeds a compact lower banner card summarizing timestamp, peak/opening potential, focus state, and dispersion.

---

## 12. Performance
Benchmarked decoupled latencies across operations:

| Latency Category | Definition | Measured Latency (`htdyr`) | Measured Latency (`upload_jvxawbcy`) | Criterion | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A. Pure Attention Intelligence** | In-memory synthesis on pre-extracted signals | **4.037 ms** | **2.206 ms** | $< 50\text{ ms}$ | **PASS** |
| **B. JSON Serialization** | Schema serialization to JSON | 0.444 ms | 0.392 ms | — | **PASS** |
| **C. Visualization Rendering** | Frame decoding, overlay blend, PNG save | 186.932 ms | 284.444 ms | Async / Report time | **PASS** |
| **D. Total Integration Overhead** | End-to-end service execution | 191.413 ms | 287.042 ms | Sub-second | **PASS** |

- **Zero Duplicate Inference**: Neither NEMAR nor TinySalNet is re-invoked.
- **Zero Discovery Overhead**: Instagram search queries, network calls, and candidate ranking are completely unaffected.

---

## 13. Quality Invariance
- **Frozen Models**:
  - TinySalNet SHA-256: `fd6f27fd5ef6334c597ee3981086fc70f639cd59c7d57399fa72253d883b95c9` verified identical.
  - NEMAR Random Forest Regressor (100 estimators, 14 multimodal features) verified identical.
- **Production Pipeline Invariance**:
  - Discovery queries generated: 100% identical.
  - Apify candidate pool & deduplication: 100% identical.
  - Candidate ranking & selection: 100% identical.
  - Reach ordering: strictly descending, 100% identical.

---

## 14. Real-Reel Validation
Validated on both verified benchmark video files:
1. `E:\new viral\htdyr.mp4` (11,300,630 bytes):
   - NEMAR: 55 windows in 9.07s
   - TinySalNet: 15 frames in 1.19s
   - Pure Intelligence: **4.037 ms** (PASS)
   - Events: 6 events detected
   - Focal State: `moderately_distributed` (ratio: 1.50, dispersion: 0.233)
   - Report: `virallens_report_phase14_htdyr_20260916_101852.json` and `.md`
2. `C:\Users\vishn\AppData\Local\Temp\virallens_uploads\upload_jvxawbcy.mp4` (42,280,809 bytes):
   - NEMAR: 53 windows in 25.04s
   - TinySalNet: 15 frames in 2.68s
   - Pure Intelligence: **2.206 ms** (PASS)
   - Events: 6 events detected
   - Focal State: `single_focus` (ratio: 20.00, dispersion: 0.266)
   - Report: `virallens_report_phase14_upload_jvxawbcy_20260916_101920.json` and `.md`

---

## 15. Tests
- Created `tests/test_attention_intelligence.py` covering all 22 required test cases:
  1. Canonical schema creation & validation
  2. NEMAR provenance
  3. TinySalNet provenance
  4. Temporal peak extraction
  5. Spatial concentration derivation
  6. Spatial dispersion validation
  7. Entropy handling
  8. Focal competition categorization
  9. Centroid movement & drift
  10. Attention event detection & structured fields
  11. Insufficient-data handling
  12. Text interaction (OCR vs saliency)
  13. Face interaction (Face presence vs saliency)
  14. Provenance preservation end-to-end
  15. No fabricated statistical confidence
  16. No causal language validation
  17. Report serialization (JSON & Markdown)
  18. Pattern Agent compatibility & subgroup denominator accounting
  19. Recommendation provenance & evidence linkage
  20. Production output invariance (discovery & ranking untouched)
  21. No duplicate model inference
  22. Performance regression check ($< 50\text{ ms}$ intelligence overhead)
- Result: **22 passed in 0.94s**.

---

## 16. Limitations
- **Perceptual Gaze Benchmark**: NEMAR predicts potential eye-tracking fixation density based on laboratory stimuli; it does not measure active user navigation or scroll-stop behavior on mobile apps.
- **Synthetic Saliency Supervision**: TinySalNet predicts bottom-up spatial visual conspicuity trained on DHF1K video frames; it does not model individual viewer intent or cognitive goals.
- **Non-Causal Interpretation**: All findings represent descriptive correlations and qualitative visual characteristics rather than guarantees of commercial virality or retention.

---

## 17. Final Status
**PHASE 14 COMPLETE**
All criteria satisfied; all models remain 100% frozen; test suite passing with zero regressions.
