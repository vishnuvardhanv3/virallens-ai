# ViralLens Attention Intelligence: Scientific & Engineering Protocol

## 1. Overview and Objective
The Attention Intelligence layer unifies two empirically validated, frozen machine-learning signals within ViralLens:
1. **NEMAR Temporal Signal**: Laboratory-derived visual-attention potential trained on eye-tracking fixation densities from the NEMAR BBBD dataset (Random Forest regressor, 100 estimators, 14 multimodal features).
2. **TinySalNet Spatial Signal**: Deep spatial visual-saliency model trained under continuous Kullback-Leibler divergence supervision on the DHF1K gaze fixation dataset (96,521 parameters, CPU inference, eval mode).

This layer transforms raw model outputs into structured, interpretable creative-intelligence telemetry. It operates strictly downstream of signal extraction without modifying model weights, altering competitor discovery, or injecting causal virality claims.

---

## 2. Non-Negotiable Scientific & Architectural Rules
- **Frozen Models**: Zero retraining, weight updates, or recalibration.
- **Strict Decoupling**: NEMAR and TinySalNet remain separately identifiable internally with distinct provenance records.
- **Zero Fabrication**: No synthetic gaze coordinates, no arbitrary engagement percentages, and no false statistical confidence intervals.
- **Scientific Language Policy**: Strictly descriptive and observational terminology:
  - *Permitted*: "predicted visual saliency is concentrated around...", "reaches a relative temporal peak...", "saliency centroid shifts...", "coincides with...", "the available signals suggest...".
  - *Prohibited*: "causes retention", "makes viewers watch longer", "guarantees virality", "scroll-stop predicted", "viewers definitely stayed engaged".
- **Distinction of Tiers**:
  - `MODEL OUTPUT`: Raw tensor/score predictions directly emitted by frozen weights.
  - `DERIVED METRIC`: Mathematical summaries computed deterministically from raw outputs (e.g. dispersion, entropy, drift).
  - `INTERPRETATION`: Grounded descriptive observations contextualized with semantic video features.
  - `LIMITATION`: Explicit scientific boundaries and disclaimers.

---

## 3. Canonical Attention Intelligence Schema

```
{
  "schema_version": "1.0.0",
  "status": "SUCCESS" | "PARTIAL" | "FAILED",
  "temporal": { ... },
  "spatial": { ... },
  "context": { ... },
  "relationships": { ... },
  "events": [ ... ],
  "interpretation": [ ... ],
  "limitations": [ ... ],
  "provenance": { ... },
  "telemetry": { ... }
}
```

### 3.1 Temporal Attention Subsystem (NEMAR)
- **Signal Source**: `NEMAR` (`laboratory_derived_attention_signal`)
- **Baseline Attention**: Mean attention across the opening segment (first 4 windows / 2.0s).
- **Peak Attention & Timestamp**: Highest predicted attention potential ($s_{peak}$) and its exact video timestamp ($t_{peak}$).
- **Relative Peak Position**: Normalized video coordinate $t_{peak} / t_{total} \in [0.0, 1.0]$.
- **Temporal Persistence**: Total duration and ratio of analyzed windows where attention exceeds the video median.
- **Modulation & Transitions**: Discrete drop and recovery events indexed by local delta exceeding $1.2 \times \sigma$ ($> 0.04$).
- **Descriptors**: Relative qualitative indicators comparing opening vs. body and trajectory stability.

### 3.2 Spatial Saliency Subsystem (TinySalNet)
- **Signal Source**: `TinySalNet_DHF1K` (`predicted_visual_saliency`)
- **Focal Centroid**: Normalized coordinates $(c_x, c_y) = \left(\sum x \cdot S(x, y), \sum y \cdot S(x, y)\right)$.
- **Peak Focus**: Normalized coordinate $(p_x, p_y) = \arg\max_{(x, y)} S(x, y)$.
- **Spatial Dispersion**: RMS radial distance from centroid:
  $$\text{Dispersion} = \sqrt{\sum \left((x - c_x)^2 + (y - c_y)^2\right) \cdot S(x, y)}$$
- **Spatial Entropy**: Shannon entropy in bits:
  $$H = -\sum S(x, y) \log_2 S(x, y)$$
- **Temporal Centroid Drift**: Frame-to-frame Euclidean displacement of the centroid:
  $$\Delta_{drift} = \frac{1}{N-1} \sum_{i=1}^{N-1} \sqrt{(c_{x, i} - c_{x, i-1})^2 + (c_{y, i} - c_{y, i-1})^2}$$
- **Focal Concentration**: Inverted dispersion index $1.0 - (\text{Dispersion} / 0.6)$ clamped to $[0.0, 1.0]$.

### 3.3 Principled Focal Competition Analysis
Visual fields are categorized based on spatial density distribution across quadrants:
- **Dominant Region Mass ($M_1$)**: Fraction of total saliency mass in the highest-density quadrant.
- **Secondary Region Mass ($M_2$)**: Fraction of total saliency mass in the second highest-density quadrant.
- **Concentration Ratio ($R$)**: $M_1 / \max(0.01, M_2)$.

| Category | Empirical Criteria | Rationale |
| :--- | :--- | :--- |
| **`single_focus`** | $R \ge 2.2$ and Dispersion $< 0.28$ | Clear dominant visual anchor concentrating the majority of visual mass. |
| **`moderately_distributed`** | $R \ge 1.5$ and Dispersion $< 0.35$ | Primary focal region leads secondary region with moderate spatial concentration. |
| **`multi_focus`** | $M_2 \ge 0.28$ and Dispersion $\ge 0.32$ | Saliency mass is divided across competing focal regions in multiple quadrants. |
| **`diffuse`** | All other conditions | Saliency is widely distributed across the frame without a clear dominant locus. |

---

## 4. Cross-Modal Relationships & Dynamics
Combines temporal potential and spatial saliency to characterize creative video execution:
1. **Stable Focal Attention**: Temporal attention remains elevated while spatial focus maintains a concentrated, stable position.
2. **Moving Focal Attention**: Temporal attention remains elevated while the saliency centroid shifts actively across frames.
3. **Diffuse Unanchored Opening**: Opening temporal attention is moderate with diffuse spatial saliency.
4. **Text-Saliency Interaction**: Evaluates whether OCR typography occupies salient screen regions (e.g., lower-third subtitles vs. upper banner headlines).
5. **Face-Saliency Interaction**: Evaluates whether detected human facial presence coincides with concentrated spatial saliency.
6. **Scene-Transition Alignment**: Quantifies inter-frame saliency shifts across cut transitions.

---

## 5. Structured Attention Events
Events are discrete, evidence-backed temporal markers:
- `OPENING_FOCUS` (0.0s–2.0s baseline setup)
- `ATTENTION_PEAK` (Temporal maximum potential window)
- `ATTENTION_DROP` (Sudden relative drop in attention potential)
- `ATTENTION_RECOVERY` (Re-escalation of attention potential)
- `SPATIAL_FOCUS_SHIFT` (Marked movement of visual focus)
- `SCENE_TRANSITION_ALIGNMENT` (Reorganization across cuts)
- `FOCAL_COMPETITION` (Multi-focus or diffuse visual field)
- `TEXT_SALIENCY_INTERACTION` (Typography entrance and saliency overlap)
- `FACE_SALIENCY_ALIGNMENT` (Facial visual dominance)

Each event contains: `event_type`, `timestamp_start`, `timestamp_end`, `evidence`, `source_signals`, `confidence`, and `interpretation`.

---

## 6. Visualization Standards
- Renders representative video frames with soft perceptual saliency overlays (`COLORMAP_TURBO`, alpha=0.32) to ensure the underlying footage is never masked.
- Clearly displays Centroid (green circle) and Peak Focus (crosshair) coordinates.
- Attaches an informational telemetry banner displaying timestamp, peak/opening scores, focal state, and dispersion.

---

## 7. Decoupled Metric Reporting
Latency is tracked across four decoupled operational boundaries:
- **A. Pure Attention Intelligence**: Signal transformation in memory ($< 50\text{ ms}$ criterion).
- **B. Serialization**: Conversion to JSON string/dictionary.
- **C. Visualization**: OpenCV frame rendering and disk write.
- **D. Total Integration Overhead**: End-to-end service execution.
