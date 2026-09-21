# Phase 15 — ViralLens Competitor Intelligence & Pattern Engine Report

**Date**: 2026-09-16  
**Status**: COMPLETE  
**Repository**: `E:\new viral`  

---

## 1. Objective

Upgrade the existing ViralLens competitor-analysis and Pattern Agent into a rigorous, evidence-backed **Competitor Intelligence Engine**. The engine compares the uploaded target Reel against the discovered, normalized, selected, downloaded, and Twelve-Labs-verified competitor Reels to identify recurring observable characteristics.

The engine specifically answers:
> *"What observable characteristics recur within the verified comparison group, and how does the target Reel compare with those characteristics?"*

### Hard Epistemic Guardrails:
- Strictly non-causal descriptive findings.
- Zero claims of "what makes a Reel go viral".
- Zero claims of guaranteed engagement or retention.
- Zero claims of commercial Instagram scroll-stop behavior or viewer psychological states.
- Zero claims of population-wide Instagram algorithm rules.

---

## 2. Existing Architecture & Production Invariance

The pipeline preserves all upstream stages without modification:

```
Target Video Upload
      ↓
Twelve Labs Multimodal Analysis + Technical Signals Extraction
      ↓
NEMAR Temporal Attention (frozen RF) + TinySalNet Spatial Saliency (frozen ConvNet)
      ↓
Attention Intelligence Integration (Phase 14)
      ↓
Niche-Anchored Discovery Profile & Query Builder
      ↓
Apify Instagram Discovery
      ↓
Candidate Normalization & Deduplication
      ↓
Reach Ranking & Creator-Diversity Selection (max 2/creator)
      ↓
MP4 Download & Twelve Labs Semantic Verification
      ↓
Canonical Competitor Features
      ↓
[PHASE 15 COMPETITOR INTELLIGENCE SERVICE]
      ↓
Pattern Agent (Exact empirical tallies)
      ↓
Strategy Agent (5-part evidence-grounded recommendations)
      ↓
Production Markdown/JSON Audit Report & Streamlit UI
```

**Production Invariance Verified**:
- Zero changes to Apify search queries or semantics.
- Zero changes to candidate ranking or selection logic.
- Zero changes to creator diversity caps (strictly max 2/creator).
- Zero changes to Twelve Labs verification thresholds.
- Zero new ML models or retrained weights.

---

## 3. Canonical Schema

Implemented in `services/competitor_intelligence.py` and validated against `models/competitor_intelligence/competitor_intelligence_schema.json`.

### A. Normalized Target Schema
Preserves field-level provenance across:
- `semantic`: entity, niche, sub_niche, topic, edit_type, content_format, style, hook_type.
- `video`: duration, resolution, fps.
- `visual_subjects`: Twelve Labs visual subjects.
- `ocr`: text, timing, text_present.
- `audio_speech`: speech_present, transcript, RMS, silence ratio.
- `temporal_attention`: NEMAR attention timeline.
- `spatial_saliency`: TinySalNet spatial dispersion and entropy.
- `attention_intelligence`: Phase 14 unified temporal-spatial intelligence.

### B. Normalized Competitor Schema
Every verified competitor is transformed into standard representation:
- `identity`: `canonical_reel_id`, `canonical_url`, `creator`, `source_queries`.
- `reach`: `views`, `plays`, `reach_source`, `reach_valid` (if missing, `reach = null`, `reach_valid = False`).
- `semantic`: `entity`, `niche`, `sub_niche`, `topic`, `edit_type`, `content_format`.
- `video`: `duration`, `resolution`, `fps`.
- `visual`: `mean_motion`, `scene_cuts`, `brightness`, `contrast`, `complexity`, `face_presence`, `text_presence`.
- `audio`: `rms`, `silence_ratio`, `speech_presence`, `spectral_characteristics`.
- `attention`: `nemar_temporal_signal`, `tinysalnet_spatial_signal`, `attention_intelligence`.
- `ocr`: `text_segments`, `confidence`, `timing`, `text_present`.
- `relevance`: `entity_match`, `topic_match`, `niche_match`, `sub_niche_match`, `edit_type_match`, `format_match`, `overall_relevance`.
- `provenance`: `canonical_reel_id`, `source_provider`, signal presence flags.

---

## 4. Comparison Groups

The engine partitions verified competitors into 4 canonical cohorts:

1. **`same_entity_and_edit_type`** (Primary Comparison Group): Matches target entity (with token overlap guardrails) and edit type.
2. **`same_edit_type`**: Broader peer group matching the creative edit style.
3. **`all_verified`**: All verified competitors in the run.
4. **`highest_reach_same_edit_type`**: Top-50% reach partition of the primary pool, strictly excluding records where `reach_valid == False`.

Every finding explicitly reports its cohort, numerator, and denominator (e.g. `6/7` or `10/21`).

---

## 5. Feature Comparisons

The engine performs non-causal comparisons across:
- **Semantic features**: entity, niche, topic, edit type, format.
- **Continuous numeric features**: visual motion, scene cuts, spatial dispersion, peak attention score.
- **Robust Continuous Statistics**: computes `target_value`, `competitor_median`, `competitor_mean`, `absolute_difference`, `relative_difference`, and empirical `percentile` ($n \ge 3$).
- Zero arbitrary virality scores or manufactured p-values.

---

## 6. Attention Comparisons

Consumes Phase 14 Attention Intelligence across 10 observable patterns:
1. `concentrated_opening_spatial_focus`
2. `stable_focal_attention`
3. `opening_temporal_attention_prominence`
4. `spatial_focus_movement`
5. `focal_competition`
6. `text_saliency_alignment`
7. `face_saliency_alignment`
8. `scene_transition_attention_alignment`
9. `temporal_recovery`
10. `temporal_attention_drop`

---

## 7. Pattern Detection & Deduplication

Recurring categorical patterns detected:
- Opening visual focal anchor
- Early typography hook
- Early human presence
- Kinetic cut pacing
- Concentrated opening spatial saliency
- Distributed spatial visual composition
- Spoken verbal opening
- Dynamic opening visual motion
- Primary entity subject continuity

**Deduplication**: Semantic aliases (e.g., `opening_visual_concentration` $\to$ `concentrated_opening_spatial_focus`) are consolidated under canonical IDs.

---

## 8. Target vs Competitor Gaps

For every dominant recurring pattern ($\ge 50\%$ prevalence in subgroup), the engine evaluates target status:
- **`aligned`**: Target exhibits the pattern present in the peer majority.
- **`divergent`**: Target differs from observed peer majority (e.g., "The target exhibits 'text_led_opening' as absent, differing from 6/7 (85.7%) of the verified comparison cohort").

---

## 9. Reach-Aware Observations

- Uses raw views/plays when available.
- Missing reach handled safely: `reach = null`, `reach_valid = False`, excluded from reach statistics.
- Stratifies cohort into top-reach and lower-reach partitions to report descriptive differences in typography, human presence, and pacing.
- Strictly non-causal: Never attributes reach outcomes to individual features.

---

## 10. Small-Sample Safeguards

Strict descriptive strength thresholds:
- $n < 5$: `insufficient_sample` / `exploratory_observation` (no strong pattern claims permitted).
- $5 \le n < 10$: `moderate_observed_pattern` or `weak_observed_pattern` with explicit cohort size warnings.
- $n \ge 10$ with $\ge 70\%$ prevalence: `strong_observed_pattern`.
- Numerator and denominator always preserved visibly.

---

## 11. Complete Provenance

Every pattern is traceable back to:
- Target Reel ID
- Exact list of matching competitor shortcodes/IDs
- Subgroup sample size ($n$)
- Underlying feature fields and signal sources

---

## 12. Strategy Agent Integration

`StrategyAgent` formulates recommendations strictly following the 5-part schema:
1. **OBSERVATION**: Target observed state.
2. **EVIDENCE**: Subgroup numerator/denominator tally.
3. **TARGET/COMPETITOR COMPARISON**: Descriptive gap assessment.
4. **RECOMMENDATION**: Actionable production refinement.
5. **LIMITATION**: Scientific epistemic boundaries.

---

## 13. UI Integration

Enhanced Section 6 of `app/streamlit_app.py`:
- Compact subgroup counts badges (`same_entity_and_edit_type`, `same_edit_type`, `all_verified`, `highest_reach_same_edit_type`).
- Pattern cards displaying exact counts, denominators, and target alignment badges.
- Collapsible gap analysis and evidence-backed creative implications.

---

## 14. Performance & Decoupled Latencies

Measured on real Reel MP4s using 32 cached verified competitors:

| Metric | `htdyr.mp4` | `upload_jvxawbcy.mp4` | Production Standard |
| :--- | :---: | :---: | :---: |
| **A. Intelligence Computation** | **2.920 ms** | **2.136 ms** | In-memory analytical layer |
| **B. Serialization (JSON)** | **8.935 ms** | **8.301 ms** | Fast schema serialization |
| **C. Report Generation (Markdown)** | **0.215 ms** | **0.246 ms** | Audit generation |
| **Total Engine Overhead** | **12.070 ms** | **10.683 ms** | Zero regression |

---

## 15. Regression Suite

Run command: `.venv\Scripts\pytest -q`

| Test Suite | Tests Passed | Status |
| :--- | :---: | :---: |
| `tests/test_competitor_intelligence.py` | 25 passed | PASS |
| `tests/test_attention_intelligence.py` | 22 passed | PASS |
| `tests/test_product_utility_eval.py` | 14 passed | PASS |
| `tests/test_real_reel_human_eval.py` | 15 passed | PASS |
| `tests/test_real_reel_shadow_eval.py` | 25 passed | PASS |
| `tests/test_shadow_mode_spatial.py` | 17 passed | PASS |
| `tests/test_human_attention.py` | 31 passed | PASS |
| `tests/test_performance_optimization.py` | 7 passed | PASS |
| `tests/test_phase13_performance.py` | 13 passed | PASS |
| **Total Test Suite** | **364 passed** | **PASS** |

---

## 16. Limitations

1. **Observational Scope**: Findings reflect characteristics observed within the discovered, verified sample pool and do not represent the entire Instagram distribution graph.
2. **Non-Causality**: Observed peer conventions do not guarantee video virality, audience retention, or algorithmic distribution.
3. **Reach Reporting**: View counts represent historical external metrics influenced by account authority, sound trends, and platform promotion.

---

## 17. Final Status

**PHASE 15 COMPLETE**  
Zero new ML models, zero retraining, zero Apify/Twelve Labs replacements, full evidence-backed Competitor Intelligence operational.
