# Phase 15 — Competitor Intelligence Engine Protocol

## 1. Architectural Role & Pipeline Position

The **Competitor Intelligence Engine** operates strictly as an in-memory, post-verification analytical synthesis layer inside ViralLens.

```
Uploaded Reel
      ↓
Twelve Labs + Technical Signals + NEMAR + TinySalNet
      ↓
Attention Intelligence (Phase 14)
      ↓
Discovery Profile & Apify Search
      ↓
Candidate Normalization & Deduplication
      ↓
Reach-Aware Ranking & Creator Diversity Selection (max 2/creator)
      ↓
Download & Twelve Labs Semantic Verification
      ↓
[PHASE 15 COMPETITOR INTELLIGENCE ENGINE]
      ↓
Pattern Agent (Exact empirical recurring tallies)
      ↓
Strategy Agent (Evidence-backed 5-part recommendations)
      ↓
Production Audit Report & Streamlit UI
```

---

## 2. Hard Constraints & Scientific Invariants

1. **Frozen Upstream Models**:
   - Zero ML additions, zero weight modifications.
   - NEMAR and TinySalNet remain frozen and untracked by new inference.
   - Zero modifications to Twelve Labs or Apify discovery pipelines.
2. **Zero Duplicate Inference / Zero Network Calls**:
   - Operates entirely in memory over cached canonical outputs.
   - Overhead must remain < 50 ms for analytical computation.
3. **Strict Non-Causal Epistemic Guardrails**:
   - Never claim "what makes a Reel go viral", "guaranteed engagement", "causes retention", "scroll-stop cause", or "viewer psychological state".
   - Findings represent empirical, observable characteristics within a verified sample cohort.
4. **Explicit Denominator Accounting**:
   - Never report "Most competitors..." without stating exact count, denominator, subgroup, and provenance (e.g. `6/7 reels in subgroup 'same_entity_and_edit_type'`).
5. **Small-Sample Safety Safeguards**:
   - For $n < 5$: strength is classified as `insufficient_sample` / `exploratory_observation`.
   - For $5 \le n < 10$: strength is classified as `moderate_observed_pattern` or `weak_observed_pattern` with explicit sample limitation warnings.
   - For $n \ge 10$ with $\ge 70\%$ prevalence: classified as `strong_observed_pattern`.
6. **Missing Reach Handling**:
   - If views/plays are missing or unparseable, set `reach = null`, `reach_valid = False`, and exclude from reach-stratified statistics. Never fabricate view counts.

---

## 3. Comparison Subgroup Hierarchy

1. **`same_entity_and_edit_type`**: Primary comparison cohort. Matches target entity (with token overlap guardrails) and edit type.
2. **`same_edit_type`**: Broad peer cohort matching target edit type (e.g., action montage, dialogue, emotional tribute).
3. **`all_verified`**: All Twelve-Labs-verified relevant competitors in the run.
4. **`highest_reach_same_edit_type`**: Top-50% reach partition of the primary pool, strictly requiring valid view counts (`reach_valid == True`).

---

## 4. Pattern Detection Specification

### A. Categorical Patterns
1. `opening_visual_focus`: Concentrated spatial focal anchor or early human presence.
2. `text_led_opening`: High-contrast on-screen typography within the opening 2.0 seconds.
3. `face_presence`: Human face / character presence in early attention windows.
4. `dynamic_transitions`: Dynamic cut pacing ($\ge 3$ scene transitions).
5. `concentrated_spatial_focus`: Opening visual saliency dispersion $< 0.32$ or concentration ratio $\ge 0.50$.
6. `distributed_spatial_focus`: Wide spatial distribution across visual field.
7. `spoken_opening`: Spoken verbal dialog/voiceover in opening segment.
8. `high_motion_opening`: Motion magnitude $> 0.03$ in opening 2.0s.
9. `visual_subject_consistency`: Persistent primary character/entity across scene cuts.

### B. Attention Intelligence Patterns (Phase 14 Integration)
1. `concentrated_opening_spatial_focus`: Unified spatial concentration.
2. `stable_focal_attention`: Spatial-temporal alignment on single subject anchor.
3. `opening_temporal_attention_prominence`: Peak attention or high retention ratio in opening 2.0s.
4. `spatial_focus_movement`: Dynamic or moderate saliency centroid drift.
5. `focal_competition`: Multi-focus or diffuse visual competition.
6. `text_saliency_alignment`: Congruent text-saliency alignment.
7. `face_saliency_alignment`: Human anchor focal alignment.
8. `scene_transition_attention_alignment`: Transition-driven attention surge or kinetic pacing.
9. `temporal_recovery`: Empirical post-dip attention recovery events.
10. `temporal_attention_drop`: Significant mid-reel attention drop events.

---

## 5. Strategy Recommendation Schema

Every action recommendation generated from competitor intelligence strictly implements the 5-part evidence schema:

1. **OBSERVATION**: Exact observed target state (e.g., "The target exhibits 'text_led_opening' as absent").
2. **EVIDENCE**: Explicit tally and denominator (e.g., "6/7 (85.7%) competitors in subgroup 'same_entity_and_edit_type' exhibit early typography").
3. **TARGET/COMPETITOR COMPARISON**: Descriptive gap assessment.
4. **RECOMMENDATION**: Concrete, non-causal creative suggestion (e.g., "Consider introducing clean on-screen title cards within the first 1.5 seconds").
5. **LIMITATION**: Formal scientific limitation (e.g., "Observational comparison within verified peer cohort; does not establish causality").
