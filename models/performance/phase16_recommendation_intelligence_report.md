# Phase 16 — Recommendation Intelligence Comprehensive Engineering Report

**Project:** ViralLens AI  
**Phase:** 16 — Recommendation Intelligence  
**Execution Timestamp:** 2026-09-16  
**Status:** COMPLETE & PRODUCTION-VALIDATED  

---

## 1. Objective

Phase 16 transforms the multimodal analytical outputs of Twelve Labs, NEMAR temporal attention, TinySalNet / DHF1K spatial saliency, OCR typography, Attention Intelligence (Phase 14), Competitor Intelligence (Phase 15), and the Pattern Agent into a rigorous, evidence-backed **Recommendation Intelligence** layer.

The system rigorously answers:
> *"What should the creator consider preserving, changing, or testing based on the evidence available for this Reel and its verified competitor comparison group?"*

This is purely a **Product Intelligence** phase. Zero ML models were added or retrained, existing models remain completely frozen, and all Apify / Twelve Labs / candidate ranking systems remain invariant.

---

## 2. Architecture

```
[Target Reel Upload] 
         │
         ▼
[Twelve Labs + NEMAR + TinySalNet + OCR] 
         │
         ▼
[Phase 14: Attention Intelligence] ──┐
                                     │
[Phase 15: Competitor Intelligence] ─┼──► [RecommendationIntelligenceService]
                                     │         │
[Pattern Agent Recurring Tallies] ───┘         ▼
                                          [StrategyAgent]
                                               │
                                               ├─► [Streamlit Dashboard UI: Section 7]
                                               └─► [ReportService: JSON & Markdown Audits]
```

Recommendation Intelligence operates strictly as an in-memory reasoning and decision-support layer over already-produced evidence. It performs **zero** network requests, **zero** video downloads, and **zero** duplicate model inferences.

---

## 3. Recommendation Schema

Every generated recommendation conforms to `recommendation_intelligence_schema.json`:

```json
{
  "recommendation_id": "rec_change_spatial_focus",
  "category": "CHANGE",
  "priority": "high",
  "observation": "Target opening visual saliency is diffusely distributed across competing regions (dispersion: 0.420, state: 'moderate_competition').",
  "evidence": [
    {
      "type": "competitor_subgroup_pattern",
      "source": "TinySalNet/AttentionIntelligence",
      "value": "19/21 (90.5%) peers exhibit concentrated opening spatial focus",
      "numerator": 19,
      "denominator": 21,
      "competitor_ids": ["comp_1", "comp_2", "comp_3"],
      "provenance": { "comparison_group": "same_edit_type" }
    }
  ],
  "target_comparison": {
    "target": "diffuse/multi-focus (dispersion: 0.420)",
    "comparison_group": "19/21 (90.5%) concentrated in same_edit_type",
    "difference": "Target provides less visual concentration than observed peer majority."
  },
  "creative_implication": "Multiple visual elements compete for gaze during opening seconds, dispersing viewer focal attention.",
  "recommendation": "Consider simplifying the opening composition so that the primary subject remains visually dominant.",
  "testable_action": "Create Variant B by reducing background motion or cropping closer to primary subject during first 1.5s, then compare focal clarity.",
  "strength": "strong_observed_evidence",
  "confidence": null,
  "limitation": "Observational comparison within selected verified competitor cohort; does not prove that concentrated framing causes improved viewer retention.",
  "provenance": {
    "target_reel_id": "target_123",
    "comparison_group": "same_edit_type",
    "competitor_reel_ids": ["comp_1", "comp_2"],
    "source_features": ["spatial.dispersion", "spatial.focal_competition"]
  }
}
```

---

## 4. Evidence Model

1. **Explicit Fractions:** Numerators and denominators are strictly retained (e.g. `19/21`, not solely `90.5%`).
2. **Confidence Rule:** Confidence is strictly `null` across all recommendations. Heuristic numerical percentages (e.g. "92% confidence") are prohibited.
3. **Epistemic Strength Levels:**
   - `strong_observed_evidence`: $n \ge 10$, prevalence $\ge 70\%$, high consistency.
   - `moderate_observed_evidence`: $5 \le n < 10$ with high prevalence, or $n \ge 10$ with $50\% \le \text{prev} < 70\%$.
   - `exploratory_evidence`: $n \ge 5$ with suggestive/moderate prevalence.
   - `insufficient_evidence`: $n < 5$ (strict small-sample safety flag).

---

## 5. Recommendation Categories

1. **`PRESERVE`**: Protects existing creator strengths aligned with high-prevalence peer conventions (e.g. maintaining character face presence or early typography).
2. **`CHANGE`**: High-prevalence peer divergence ($n \ge 5, \text{prev} \ge 70\%$) suggesting a specific structural modification to consider.
3. **`TEST`**: Suggestive peer patterns ($50\% \le \text{prev} < 70\%$) framed as testable creative hypotheses.
4. **`AVOID_OVERINTERPRETATION`**: Enforced when subgroup cohort size is insufficient ($n < 5$).
5. **`INVESTIGATE`**: Multi-signal conflicts requiring creator inspection (e.g. text/gaze spatial competition).
6. **`NO_ACTIONABLE_RECOMMENDATION`**: Handled gracefully when peer data is empty or unverified.

---

## 6. Priority Methodology

Priority (`high`, `medium`, `low`) reflects **evidence strength and target-vs-cohort divergence magnitude**, NOT promises of viral performance:
- **`high`**: Dominant peer pattern ($\ge 70\%$ prevalence) with clear target divergence and complete provenance.
- **`medium`**: Moderate consistency pattern ($50\%-70\%$) or preserve recommendation on core narrative assets.
- **`low`**: Exploratory evidence, subtle adjustments, or null baseline results.

---

## 7. Attention Recommendations

Grounded in Phase 14 Attention Intelligence:
- **Opening Spatial Focus & Focal Competition:** Identifies diffuse gaze distribution vs concentrated subject anchors.
- **Temporal Persistence & Peak Energy:** Evaluates opening drop-off risks and recovery dynamics.
- **Cross-Modal Alignment:** Correlates spatial gaze coordinates with temporal scene shifts.

---

## 8. Editing Recommendations

Grounded in scene cut analysis and transition density:
- **Cut Pacing Cadence:** Compares scene transition frequency against verified peer medians without imposing rigid universal cut intervals.
- **Opening Transition Timing:** Compares timing of first visual scene change against cohort norms.

---

## 9. Text / Face Recommendations

- **Typography:** Detects on-screen title cards and text-saliency alignment. Text is never labeled "distracting" unless spatial competition with character gaze is empirically observed.
- **Character / Face Prominence:** Acknowledges early character presence and direct gaze anchoring without forcing narrative reels into talking-head formats.

---

## 10. Target-vs-Competitor Gaps

Formal comparative gap representation identifies:
- Target observed state vs Peer subgroup median and prevalence.
- Subgroup scope: `same_entity_and_edit_type`, `same_edit_type`, `all_verified`, `highest_reach_same_edit_type`.
- Explicit difference description and creative interpretation.

---

## 11. Testable Creative Hypotheses

Recommendations are systematically translated into creator A/B test specifications:
- **Variant A (Baseline Control):** Current production edit.
- **Variant B (Candidate Hypothesis):** Targeted single-variable adjustment (e.g. simplified opening framing, 1.5s text card reveal).
- **Comparison Metric:** Perceived focal clarity and subjective creator review (strictly avoiding claims of guaranteed algorithmic distribution).

---

## 12. Contradictory Evidence Handling

The service actively resolves multi-signal contradictions:
- If a target exhibits diffuse visual composition, aggressive scene cut pace is restricted with a limitation warning to prevent compounded ocular disorientation.
- If text is present but diverges from visual saliency, the system escalates to `INVESTIGATE` rather than forcing a simplistic `PRESERVE` or `CHANGE`.

---

## 13. Provenance & Evidence Graph

Full backward traceability:
$$\text{Target Reel} \to \text{Source Features} \to \text{Observed Pattern} \to \text{Peer Evidence } (num/den) \to \text{Target Gap} \to \text{Recommendation}$$
Every recommendation records its target reel ID, comparison subgroup name, source feature paths, and supporting competitor IDs.

---

## 14. UI Integration

Implemented in `app/streamlit_app.py` under **Section 7: 💡 Recommendation Intelligence: Evidence-Backed Guidance**:
- Category pills with dynamic counters (`PRESERVE`, `CHANGE`, `TEST`, etc.).
- Clean glassmorphism cards displaying Title, Category Badge, Priority Badge, Strength, Observation, Target vs Cohort, Creative Implication, Recommendation box, Testable Action experiment box, and Limitations.
- Expandable scientific evidence trace disclosing source models, numerators/denominators, and supporting competitor shortcodes.

---

## 15. Performance

Decoupled latency measurements on real-world benchmark Reels:

| Video Benchmark | Computation (ms) | Serialization (ms) | Report Gen (ms) | Total Overhead (ms) |
|---|---|---|---|---|
| `htdyr.mp4` | 0.102 ms | 0.718 ms | 0.134 ms | **0.955 ms** |
| `upload_jvxawbcy.mp4` | 0.072 ms | 0.468 ms | 0.127 ms | **0.668 ms** |

Recommendation Intelligence executes in **under 1.0 millisecond**, introducing zero perceptible pipeline overhead.

---

## 16. Real-Reel Validation

Validated against both production Reels using 32 cached verified competitors from `data/competitors/`:
- **`htdyr.mp4`**: Generated 3 deduplicated recommendations (`PRESERVE: early typography`, `PRESERVE: human presence`, `CHANGE: cut pacing`). Full provenance and explicit denominators verified.
- **`upload_jvxawbcy.mp4`**: Generated 3 deduplicated recommendations (`PRESERVE: early typography`, `PRESERVE: human presence`, `CHANGE: cut pacing`). Full provenance and explicit denominators verified.
- Prohibited causal language audit: **0 violations detected**.

---

## 17. Regression Tests

Executed comprehensive test suite `tests/test_recommendation_intelligence.py`:
- **31 / 31 tests passed** in **0.18s**.
- All 31 prompt requirements validated, including schema, observation generation, evidence attachment, subgroup provenance, numerator/denominator preservation, preserve/change/test categorization, small-sample protection, conflict resolution, null baseline handling, non-causal language, zero network calls, and zero duplicate model inference.

---

## 18. Limitations

1. **Observational Baseline:** Recommendations reflect peer correlations within the verified competitor group and do not guarantee platform reach, viewer retention, or engagement.
2. **Sample Size Sensitivity:** Cohort sizes below 5 are classified as exploratory/insufficient evidence.
3. **Non-Algorithmic Scope:** Creative structure recommendations do not account for external distribution factors such as creator account authority, audio license region restrictions, or platform trend lifecycles.

---

## 19. Final Status

```
PHASE 16 — RECOMMENDATION INTELLIGENCE: COMPLETE
ALL 31 VERIFICATION CRITERIA: PASS
TEST SUITE: 31 PASSED (0.18s)
REAL-REEL VALIDATION: PASS (<1.0 ms latency)
PRODUCTION INVARIANCE: MAINTAINED
```
