# Phase 16 — Recommendation Intelligence Real-Reel Validation

**Execution Date:** 2026-09-16T21:21:00.483582  
**Verified Competitor Cohort:** 32 Reels from `data/competitors/`  

---

## 1. Benchmark Videos Validated

1. **`htdyr.mp4`**
   - **Path:** `E:\new viral\htdyr.mp4`
   - **Cohort Subgroup:** `same_entity_and_edit_type` (n=21)
   - **Recommendations Count:** 3
   - **Category Distribution:** {'PRESERVE': 1, 'CHANGE': 1, 'TEST': 1, 'AVOID_OVERINTERPRETATION': 0, 'INVESTIGATE': 0, 'NO_ACTIONABLE_RECOMMENDATION': 0}
   - **Decoupled Latencies:**
     - Computation: `0.102 ms`
     - Serialization: `0.718 ms`
     - Report Generation: `0.134 ms`
     - **Total Pipeline Addition:** `0.955 ms`

2. **`upload_jvxawbcy.mp4`**
   - **Path:** `C:\Users\vishn\AppData\Local\Temp\virallens_uploads\upload_jvxawbcy.mp4`
   - **Cohort Subgroup:** `same_entity_and_edit_type` (n=21)
   - **Recommendations Count:** 3
   - **Category Distribution:** {'PRESERVE': 0, 'CHANGE': 1, 'TEST': 2, 'AVOID_OVERINTERPRETATION': 0, 'INVESTIGATE': 0, 'NO_ACTIONABLE_RECOMMENDATION': 0}
   - **Decoupled Latencies:**
     - Computation: `0.072 ms`
     - Serialization: `0.468 ms`
     - Report Generation: `0.127 ms`
     - **Total Pipeline Addition:** `0.668 ms`

---

## 2. Guardrail & Epistemic Verification

- **Confidence Rule:** `confidence: null` across all generated items (zero fabricated percentages like '92%').
- **Explicit Denominators:** All supporting evidence records preserve explicit fractions (e.g. 19/21, 6/7, 8/10).
- **Non-Causal Language:** 0 occurrences of prohibited causal claims ("guaranteed engagement", "causes virality", "will improve retention", "scroll-stop prediction").
- **External Network Calls:** 0 network requests; pure in-memory execution.
- **Model Invariance:** Zero duplicate ML inferences (NEMAR and TinySalNet remain frozen and uncalled).
- **Report Integration:** `# Recommendation Intelligence` Markdown section generated with all 6 required subsections.

---

## 3. Sample Recommendations from Verified Real Reels

### `htdyr.mp4`
```json
[
  {
    "id": "rec_change_early_typography",
    "category": "CHANGE",
    "priority": "high",
    "strength": "strong_observed_evidence",
    "observation": "No readable on-screen typography was detected within the opening 2.0 seconds.",
    "recommendation": "Consider introducing a punchy on-screen title card or key dialogue quote within the opening 1.5 seconds.",
    "testable_action": "Create Variant B with a 2-to-4 word hook title card in the opening 1.0s and test comprehension in review.",
    "limitation": "Text overlay value depends heavily on font legibility, contrast against background, and safe-zone compliance.",
    "evidence_sample": "19/21 (90.5%) peers utilize on-screen typography within the first 2.0 seconds"
  },
  {
    "id": "rec_preserve_spatial_focus",
    "category": "PRESERVE",
    "priority": "medium",
    "strength": "exploratory_evidence",
    "observation": "Target exhibits concentrated opening visual focus (dispersion: 0.280, focal concentration: 0.62).",
    "recommendation": "PRESERVE: Maintain the current concentrated visual framing around the primary subject in the opening 1.5 seconds.",
    "testable_action": "Keep the existing opening framing intact as the baseline control for creative variations.",
    "limitation": "Laboratory-derived gaze prediction measures visual conspicuity and does not guarantee mobile retention.",
    "evidence_sample": "0/21 (0.0%) peers exhibit concentrated opening spatial focus"
  }
]
```

### `upload_jvxawbcy.mp4`
```json
[
  {
    "id": "rec_change_early_typography",
    "category": "CHANGE",
    "priority": "high",
    "strength": "strong_observed_evidence",
    "observation": "No readable on-screen typography was detected within the opening 2.0 seconds.",
    "recommendation": "Consider introducing a punchy on-screen title card or key dialogue quote within the opening 1.5 seconds.",
    "testable_action": "Create Variant B with a 2-to-4 word hook title card in the opening 1.0s and test comprehension in review.",
    "limitation": "Text overlay value depends heavily on font legibility, contrast against background, and safe-zone compliance.",
    "evidence_sample": "19/21 (90.5%) peers utilize on-screen typography within the first 2.0 seconds"
  },
  {
    "id": "rec_change_spatial_focus",
    "category": "TEST",
    "priority": "medium",
    "strength": "exploratory_evidence",
    "observation": "Target opening visual saliency is diffusely distributed across competing regions (dispersion: 0.420, state: 'moderate_competition').",
    "recommendation": "Consider simplifying the opening composition so that the primary subject or character remains visually dominant.",
    "testable_action": "Create Variant B by reducing background motion, cropping closer to the primary character, or dimming peripheral graphics during the first 1.5s, then compare focal clarity against current Variant A.",
    "limitation": "Observational comparison within the selected verified competitor cohort; does not prove that concentrated framing causes improved viewer retention.",
    "evidence_sample": "0/21 (0.0%) peers exhibit concentrated opening spatial focus"
  }
]
```

---

## 4. Status
**REAL-REEL VALIDATION: PASS**
