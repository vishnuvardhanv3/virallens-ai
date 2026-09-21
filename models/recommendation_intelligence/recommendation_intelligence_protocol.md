# Recommendation Intelligence Protocol & Epistemic Specification

**Phase 16 — ViralLens Product Intelligence Layer**  
**Version:** 1.0.0  
**Status:** VALIDATED IN PRODUCTION  

---

## 1. Architectural Mandate

Recommendation Intelligence is an evidence-backed decision-support system synthesizing:
- Laboratory-derived human visual attention (TinySalNet spatial saliency, NEMAR temporal attention)
- Multimodal understanding (Twelve Labs semantic topics, OCR typography, scene transitions)
- Verified peer competitor cohorts (Apify discovered, Twelve Labs semantically verified, reach-ordered)
- Empirical pattern tallies from the Pattern Agent

### Epistemic Guardrails & Hard Constraints
1. **Strictly Non-Causal:** The system NEVER claims "what makes a reel go viral", "guaranteed engagement", "improves viewer retention", or "predicts scroll-stop behavior".
2. **Zero ML Model Alterations:** TinySalNet, NEMAR, and DHF1K remain 100% frozen. No new ML models are introduced.
3. **Zero Network Calls:** Pure analytical synthesis in-memory over precomputed, verified data.
4. **No Fabricated Confidence:** Confidence is strictly `confidence: null`. No arbitrary percentages (e.g. "92% confidence") from heuristic scoring.
5. **Traceable Evidence Chain:** Every recommendation must trace backwards from recommendation $\to$ testable action $\to$ creative implication $\to$ target-vs-peer gap $\to$ explicit competitor counts ($num/den$) $\to$ source features.

---

## 2. 7-Stage Recommendation Pipeline

Every recommendation emitted by ViralLens must follow this canonical progression:

```
OBSERVATION
    ↓
EVIDENCE (Numerator / Denominator + Supporting Competitor IDs)
    ↓
TARGET vs COMPARISON (Formal Gap Representation)
    ↓
CREATIVE IMPLICATION
    ↓
RECOMMENDATION
    ↓
TESTABLE ACTION (A/B Creator Experiment)
    ↓
LIMITATION
```

### Example Canonical Flow:
- **Observation:** Target opening visual saliency is diffusely distributed across competing regions (dispersion: 0.420, state: 'moderate_competition').
- **Evidence:** 19/21 (90.5%) peers in `same_edit_type` exhibit concentrated opening spatial focus around a dominant subject.
- **Target vs Comparison:** Target dispersion=0.420 vs 19/21 concentrated in cohort. Target provides less focal concentration than observed peer majority.
- **Creative Implication:** Multiple visual regions compete for viewer gaze during opening seconds, dispersing ocular concentration.
- **Recommendation:** Consider simplifying the opening composition so the primary subject remains visually dominant.
- **Testable Action:** Create Variant B by cropping closer or dimming peripheral graphics during the first 1.5s, then compare focal clarity against current Variant A.
- **Limitation:** Observational comparison within the selected verified competitor cohort; does not prove that concentrated framing causes improved viewer retention.

---

## 3. Recommendation Categories

1. **`PRESERVE`**: The target reel already exhibits a characteristic aligned with high-prevalence peer conventions. Prevents unnecessary changes.
2. **`CHANGE`**: Strong, consistent evidence ($n \ge 5$, prevalence $\ge 70\%$) reveals a clear divergence between target and peer majority.
3. **`TEST`**: Evidence is suggestive ($50\% \le \text{prevalence} < 70\%$) or exploratory, formulated as a creative hypothesis for evaluation.
4. **`AVOID_OVERINTERPRETATION`**: Sample size is small ($n < 5$) or variance across peers is high, precluding confident creative intervention.
5. **`INVESTIGATE`**: Multi-signal conflict detected (e.g. text overlay diverging from character gaze anchor). Requires human review.
6. **`NO_ACTIONABLE_RECOMMENDATION`**: When zero verified competitors exist or peer data lacks actionable contrast.

---

## 4. Evidence Strength & Small-Sample Safety Rules

| Sample Size ($n$) | Prevalence ($P$) | Strength Classification | Interpretation |
|---|---|---|---|
| $n < 5$ | Any | `insufficient_evidence` | Strictly exploratory; small cohort flag appended to limitation |
| $5 \le n < 10$ | $P \ge 65\%$ | `moderate_observed_evidence` | Cautious descriptive pattern |
| $5 \le n < 10$ | $P < 65\%$ | `exploratory_evidence` | Hypothesis only |
| $n \ge 10$ | $P \ge 70\%$ | `strong_observed_evidence` | High consistency descriptive benchmark |
| $n \ge 10$ | $50\% \le P < 70\%$ | `moderate_observed_evidence` | Moderate consistency |
| $n \ge 10$ | $P < 50\%$ | `exploratory_evidence` | Weak peer signal |

---

## 5. Conflict Resolution & Deduplication

- **Deduplication Matrix:** Recommendations are mapped to single analytical domains (`spatial`, `typography`, `human`, `editing`, `temporal`, `audio`). Multiple raw cues in the same domain are unified into a single recommendation with aggregated supporting evidence.
- **Contradictory Signal Handling:** If spatial composition is diffuse, aggressive scene cut pace is flagged to prevent compounded visual disorientation. Conflicting text/gaze signals are automatically escalated to `INVESTIGATE`.

---

## 6. Real-Reel Benchmark Invariance

Validation on `htdyr.mp4` and `upload_jvxawbcy.mp4` proved:
- Latency overhead: `< 1.0 ms` total per reel.
- Production invariance: Zero mutations to input target profiles, Twelve Labs metadata, NEMAR curves, or TinySalNet tensors.
