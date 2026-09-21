# ViralLens AI — Release Notes (v0.18.0-rc1)

## 1. Product Overview
ViralLens AI is an advanced multimodal video understanding and competitor intelligence platform designed for short-form video (Instagram Reels). It bridges state-of-the-art multimodal video AI (Twelve Labs), empirical human visual attention models (NEMAR laboratory eye-tracking & TinySalNet spatial saliency), and provider-independent competitor discovery into an evidence-grounded creative decision support system.

## 2. Component Inventory & Classification

### A. Production Components (Verified & Hardened)
- **Twelve Labs Multimodal Video Understanding**: Native video analysis via Pegasus 1.5 extracting narrative structure, visual topics, soundscapes, and scene semantics.
- **NEMAR Human Visual Attention Model**: Laboratory-calibrated temporal visual attention potential predictions derived from eye-tracking fixations (v1.0.0).
- **Specialist Multi-Agent Analytical Suite**:
  - *Human Attention Agent*: Pinpoints initial attention spikes and decay curves.
  - *Specialist Video OCR Agent*: High-precision on-screen typography detection with timestamped bounding coordinates.
  - *Hook Agent*: Evaluates the crucial opening 1.0–3.0s window for visual novelty and framing.
  - *Human Behavior Agent*: Differentiates observable posture and gestures from subjective interpretation.
  - *Emotion Agent*: Quantifies peak expressive intensity and affective cues.
  - *Editing Agent*: Measures pace, shot boundaries, and visual transitions.
  - *Communication Agent*: Analyzes direct address, clarity, and auditory delivery.
  - *Discovery Agent*: Derives niche-anchored multi-word queries without single-word generic terms.
- **Discovery Provider Architecture**:
  - *Apify Instagram Search Scraper*: Primary production provider with query-level resilience, unretried platform blocks, bounded network retries, and deterministic caching.
  - *Canonical Candidate Normalization*: Strict shortcode/URL deduplication, view-based reach preservation, and provenance tracking.
- **Relevance & Reach Ranking**:
  - Semantic relevance verification gating competitors against target video profile.
  - Top-reach competitor prioritization (up to 10 verified Reels, max 2 per creator).
- **Competitor Intelligence & Pattern Agent**:
  - Analyzes recurring observable characteristics across verified peer comparison groups.
  - Transparent denominator accounting (e.g., `2/2 (100.0%)`).
- **Recommendation Intelligence Engine**:
  - Evidence-backed creative recommendations with testable hypotheses and explicit limitations.
- **Streamlit Dashboard & Diagnostics**:
  - Compact pre-flight system health (`System: 🟢 READY`), query-level discovery status, and real-time execution telemetry.

### B. Experimental & Shadow Components
- **TinySalNet Spatial Saliency (Shadow Mode)**:
  - Evaluates spatial fixation dispersion and visual conspicuity across sampled frames without blocking or gating the primary pipeline.
- **Agent-Reach / OpenCLI Discovery Adapter**:
  - Experimental secondary discovery provider with truthful availability detection and isolated fallback capabilities.

## 3. Strict Epistemic Boundaries
- **Non-Causal Evidence**: ViralLens provides descriptive peer comparisons and creative decision support. It does **not** predict platform algorithm distribution, guarantee viral reach, or measure actual audience scroll-stopping behavior.
- **Data Integrity**: Genuinely observed views are preserved; missing reach is strictly represented as `None` (never `0`, never synthesized from likes or comments).
