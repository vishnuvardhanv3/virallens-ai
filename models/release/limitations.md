# ViralLens AI — Scientific and Operational Limitations

This document explicitly defines the boundaries and limitations of ViralLens AI. Overclaiming, causal inference, and unsupported generalizations are strictly prohibited across all documentation, UI presentations, and reports.

## 1. Human Visual Attention Models (NEMAR)
- **Laboratory-Derived Context**: NEMAR temporal attention models are trained on eye-tracking fixations recorded in controlled laboratory environments.
- **Not Social Media Retention**: Predicted visual attention potential reflects sensory visual salience and ocular fixation patterns. It does **not** reflect real-world smartphone scrolling behavior, app retention telemetry, or platform algorithmic distribution.
- **Participant Scope**: Laboratory eye-tracking datasets reflect a finite cohort of participants and standardized screen displays.

## 2. Spatial Saliency & TinySalNet (Shadow Mode)
- **Visual Conspicuity vs Mobile Engagement**: Saliency maps identify where luminance, contrast, and visual movement naturally draw gaze. They do **not** measure viewer psychological engagement or comprehension.
- **Shadow Status**: Spatial attention metrics operate purely in non-blocking shadow mode to ensure zero disruption to production discovery and intelligence.

## 3. Instagram Competitor Discovery & Provider Dependencies
- **Third-Party Availability**: ViralLens relies on external scraping providers (primarily Apify `apify/instagram-search-scraper`). Changes in platform security, anti-bot mechanisms, rate limits, or regional routing can cause individual queries to return blocked or empty states.
- **Reach Metadata Integrity**: Instagram does not uniformly expose public play counts across all post types. ViralLens strictly treats missing views as `None` rather than substituting zeros, likes, or comments.
- **Dynamic Content**: Instagram search rankings change dynamically based on active platform trends.

## 4. Competitor Intelligence & Pattern Discovery
- **Finite Sample Size**: Competitor analysis operates on a capped, verified peer group (up to 10 verified Reels). Findings represent descriptive observations of that specific peer group, not population-wide rules.
- **Descriptive, Non-Causal**: Shared characteristics among top-performing Reels reflect observed correlation, never causal guarantees that replicating those traits will yield virality.

## 5. Recommendation Intelligence
- **Decision Support**: Creative recommendations represent evidence-grounded hypotheses for creators to evaluate and A/B test.
- **No Virality Guarantees**: No model or algorithm can guarantee algorithmic promotion or audience engagement.
