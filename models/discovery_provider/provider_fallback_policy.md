# Provider Fallback Policy

## 1. Principle of Bounded Fallback
Fallback from the primary provider (`apify`) to a secondary provider (`agent_reach`) must be **explicit, deterministic, and auditable**. Silent failover is strictly prohibited.

## 2. Trigger Conditions
A single blocked query is **NEVER** sufficient by itself to trigger provider fallback. The system relies first on the multi-query discovery resilience established in Phase 13.

Fallback to secondary occurs **ONLY** when:
1. **Primary Provider Hard Failure**: `primary.health() == FAILED` (e.g., missing API token or unresolvable infrastructure downtime).
2. **Exhaustion with Zero Results**: Primary multi-query pipeline completed across all attempted queries with `discovery_health == "failed"` and 0 candidates returned.
3. **Candidate Budget Exhaustion**: Primary completed with partial/degraded health and total discovered candidates fall below `TARGET_DISCOVERY_CANDIDATES` (default: 30), and configuration explicitly enables fallback mode (`DISCOVERY_PROVIDER=fallback`).

## 3. Mandatory Fallback Event Logging
Every fallback invocation emits a structured event:
```json
{
  "primary_provider": "apify",
  "secondary_provider": "agent_reach",
  "trigger": "primary_candidate_budget_exhausted",
  "candidate_count_before_fallback": 8,
  "candidate_count_after_fallback": 22,
  "timestamp": 1726568100.123
}
```
Candidates from both providers are merged using canonical shortcode deduplication, preserving cross-provider provenance in `raw_provenance`.
