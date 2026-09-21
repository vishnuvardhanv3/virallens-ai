"""Discovery Agent.

Generates 3–5 high-precision, entity-first, niche-anchored multi-word Instagram Reel discovery queries.
Ensures zero single-word searches and preserves the primary entity/topic across
progressively broader search levels. Rejects malformed queries with dangling adjectives
or duplicate terms.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from utils.query_builder import QueryBuilder
from services.discovery_profile import build_discovery_profile


class DiscoveryAgent:
    """Agent generating multi-word entity/topic-anchored queries for Instagram Reel discovery."""

    @staticmethod
    def generate_queries(
        discovery_profile: Dict[str, Any],
        max_queries: int = 5,
    ) -> Dict[str, Any]:
        """Produce 3–5 high-precision multi-word queries prioritizing primary entity and topic."""
        # Ensure profile is structured and entity-validated
        if "entity_confidence" not in discovery_profile:
            profile = build_discovery_profile(discovery_profile)
        else:
            profile = discovery_profile

        primary_entity = str(profile.get("primary_entity") or "").strip()
        topic = str(profile.get("topic") or "").strip()
        niche = str(profile.get("niche") or "").strip()
        entity_conf = float(profile.get("entity_confidence", 0.0))

        raw_queries = QueryBuilder.build_structured_queries(profile)

        # Validate each query
        validated_queries = []
        for q in raw_queries:
            query_text = q.get("query", "").strip()
            if QueryBuilder.validate_query(query_text, primary_entity, topic, niche):
                validated_queries.append(q)

        # Cap at requested max
        validated_queries = validated_queries[:max_queries]
        query_strings = [q["query"] for q in validated_queries]

        if primary_entity and entity_conf >= 0.5:
            findings = [
                f"[MODEL_DERIVED] Generated {len(validated_queries)} entity-anchored multi-word queries prioritizing primary entity '{primary_entity}' (confidence: {entity_conf}).",
                f"[MODEL_DERIVED] Query escalation structured across levels: {', '.join(q.get('search_level', 'CORE') for q in validated_queries)}.",
            ]
        else:
            findings = [
                f"[MODEL_DERIVED] Generated {len(validated_queries)} topic/niche-anchored multi-word queries without invented entity (topic: '{topic}', niche: '{niche}').",
                f"[MODEL_DERIVED] Query escalation structured across levels: {', '.join(q.get('search_level', 'CORE') for q in validated_queries)}.",
            ]

        evidence = [f"[OBSERVED] Validated discovery query {i+1}: '{q}'" for i, q in enumerate(query_strings)]

        return {
            "agent": "discovery",
            "findings": findings,
            "evidence": evidence,
            "confidence": 0.95 if entity_conf >= 0.5 else 0.85,
            "uncertainties": [
                "Instagram search results depend on Apify scrape index and platform hashtag recency."
            ],
            "recommendations": [
                "[RECOMMENDATION] Execute search across multi-word queries sequentially, capping at 25 candidates per query."
            ],
            "queries": validated_queries,
            "query_strings": query_strings,
            "profile": profile,
        }

