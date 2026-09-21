"""Authoritative QueryBuilder for ViralLens AI.

Constructs 3-5 meaningful, entity-first, niche-anchored multi-word Instagram search queries from Reel analysis.

CRITICAL RULES (Sections 2, 5, 6, 7):
- Prioritizes ENTITY + TOPIC + FORMAT/STYLE.
- STRICTLY FORBIDDEN:
  - Generic styles replacing primary entities (e.g. NEVER "cinematic edit creative storytelling...")
  - Dangling tokens or stray adjectives (e.g. NEVER "superhero spider man high", "superhero high edit")
  - Single-word queries (e.g. NEVER "superhero", "spider man")
  - Repeated words or repeated "edit" tokens within a single query
  - Meaningless filler and purely generic combinations
- High precision:
  "{entity} {topic} {format}"
  "{entity} {topic} {style} edit"
- Medium:
  "{entity} {niche} edit"
  "{entity} {topic} reel"
- Broader:
  "{entity} {niche}"
  "{topic} {niche} edit"
- When no confident entity exists, safely falls back to topic + niche + format without inventing an entity.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, List, Dict, Set
from utils.helpers import sanitize_term, STOP_WORDS
from services.discovery_profile import (
    is_generic_entity,
    is_artifact_entity,
    GENERIC_STYLE_AND_FORMAT_TERMS,
    APPLICATION_AND_EDITOR_NAMES,
    TEST_AND_DEMO_ARTIFACTS,
)

DANGLING_MODIFIERS: Set[str] = {
    "high", "low", "fast", "slow", "dark", "light", "deep", "top", "hot",
    "new", "best", "good", "bad", "hard", "soft", "wide", "close", "raw",
    "contrast", "visual", "audio", "video", "post",
}

DISALLOWED_INTERIOR_MODIFIERS: Set[str] = {
    "high", "low", "contrast"
}


class QueryBuilder:
    """Builds and validates progressive multi-word discovery queries strictly prioritizing entity & topic."""

    @classmethod
    def clean_query_text(cls, query: str) -> str:
        """Sanitize and deduplicate adjacent/repeated words in a query string."""
        raw_words = query.strip().lower().split()
        seen = set()
        clean_words = []
        for w in raw_words:
            # Strip punctuation
            w_clean = re.sub(r"[^\w\s-]", "", w).strip()
            if not w_clean:
                continue
            if w_clean not in seen:
                seen.add(w_clean)
                clean_words.append(w_clean)
        return " ".join(clean_words).strip()

    @classmethod
    def is_meaningful_query(cls, query: str) -> bool:
        """Check if query is meaningful (at least 2 words, no dangling modifiers, no repeated words)."""
        q = str(query or "").strip().lower()
        words = q.split()
        if len(words) < 2:
            return False

        # Disallow ending in dangling modifier
        last_word = words[-1]
        if last_word in DANGLING_MODIFIERS:
            return False

        # Disallow duplicate words
        if len(words) != len(set(words)):
            return False

        # Disallow repeated "edit" or "edits"
        edit_count = sum(1 for w in words if w in {"edit", "edits"})
        if edit_count > 1:
            return False

        # Disallow editor/application names and test artifacts
        if any(w in APPLICATION_AND_EDITOR_NAMES or w in TEST_AND_DEMO_ARTIFACTS for w in words):
            return False

        # Disallow malformed interior modifiers like "superhero high edit"
        for idx, w in enumerate(words[:-1]):
            if w in DISALLOWED_INTERIOR_MODIFIERS:
                next_w = words[idx + 1]
                if next_w in {"edit", "edits", "reel", "reels", "scene", "clip"}:
                    return False

        # Reject if query consists purely of generic style/format terms
        non_stop = [w for w in words if w not in STOP_WORDS]
        if non_stop and all(w in GENERIC_STYLE_AND_FORMAT_TERMS for w in non_stop):
            return False

        return True

    @classmethod
    def validate_query(
        cls,
        query: str,
        entity: str = "",
        topic: str = "",
        niche: str = "",
    ) -> bool:
        """Validate that a query satisfies all scientific search intent requirements."""
        if not cls.is_meaningful_query(query):
            return False

        q = query.strip().lower()
        words = q.split()

        # At least 2 words, preferably 3-6
        if len(words) < 2 or len(words) > 7:
            return False

        # Check for nonsensical patterns
        if re.search(r"\bhigh\b", q) and (q.endswith("high") or "high edit" in q):
            return False

        # Entity/topic retention check:
        # If an entity was provided, ensure at least one substantive entity word appears
        clean_entity_text = entity.lower().replace("-", " ")
        entity_substantive = [w for w in clean_entity_text.split() if w not in STOP_WORDS and len(w) > 2]
        if entity_substantive:
            if not any(w in q for w in entity_substantive):
                return False
        else:
            # If no entity, topic or niche must appear
            clean_anchor_text = (topic.lower() + " " + niche.lower()).replace("-", " ")
            anchor_words = [
                w for w in clean_anchor_text.split()
                if w not in STOP_WORDS and len(w) > 2 and w not in GENERIC_STYLE_AND_FORMAT_TERMS
            ]
            if anchor_words and not any(w in q for w in anchor_words):
                return False

        return True

    @classmethod
    def build_structured_queries(
        cls, profile: Mapping[str, Any] | dict | None = None, **kwargs: Any
    ) -> list[dict[str, str]]:
        """Generate 3 to 5 validated, entity-first search queries with metadata."""
        data: dict[str, Any] = {}
        if isinstance(profile, Mapping):
            data.update(profile)
        data.update(kwargs)

        # 1. Extract and sanitize entity
        raw_entity = (
            data.get("primary_entity")
            or data.get("primary_subject")
            or data.get("subject")
            or data.get("entity")
            or ""
        )
        entity = sanitize_term(raw_entity)
        entity_conf = float(data.get("entity_confidence", 1.0 if entity else 0.0))
        entity_usable = (
            bool(data.get("entity_usable_for_discovery", True))
            and entity_conf >= 0.50
            and not is_artifact_entity(entity)
            and not is_generic_entity(entity)
        )
        if not entity_usable:
            entity = ""

        # 2. Extract and sanitize topic
        raw_topic = data.get("topic") or data.get("primary_topic") or ""
        topic = sanitize_term(raw_topic)
        if is_generic_entity(topic) or is_artifact_entity(topic):
            topic = ""

        # 3. Extract and sanitize niche
        raw_niche = (
            data.get("niche")
            or data.get("main_niche")
            or data.get("primary_niche")
            or ""
        )
        niche = sanitize_term(raw_niche)
        if niche.lower() in {"other", "general", "video", "reel", "creative"}:
            niche = ""

        # 4. Extract sub_niche and edit_type signals
        raw_sub = sanitize_term(data.get("sub_niche") or "")
        raw_edit_type = sanitize_term(data.get("edit_type") or "")
        format_term = sanitize_term(data.get("format") or data.get("content_format") or "")
        style_term = sanitize_term(data.get("style") or data.get("visual_style") or "")

        # Clean individual components (remove dangling modifiers from component strings)
        def clean_component(c: str) -> str:
            w_list = [w for w in c.split() if w not in DANGLING_MODIFIERS]
            return " ".join(w_list).strip()

        clean_entity = clean_component(entity)
        clean_topic = clean_component(topic)
        clean_niche = clean_component(niche)
        clean_sub_niche = clean_component(raw_sub)
        clean_edit_type = clean_component(raw_edit_type)

        if not clean_edit_type or is_generic_entity(clean_edit_type) or clean_edit_type in {"mixed", "unknown"}:
            if "montage" in clean_topic or "action" in clean_topic or "montage" in format_term:
                clean_edit_type = "action montage"
            elif "car" in clean_niche or "automotive" in clean_niche or "car" in clean_topic:
                clean_edit_type = "car edit"
            elif "tutorial" in format_term or "tutorial" in clean_topic:
                clean_edit_type = "tutorial guide"
            elif "dialogue" in format_term or "quote" in clean_topic:
                clean_edit_type = "dialogue edit"
            elif "vfx" in format_term:
                clean_edit_type = "vfx edit"
            else:
                clean_edit_type = "fan edit"

        # Select clean format modifier
        clean_format = "fan edit"
        if "scene" in format_term or "scene" in clean_topic:
            clean_format = "scene edit"
        elif "tutorial" in format_term or "tutorial" in clean_topic:
            clean_format = "tutorial guide"
        elif "skit" in format_term or "comedy" in clean_niche:
            clean_format = "comedy skit"
        elif "montage" in format_term or "action montage" in clean_edit_type:
            clean_format = "montage"

        clean_style = "cinematic"
        if style_term and style_term not in GENERIC_STYLE_AND_FORMAT_TERMS:
            clean_style = clean_component(style_term) or "cinematic"

        def semantic_compose(*parts: str, max_words: int = 4) -> str:
            """Compose parts preserving semantic concept order without duplicate tokens, bounded to max_words."""
            tokens = []
            seen = set()
            for part in parts:
                if not part:
                    continue
                words = part.lower().split()
                for w in words:
                    w_clean = re.sub(r"[^\w-]", "", w).strip()
                    if w_clean and w_clean not in seen and w_clean not in STOP_WORDS:
                        seen.add(w_clean)
                        tokens.append(w_clean)
                        if len(tokens) >= max_words:
                            return " ".join(tokens)
            return " ".join(tokens)

        # Construct template pool based on entity presence and usability
        template_candidates: list[tuple[str, str]] = []

        if clean_entity:
            entity_words = clean_entity.split()
            if len(entity_words) >= 3:
                if entity_words[0].lower() in {"mega", "super", "the", "a", "an"}:
                    core_entity = entity_words[-1]
                else:
                    core_entity = f"{entity_words[0]} {entity_words[-1]}"
            else:
                core_entity = clean_entity

            # Priority 1: Entity anchor
            # Multi-word entities with > 10 chars are indexed Instagram topics (e.g. "canelo alvarez", "mega man megamind")
            if len(entity_words) >= 2 and len(clean_entity) > 10:
                q1 = clean_entity
            else:
                q1 = semantic_compose(clean_entity, clean_edit_type or "edit", max_words=4)
            template_candidates.append((q1, "high_precision"))

            # Priority 2: Universal Entity Edit (the primary hashtag/search pattern on Instagram Reels)
            q2_edit = semantic_compose(clean_entity if len(entity_words) <= 2 else core_entity, "edit", max_words=3)
            template_candidates.append((q2_edit, "high_precision"))

            # Priority 3: Entity + specific edit type or style
            if clean_edit_type and clean_edit_type not in {"edit", "fan edit"}:
                q3_type = semantic_compose(clean_entity if len(entity_words) <= 2 else core_entity, clean_edit_type, max_words=4)
                template_candidates.append((q3_type, "high_precision"))
            elif clean_topic and not any(w in clean_topic for w in entity_words):
                q3_topic = semantic_compose(clean_entity, clean_topic, max_words=4)
                template_candidates.append((q3_topic, "high_precision"))
            else:
                q3_style = semantic_compose(clean_entity, clean_style or "cinematic", "edit", max_words=4)
                template_candidates.append((q3_style, "high_precision"))

            # Priority 4: Entity + niche
            q4 = semantic_compose(clean_entity if len(entity_words) <= 2 else core_entity, clean_niche or "reel", max_words=4)
            template_candidates.append((q4, "medium"))

            # Priority 5: Fallback entity/core entity + niche + edit
            q5 = semantic_compose(core_entity, clean_niche or "video", "edit", max_words=4)
            template_candidates.append((q5, "fallback"))

        else:
            # When entity is unusable: fallback strictly to topic/niche/format
            anchor_topic = clean_topic or "storytelling"
            anchor_niche = clean_niche or "entertainment"

            # Priority 1: topic + niche + edit_type
            q1 = semantic_compose(anchor_topic, anchor_niche, clean_edit_type or clean_format, max_words=4)
            template_candidates.append((q1, "high_precision"))

            # Priority 2: topic + style + edit
            q2 = semantic_compose(anchor_topic, clean_style, "edit", max_words=3)
            template_candidates.append((q2, "high_precision"))

            # Priority 3: topic + niche
            q3 = semantic_compose(anchor_topic, anchor_niche, max_words=3)
            template_candidates.append((q3, "medium"))

            # Priority 4: niche + format
            q4 = semantic_compose(anchor_niche, clean_edit_type or clean_format, max_words=3)
            template_candidates.append((q4, "medium"))

            # Priority 5: Fallback broader query
            q5 = semantic_compose(anchor_niche, "video edit", max_words=3)
            template_candidates.append((q5, "fallback"))

        validated_queries: list[dict[str, str]] = []
        seen_texts: set[str] = set()

        for raw_q, level in template_candidates:
            clean_q = cls.clean_query_text(raw_q)
            if not clean_q or clean_q in seen_texts:
                continue

            if cls.validate_query(clean_q, clean_entity, clean_topic, clean_niche):
                seen_texts.add(clean_q)
                validated_queries.append({
                    "query": clean_q,
                    "search_level": level,
                })

            if len(validated_queries) >= 5:
                break

        # Fallback if fewer than 3 queries pass
        if len(validated_queries) < 3:
            subject = clean_entity or clean_topic or clean_niche or "video"
            fallbacks = [
                (f"{subject} cinematic fan edit", "high_precision"),
                (f"{subject} scene breakdown edit", "medium"),
                (f"{subject} highlight reel", "broader"),
            ]
            for fb_text, fb_level in fallbacks:
                clean_fb = cls.clean_query_text(fb_text)
                if clean_fb not in seen_texts and cls.is_meaningful_query(clean_fb):
                    seen_texts.add(clean_fb)
                    validated_queries.append({
                        "query": clean_fb,
                        "search_level": fb_level,
                    })
                if len(validated_queries) >= 3:
                    break

        return validated_queries[:5]


def build_structured_queries(
    profile: Mapping[str, Any] | dict | None = None, **kwargs: Any
) -> list[dict[str, str]]:
    """Functional convenience wrapper for QueryBuilder."""
    return QueryBuilder.build_structured_queries(profile, **kwargs)


is_meaningful_query = QueryBuilder.is_meaningful_query
validate_query = QueryBuilder.validate_query
build_queries = build_structured_queries
QueryBuilder.build_queries = QueryBuilder.build_structured_queries


