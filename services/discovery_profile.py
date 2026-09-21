"""Builds normalized, evidence-grounded discovery profiles from uploaded Reel analysis.

Enforces:
1. Validates primary entity against multimodal video understanding evidence.
2. Rejects generic style/format phrases from being treated as primary entities.
3. Classifies entity_type:
   - content_entity
   - brand_entity
   - person_entity
   - place_entity
   - product_entity
   - media_entity
   - artifact
   - unknown
4. Rejects application/editor names (CapCut, Premiere, ViralLens, etc.) and test/demo/sequence artifacts contextually.
5. Sets entity_usable_for_discovery = False if the entity is an artifact, generic, watermark-only, or has confidence < 0.50.
6. Emits structured profile with visual_subjects, keywords, content_format, style, and confidence scores.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, List, Set
from utils.helpers import sanitize_term, STOP_WORDS

GENERIC_STYLE_AND_FORMAT_TERMS: Set[str] = {
    "cinematic", "cinematic edit", "edit", "edits", "video", "videos",
    "reel", "reels", "clip", "clips", "post", "posts", "content",
    "general", "general content", "creative", "creative storytelling", "storytelling",
    "visual", "visual edit", "visual pacing edit", "pacing edit", "visual effects", "vfx",
    "high quality", "hd", "4k", "fan edit", "scene edit", "montage",
    "trending", "viral", "viral video", "video edit", "short", "shorts", "footage",
    "story", "stories", "social media", "instagram", "tiktok", "aesthetic edit",
    "reel edit"
}

APPLICATION_AND_EDITOR_NAMES: Set[str] = {
    "capcut", "premiere", "premiere pro", "inshot", "virallens", "canva",
    "filmora", "vn", "davinci", "davinci resolve", "after effects", "adobe", "photoshop",
    "alight motion", "kinemaster", "final cut", "final cut pro", "kdenlive",
    "lightcut", "powerdirector"
}

TEST_AND_DEMO_ARTIFACTS: Set[str] = {
    "test", "demo", "sample", "test sequence", "frame sequence", "test reel",
    "benchmark", "test sequence reel", "test video", "sample video", "demo reel",
    "test frame", "frame sequence reel", "test pattern"
}


def is_generic_entity(term: str) -> bool:
    """Check if a candidate entity is merely a generic style, format, or filler phrase."""
    t = sanitize_term(term).lower()
    if not t:
        return True
    if t in GENERIC_STYLE_AND_FORMAT_TERMS:
        return True

    # Check if phrase reduces to empty when removing generic phrases
    temp_t = t
    for g in sorted(GENERIC_STYLE_AND_FORMAT_TERMS, key=len, reverse=True):
        temp_t = re.sub(r"\b" + re.escape(g) + r"\b", " ", temp_t)
    if not temp_t.strip():
        return True

    # Check if every non-stopword in the candidate entity is a generic term
    words = [w for w in t.split() if w not in STOP_WORDS]
    if not words:
        return True
    if all(w in GENERIC_STYLE_AND_FORMAT_TERMS for w in words):
        return True

    return False


def is_artifact_entity(term: str) -> bool:
    """Detect if a candidate entity is an editor name, application name, or test/demo artifact."""
    raw_lower = str(term or "").strip().lower()
    if raw_lower.startswith("@"):
        return True
    t = sanitize_term(term).lower()
    if not t:
        return False
    if any(app in t for app in APPLICATION_AND_EDITOR_NAMES):
        return True
    if any(art in t for art in TEST_AND_DEMO_ARTIFACTS):
        return True
    words = [w for w in t.split() if w not in STOP_WORDS]
    if any(w in APPLICATION_AND_EDITOR_NAMES for w in words):
        return True
    if all(w in TEST_AND_DEMO_ARTIFACTS for w in words):
        return True
    return False


def classify_entity_type(term: str, context: Mapping[str, Any] | None = None) -> str:
    """Classify the entity into one of the canonical entity_type categories."""
    if is_artifact_entity(term):
        return "artifact"
    t = sanitize_term(term).lower()
    if not t:
        return "unknown"
    if is_artifact_entity(t):
        return "artifact"
    if is_generic_entity(t):
        return "unknown"

    ctx = context or {}
    niche = str(ctx.get("niche") or "").lower()
    topic = str(ctx.get("topic") or "").lower()

    # Known media / fiction / entertainment characters
    if niche in {"superhero", "entertainment", "movie", "film", "gaming", "anime"}:
        return "media_entity"

    # Brands and products
    if any(k in niche or k in topic for k in ["tech", "software", "product", "gadget", "commerce", "brand"]):
        return "brand_entity"

    # Places / travel
    if any(k in niche or k in topic for k in ["travel", "city", "country", "nature", "place"]):
        return "place_entity"

    # Person
    if any(k in niche or k in topic for k in ["fitness", "creator", "interview", "podcast", "vlog", "person"]):
        return "person_entity"

    return "content_entity"


EDIT_TYPES_TAXONOMY: dict[str, list[str]] = {
    "action montage": ["action", "montage", "fight", "chase", "combat", "stunt", "dynamic", "beat-sync", "swings", "punch", "hero action"],
    "emotional edit": ["sad", "emotional", "nostalgic", "tribute", "drama", "heartbreaking", "slow", "tear", "loss", "memory", "melancholy"],
    "dialogue edit": ["dialogue", "quote", "speech", "talking", "monologue", "voiceover", "conversation", "words"],
    "vfx edit": ["vfx", "cgi", "visual effects", "transformation", "glow", "glow edit", "smooth transition", "compositing"],
    "meme edit": ["funny", "meme", "humor", "comedy", "parody", "satire", "joke", "prank"],
    "tutorial": ["tutorial", "how to", "guide", "tips", "technique", "breakdown"],
}


def infer_sub_niche_and_edit_type(
    data: dict[str, Any],
    clean_entity: str = "",
    clean_niche: str = "",
    clean_topic: str = "",
    clean_format: str = "",
    clean_style: str = "",
    **kwargs: Any,
) -> tuple[str, str, float, float]:
    """Contextually infer sub_niche and edit_type with calibrated confidence."""
    if not clean_entity:
        clean_entity = sanitize_term(data.get("primary_entity") or data.get("primary_subject") or "")
    if not clean_niche:
        clean_niche = sanitize_term(data.get("niche") or "")
    if not clean_topic:
        clean_topic = sanitize_term(data.get("topic") or "")
    if not clean_format:
        clean_format = sanitize_term(data.get("format") or "")

    evidence_tokens = []
    for k in ["topic", "sub_niche", "edit_type", "style", "visual_style", "editing_style", "format", "tone"]:
        v = data.get(k)
        if isinstance(v, str) and v:
            evidence_tokens.append(v.lower())

    for list_k in ["visual_subjects", "keywords", "discovery_keywords"]:
        v_list = data.get(list_k)
        if isinstance(v_list, list):
            for item in v_list:
                if isinstance(item, str):
                    evidence_tokens.append(item.lower())

    tech = data.get("technical_signals") or data.get("video") or {}
    cut_rate = float(tech.get("cut_rate") or tech.get("scene_cut_rate") or kwargs.get("scene_cut_rate") or 0.0)
    motion = float(tech.get("motion_magnitude") or kwargs.get("motion_magnitude") or 0.0)
    pacing = str(data.get("pacing") or tech.get("pacing") or kwargs.get("pacing") or "").lower()

    if cut_rate > 0.45 or "fast" in pacing or motion > 0.04:
        evidence_tokens.append("action")
        evidence_tokens.append("dynamic")

    evidence_text = " ".join(evidence_tokens)

    # 1. Infer edit_type
    edit_type_scores: dict[str, int] = {et: 0 for et in EDIT_TYPES_TAXONOMY}
    for et, keywords in EDIT_TYPES_TAXONOMY.items():
        for kw in keywords:
            if kw in evidence_text:
                edit_type_scores[et] += 1

    best_et, best_score = max(edit_type_scores.items(), key=lambda x: x[1])

    if best_score >= 2:
        edit_type = best_et
        edit_type_conf = min(0.95, 0.65 + 0.10 * best_score)
    elif best_score == 1:
        edit_type = best_et
        edit_type_conf = 0.65
    elif any(k in clean_format for k in ["montage", "action"]):
        edit_type = "action montage"
        edit_type_conf = 0.60
    elif any(k in clean_format for k in ["tutorial", "guide"]):
        edit_type = "tutorial"
        edit_type_conf = 0.60
    elif clean_format:
        edit_type = "action montage" if "edit" in clean_format else "mixed"
        edit_type_conf = 0.50
    else:
        edit_type = "unknown"
        edit_type_conf = 0.0

    # 2. Infer sub_niche
    raw_sub = sanitize_term(data.get("sub_niche") or "")
    if raw_sub and raw_sub not in GENERIC_STYLE_AND_FORMAT_TERMS and not is_artifact_entity(raw_sub):
        sub_niche = raw_sub
        sub_niche_conf = 0.85
    elif clean_entity and clean_niche in {"superhero", "entertainment", "anime", "movie", "film", "gaming"}:
        sub_niche = f"{clean_entity} fan edits"
        sub_niche_conf = 0.88
    elif clean_niche and clean_niche != "creative":
        sub_niche = f"{clean_niche} edits" if "edit" in edit_type else f"{clean_niche} content"
        sub_niche_conf = 0.65
    else:
        sub_niche = "creative video"
        sub_niche_conf = 0.40

    return sub_niche, edit_type, round(sub_niche_conf, 2), round(edit_type_conf, 2)


def build_discovery_profile(analysis: Mapping[str, Any] | dict | None) -> dict[str, Any]:
    """Construct a clean, structured Reel profile used for entity-first query generation."""
    if not analysis:
        return {
            "primary_entity": "",
            "entity_type": "unknown",
            "entity_usable_for_discovery": False,
            "topic": "general content",
            "niche": "creative",
            "content_format": "cinematic edit",
            "style": "",
            "visual_subjects": [],
            "keywords": [],
            "entity_confidence": 0.0,
            "topic_confidence": 0.30,
            "niche_confidence": 0.30,
            "sub_niche": "",
            "edit_type": "unknown",
            "sub_niche_confidence": 0.0,
            "edit_type_confidence": 0.0,
            "format": "cinematic edit",
            "search_terms": ["creative video"],
        }

    # Handle nested canonical or analysis structures if present
    data: dict[str, Any] = dict(analysis)
    if "canonical" in data and isinstance(data["canonical"], dict):
        canonical = data["canonical"]
        for k in ["primary_entity", "topic", "niche", "sub_niche", "format", "style", "keywords"]:
            if canonical.get(k) and not data.get(k):
                data[k] = canonical[k]
    if "analysis" in data and isinstance(data["analysis"], dict):
        nested_analysis = data["analysis"]
        for k in ["primary_entity", "topic", "niche", "sub_niche", "format", "visual_style", "style", "discovery_keywords"]:
            if nested_analysis.get(k) and not data.get(k):
                data[k] = nested_analysis[k]

    # 1. Topic Extraction
    raw_topic = sanitize_term(data.get("topic") or data.get("primary_topic") or "")
    if is_generic_entity(raw_topic):
        clean_topic = "general content"
        topic_conf = 0.30
    elif is_artifact_entity(raw_topic):
        clean_topic = "general content"
        topic_conf = 0.30
    elif raw_topic:
        clean_topic = raw_topic
        topic_conf = 0.85
    else:
        clean_topic = "general content"
        topic_conf = 0.30

    # 2. Niche Extraction
    raw_niche = sanitize_term(
        data.get("niche")
        or data.get("main_niche")
        or data.get("primary_niche")
        or ""
    )
    if raw_niche.lower() in {"other", "general", "video", "reel"}:
        clean_niche = "creative"
        niche_conf = 0.30
    elif raw_niche:
        clean_niche = raw_niche
        niche_conf = 0.85
    else:
        clean_niche = "creative"
        niche_conf = 0.30

    sub_niche = sanitize_term(data.get("sub_niche") or "")

    # 3. Format & Style
    content_format = sanitize_term(data.get("format") or data.get("content_format") or "cinematic edit")
    style = sanitize_term(data.get("visual_style") or data.get("style") or "")

    # 4. Visual Subjects and Evidence Extraction
    visual_subjects: List[str] = []
    raw_vs = data.get("visual_subjects") or []
    if isinstance(raw_vs, list):
        for vs in raw_vs:
            c = sanitize_term(vs)
            if c and c not in visual_subjects and not is_artifact_entity(c):
                visual_subjects.append(c)

    # Check hook and structure evidence for visual subjects
    for block_key in ["hook", "structure", "visual"]:
        block = data.get(block_key)
        if isinstance(block, dict):
            ev_list = block.get("evidence") or []
            if isinstance(ev_list, list):
                for ev in ev_list:
                    c = sanitize_term(ev)
                    if c and len(c.split()) <= 4 and c not in visual_subjects and not is_artifact_entity(c):
                        visual_subjects.append(c)

    # 5. Keywords
    raw_keywords = data.get("discovery_keywords") or data.get("keywords") or []
    keywords: List[str] = []
    if isinstance(raw_keywords, list):
        for kw in raw_keywords:
            clean_kw = sanitize_term(kw)
            if clean_kw and clean_kw not in keywords and not is_artifact_entity(clean_kw):
                keywords.append(clean_kw)

    # 6. Primary Entity Extraction, Artifact Filtering & Validation
    raw_entity = sanitize_term(
        data.get("primary_entity")
        or data.get("primary_subject")
        or data.get("subject")
        or data.get("entity")
        or ""
    )

    entity_type = classify_entity_type(raw_entity, {"niche": clean_niche, "topic": clean_topic})

    if is_artifact_entity(raw_entity):
        clean_entity = ""
        entity_type = "artifact"
        entity_conf = 0.0
        entity_usable = False
    elif is_generic_entity(raw_entity):
        clean_entity = ""
        entity_type = "unknown"
        entity_conf = 0.0
        entity_usable = False
    else:
        clean_entity = raw_entity
        entity_conf = 0.70  # Baseline for candidate entity

        # Check corroboration across evidence sources
        entity_lower = clean_entity.lower()
        corroborated = (
            any(entity_lower in kw.lower() for kw in keywords)
            or any(entity_lower in vs.lower() for vs in visual_subjects)
            or (entity_lower in clean_topic.lower())
        )
        if corroborated:
            entity_conf = 0.92

        # OCR-only or single-watermark check:
        # If entity is only mentioned in OCR text and not corroborated in visual subjects or topic, downweight
        ocr_obj = data.get("ocr") or {}
        ocr_str = str(data.get("onscreen_text") or ocr_obj.get("ocr_text") or "").lower()
        if clean_entity and entity_lower in ocr_str and not corroborated:
            # Watermark or text-only artifact
            entity_conf = 0.35

        entity_usable = bool(entity_conf >= 0.50 and entity_type != "artifact" and clean_entity)

    sub_niche, edit_type, sub_niche_conf, edit_type_conf = infer_sub_niche_and_edit_type(
        data, clean_entity, clean_niche, clean_topic, content_format, style
    )
    search_terms = [t for t in [clean_entity, clean_topic, clean_niche, sub_niche, edit_type] if t]

    return {
        "primary_entity": clean_entity if entity_usable else "",
        "entity_type": entity_type,
        "entity_usable_for_discovery": entity_usable,
        "topic": clean_topic,
        "niche": clean_niche,
        "sub_niche": sub_niche,
        "edit_type": edit_type,
        "sub_niche_confidence": sub_niche_conf,
        "edit_type_confidence": edit_type_conf,
        "content_format": content_format,
        "style": style,
        "visual_subjects": visual_subjects,
        "keywords": keywords,
        "entity_confidence": round(entity_conf, 2),
        "topic_confidence": round(topic_conf, 2),
        "niche_confidence": round(niche_conf, 2),
        # Backward compatibility aliases
        "format": content_format,
        "search_terms": search_terms,
    }
