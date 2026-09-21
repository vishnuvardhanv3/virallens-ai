"""Official Twelve Labs multimodal video-understanding provider.

Handles:
- Uploaded Reel video understanding
- Competitor Reel video understanding
- Video context preparation (base64 / direct asset)
- SHA-256 based deterministic analysis caching
- Retries and timeout management
- Semantic relevance verification between uploaded and competitor Reels
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

from config import settings
from services.canonical_schema import normalize_score_0_to_1

ANALYSIS_CACHE_VERSION = "twelvelabs-v2"

ANALYSIS_PROMPT = """Analyze this Instagram Reel video in detail and respond ONLY with a valid, raw JSON object (no markdown formatting, no code blocks).
Provide an objective, evidence-based breakdown matching this exact schema:

{
  "niche": "<primary content niche, e.g. fitness, comedy, superhero, tech>",
  "sub_niche": "<specific sub-niche, e.g. cinematic edit, workout routine>",
  "topic": "<specific topic of this reel>",
  "primary_entity": "<key character, person, brand, or subject, e.g. spiderman, batman>",
  "format": "<content format, e.g. cinematic edit, tutorial, reaction, skit>",
  "visual_style": "<visual aesthetics, e.g. high-contrast cinematic, dark grading>",
  "editing_style": "<editing style, e.g. fast-paced beat sync, seamless match cuts>",
  "tone": "<emotional tone, e.g. intense, motivational, humorous, dramatic>",
  "language": "<spoken or written language, e.g. English>",
  "hook": {
    "summary": "<what happens in the first 3 seconds to capture attention>",
    "type": "<visual hook, question, curiosity gap, action sequence>",
    "first_3_seconds": "<exact description of opening 3 seconds>",
    "strength": <score from 0.0 to 1.0 or 1 to 10>,
    "evidence": ["<specific visual/audio cue observed>"]
  },
  "structure": {
    "opening": "<opening setup>",
    "development": "<middle development or sequence>",
    "payoff": "<resolution, climax, or reveal>",
    "cta": "<call to action if present, or null>",
    "evidence": ["<timestamped or sequence evidence>"]
  },
  "visual": {
    "style": "<visual style summary>",
    "scene_count": <estimated number of cuts/scenes>,
    "text_density": "<low, medium, high>",
    "visual_impact": <score from 0.0 to 1.0 or 1 to 10>,
    "evidence": ["<key visual characteristics>"]
  },
  "editing": {
    "pacing": "<fast, moderate, slow>",
    "transitions": "<cut, whip pan, zoom, match cut>",
    "effects": ["<visual effects observed>"],
    "motion_level": "<high, medium, low>"
  },
  "audio": {
    "speech_present": <true or false>,
    "music_present": <true or false>,
    "audio_energy": "<high, medium, low>",
    "sound_effects": <true or false>,
    "quality": <score from 0.0 to 1.0 or 1 to 10>
  },
  "engagement": {
    "curiosity": <score 0.0 to 1.0 or 1 to 10>,
    "novelty": <score 0.0 to 1.0 or 1 to 10>,
    "shareability": <score 0.0 to 1.0 or 1 to 10>,
    "saveability": <score 0.0 to 1.0 or 1 to 10>,
    "comment_potential": <score 0.0 to 1.0 or 1 to 10>
  },
  "discovery_keywords": ["<3-7 multi-word keyword phrases for Instagram discovery>"]
}
"""


def _sha256_file(path: Path | str) -> str:
    """Compute SHA-256 hash of a file for deterministic caching."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TwelveLabsProvider:
    """Primary multimodal video intelligence provider powered by Twelve Labs."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client_instance = None

    def is_configured(self) -> bool:
        """Return True if Twelve Labs API key is set."""
        return bool(settings.TWELVELABS_API_KEY.strip())

    def _get_client(self):
        """Lazy load TwelveLabs client."""
        if self._client_instance is None:
            if not self.is_configured():
                raise ValueError("TWELVELABS_API_KEY is not configured")
            from twelvelabs import TwelveLabs
            self._client_instance = TwelveLabs(api_key=settings.TWELVELABS_API_KEY.strip())
        return self._client_instance

    def _video_context(self, path: Path) -> Any:
        """Create video context for Twelve Labs analyze API call."""
        file_size = path.stat().st_size
        max_base64_size = 30 * 1024 * 1024  # 30 MB

        try:
            from twelvelabs.types import VideoContext_AssetId, VideoContext_Base64String
        except Exception:
            if file_size <= max_base64_size:
                return {"base64": base64.b64encode(path.read_bytes()).decode("ascii")}
            return {"file_path": str(path)}

        if file_size <= max_base64_size:
            return VideoContext_Base64String(
                base_64_string=base64.b64encode(path.read_bytes()).decode("ascii")
            )

        # Direct asset creation for larger files
        client = self._get_client()
        with path.open("rb") as video_file:
            asset = client.assets.create(
                method="direct", file=video_file, filename=path.name
            )
        asset_id = getattr(asset, "id", None)
        if not asset_id:
            raise ValueError("Twelve Labs did not return an asset ID")

        deadline = time.monotonic() + settings.TWELVELABS_REQUEST_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            current = client.assets.retrieve(asset_id)
            status = str(getattr(current, "status", "")).lower()
            if status == "ready":
                return VideoContext_AssetId(asset_id=asset_id)
            if status in {"failed", "error"}:
                raise ValueError(f"Twelve Labs asset processing failed with status {status}")
            time.sleep(2)

        raise TimeoutError(f"Twelve Labs asset processing timed out for {path.name}")

    def _cache_file_for(self, file_path: Path | str, cache_key: str | None = None) -> Path:
        p = Path(file_path)
        if cache_key:
            safe_key = re.sub(r"[^A-Za-z0-9_-]", "_", cache_key)
            return self.cache_dir / f"tl_{safe_key}.json"
        try:
            file_hash = _sha256_file(p)[:16]
        except Exception:
            file_hash = p.name
        return self.cache_dir / f"tl_{file_hash}.json"

    def analyze_video(
        self,
        video_path: str | Path,
        cache_key: str | None = None,
        use_cache: bool = True,
        shared_asset: Any = None,
        profiler: Any = None,
        parent: str | None = "twelvelabs",
    ) -> dict[str, Any]:
        """Analyze a local MP4 file using Twelve Labs multimodal understanding.

        If a SharedVideoAsset is provided, uses its optimized analysis derivative to reduce
        base64 upload payload and latency while preserving full audio/visual content.
        Deterministic cache key is always anchored on the original video.
        """
        path = Path(video_path)
        if not path.exists():
            return {
                "success": False,
                "provider": "twelvelabs",
                "analysis_status": "failed",
                "error": f"VIDEO_ANALYSIS_FAILED: Video file does not exist: {path}",
                "analysis": None,
            }

        if not self.is_configured():
            return {
                "success": False,
                "provider": "twelvelabs",
                "analysis_status": "failed",
                "error": "VIDEO_ANALYSIS_FAILED: TWELVELABS_API_KEY is not configured.",
                "analysis": None,
            }

        t_start = time.perf_counter()
        cache_file = self._cache_file_for(path, cache_key)
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("success") and cached.get("analysis"):
                    if profiler is not None:
                        try:
                            profiler.record_stage_direct(
                                "twelvelabs_cache_lookup",
                                round(time.perf_counter() - t_start, 3),
                                category="LOCAL_IO",
                                parent=parent,
                            )
                        except Exception:
                            pass
                    return cached
            except Exception:
                pass

        max_attempts = max(1, settings.TWELVELABS_RETRY_ATTEMPTS)
        delay = settings.TWELVELABS_REQUEST_DELAY_SECONDS
        last_error = ""

        # Choose upload target: use analysis derivative if available to optimize upload payload
        upload_target = path
        if shared_asset and getattr(shared_asset, "analysis_video_path", None):
            candidate_target = Path(shared_asset.analysis_video_path)
            if candidate_target.exists():
                upload_target = candidate_target

        client = self._get_client()

        for attempt in range(1, max_attempts + 1):
            if attempt > 1 and delay > 0:
                time.sleep(delay)
            try:
                t_total_start = time.perf_counter()
                
                # Measure context preparation / upload
                t_upload_start = time.perf_counter()
                v_ctx = self._video_context(upload_target)
                upload_duration = round(time.perf_counter() - t_upload_start, 3)

                # Measure API analysis
                t_analysis_start = time.perf_counter()
                response = client.analyze(
                    model_name=settings.TWELVELABS_ENGINE,
                    video=v_ctx,
                    prompt=ANALYSIS_PROMPT,
                    max_tokens=4096,
                )
                analysis_duration = round(time.perf_counter() - t_analysis_start, 3)

                # Measure parsing & normalization
                t_parse_start = time.perf_counter()
                raw_text = ""
                raw_data = getattr(response, "data", None)
                if raw_data is not None:
                    raw_text = str(raw_data)
                elif hasattr(response, "text"):
                    raw_text = response.text or ""
                elif isinstance(response, dict):
                    raw_text = response.get("text") or response.get("output") or response.get("data") or ""
                else:
                    raw_text = str(response)

                parsed_json = self._parse_json_defensive(raw_text)
                if not parsed_json:
                    raise ValueError(f"Could not parse valid JSON from Twelve Labs output: {raw_text[:200]}")

                # Normalize hook and visual scores internally to 0.0 - 1.0
                parsed_json = self._normalize_analysis_dict(parsed_json)
                parsing_duration = round(time.perf_counter() - t_parse_start, 3)
                total_duration = round(time.perf_counter() - t_total_start, 3)

                result = {
                    "success": True,
                    "provider": "twelvelabs",
                    "analysis_status": "success",
                    "error": "",
                    "analysis": parsed_json,
                    "telemetry": {
                        "upload_duration_sec": upload_duration,
                        "analysis_duration_sec": analysis_duration,
                        "parsing_duration_sec": parsing_duration,
                        "total_duration_sec": total_duration,
                        "payload_file": str(upload_target),
                        "payload_size_mb": round(upload_target.stat().st_size / (1024 * 1024), 2),
                    },
                }

                if profiler is not None:
                    try:
                        profiler.record_stage_direct(
                            "twelvelabs_upload",
                            upload_duration,
                            category="NETWORK",
                            parent=parent,
                        )
                        profiler.record_stage_direct(
                            "twelvelabs_processing_wait",
                            analysis_duration,
                            category="TWELVE_LABS",
                            parent=parent,
                        )
                        profiler.record_stage_direct(
                            "twelvelabs_result_parsing",
                            parsing_duration,
                            category="LOCAL_CPU",
                            parent=parent,
                        )
                    except Exception:
                        pass

                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

                return result

            except Exception as exc:
                last_error = f"Twelve Labs attempt {attempt} failed: {str(exc)[:250]}"

        return {
            "success": False,
            "provider": "twelvelabs",
            "analysis_status": "failed",
            "error": f"VIDEO_ANALYSIS_FAILED: {last_error}",
            "analysis": None,
        }

    def _parse_json_defensive(self, text: str) -> dict[str, Any] | None:
        """Robust JSON extractor that handles code fences and surrounding commentary."""
        if not text:
            return None
        cleaned = text.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()

        try:
            val = json.loads(cleaned)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                val = json.loads(snippet)
                if isinstance(val, dict):
                    return val
            except Exception:
                pass

        return None

    def _normalize_analysis_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Normalize hook, visual, and engagement metrics to 0.0-1.0 scale."""
        if "hook" in d and isinstance(d["hook"], dict):
            raw_strength = d["hook"].get("strength") or d["hook"].get("score")
            d["hook"]["strength"] = normalize_score_0_to_1(raw_strength)
        if "visual" in d and isinstance(d["visual"], dict):
            raw_impact = d["visual"].get("visual_impact") or d["visual"].get("impact")
            d["visual"]["visual_impact"] = normalize_score_0_to_1(raw_impact)
        if "audio" in d and isinstance(d["audio"], dict):
            raw_qual = d["audio"].get("quality") or d["audio"].get("score")
            d["audio"]["quality"] = normalize_score_0_to_1(raw_qual)
        if "engagement" in d and isinstance(d["engagement"], dict):
            for k in ("curiosity", "novelty", "shareability", "saveability", "comment_potential"):
                if k in d["engagement"]:
                    d["engagement"][k] = normalize_score_0_to_1(d["engagement"][k])
        return d

    def verify_semantic_relevance(
        self,
        uploaded_analysis: Mapping[str, Any] | dict | None,
        competitor_analysis: Mapping[str, Any] | dict | None,
    ) -> dict[str, Any]:
        """Verify that a competitor Reel is genuinely semantically related to the uploaded Reel.

        Builds a transparent relevance evidence object containing:
        - entity_similarity
        - topic_similarity
        - niche_similarity
        - semantic_video_similarity
        - visual_format_similarity
        - twelvelabs_evidence
        - transcript_similarity
        - caption_hashtag_similarity
        - evidence_label ('Entity-based relevance evidence' vs 'Multimodal semantic competitor match')
        """
        if not uploaded_analysis or not competitor_analysis:
            return {
                "video_relevance_eligible": False,
                "video_relevance_score": 0.0,
                "relevance_status": "analyzed_not_relevant",
                "video_verification_status": "analyzed_not_relevant",
                "video_relevance_reason": "Missing analysis for comparison.",
                "evidence_label": "No comparison data",
                "relevance_evidence": {
                    "entity_similarity": 0.0,
                    "topic_similarity": 0.0,
                    "niche_similarity": 0.0,
                    "semantic_video_similarity": 0.0,
                    "visual_format_similarity": 0.0,
                    "transcript_similarity": None,
                    "caption_hashtag_similarity": None,
                    "twelvelabs_evidence": ["Missing video analysis"],
                },
                "video_relevance_dimensions": {},
            }

        u_niche = str(uploaded_analysis.get("niche") or "").lower().strip()
        c_niche = str(competitor_analysis.get("niche") or "").lower().strip()

        u_entity = str(uploaded_analysis.get("primary_entity") or uploaded_analysis.get("primary_subject") or "").lower().strip()
        c_entity = str(competitor_analysis.get("primary_entity") or competitor_analysis.get("primary_subject") or "").lower().strip()

        u_topic = str(uploaded_analysis.get("topic") or "").lower().strip()
        c_topic = str(competitor_analysis.get("topic") or "").lower().strip()

        u_format = str(uploaded_analysis.get("format") or "").lower().strip()
        c_format = str(competitor_analysis.get("format") or "").lower().strip()

        u_sub_niche = str(uploaded_analysis.get("sub_niche") or "").lower().strip()
        c_sub_niche = str(competitor_analysis.get("sub_niche") or "").lower().strip()

        u_edit_type = str(uploaded_analysis.get("edit_type") or "").lower().strip()
        c_edit_type = str(competitor_analysis.get("edit_type") or "").lower().strip()

        # If competitor edit_type not explicitly given, infer from competitor fields
        c_text_combined = f"{c_format} {c_topic} {competitor_analysis.get('caption', '')} {competitor_analysis.get('summary', '')}".lower()
        if not c_edit_type:
            if "cosplay" in c_text_combined:
                c_edit_type = "cosplay"
            elif any(w in c_text_combined for w in ["review", "analysis", "breakdown", "theory"]):
                c_edit_type = "review"
            elif "reaction" in c_text_combined:
                c_edit_type = "reaction"
            elif any(w in c_text_combined for w in ["meme", "funny", "comedy", "parody"]):
                c_edit_type = "meme edit"
            elif any(w in c_text_combined for w in ["news", "update", "rumor", "announcement"]):
                c_edit_type = "news"
            elif any(w in c_text_combined for w in ["montage", "action", "fight", "battle"]):
                c_edit_type = "action montage"
            elif any(w in c_text_combined for w in ["emotional", "sad", "tribute"]):
                c_edit_type = "emotional edit"
            elif any(w in c_text_combined for w in ["dialogue", "quote", "speech"]):
                c_edit_type = "dialogue edit"
            elif any(w in c_text_combined for w in ["vfx", "cgi", "after effects", "effects"]):
                c_edit_type = "vfx edit"
            elif any(w in c_text_combined for w in ["tutorial", "how to", "workout", "routine"]):
                c_edit_type = "tutorial"

        # 1. Entity similarity
        entity_score = 0.0
        if u_entity and c_entity:
            u_clean = u_entity.replace("-", " ")
            c_clean = c_entity.replace("-", " ")
            if u_entity in c_entity or c_entity in u_entity or u_clean in c_clean or c_clean in u_clean:
                entity_score = 1.0
            else:
                u_words = set(u_clean.split())
                c_words = set(c_clean.split())
                if u_words & c_words:
                    entity_score = 0.8
        elif not u_entity and not c_entity:
            entity_score = 0.5

        # 2. Niche similarity
        niche_score = 0.0
        if u_niche and c_niche:
            if u_niche == c_niche or u_niche in c_niche or c_niche in u_niche:
                niche_score = 1.0
            else:
                niche_score = 0.2
        elif not u_niche and not c_niche:
            niche_score = 0.4

        # 3. Topic similarity
        topic_score = 0.0
        if u_topic and c_topic:
            u_top_words = set(u_topic.split())
            c_top_words = set(c_topic.split())
            overlap = u_top_words & c_top_words
            if overlap:
                topic_score = min(1.0, len(overlap) / max(1, len(u_top_words)))
            elif u_topic in c_topic or c_topic in u_topic:
                topic_score = 0.8
            else:
                topic_score = 0.15

        # 4. Format similarity
        format_score = 0.0
        if u_format and c_format:
            if u_format == c_format or u_format in c_format or c_format in u_format:
                format_score = 1.0
            else:
                format_score = 0.35

        # 5. Sub-Niche similarity
        sub_niche_score = 0.5
        if u_sub_niche and c_sub_niche:
            if u_sub_niche == c_sub_niche or u_sub_niche in c_sub_niche or c_sub_niche in u_sub_niche:
                sub_niche_score = 1.0
            else:
                u_sn_words = set(u_sub_niche.split())
                c_sn_words = set(c_sub_niche.split())
                overlap = u_sn_words & c_sn_words
                sub_niche_score = min(1.0, len(overlap) / max(1, len(u_sn_words))) if overlap else 0.2

        # 6. Edit-Type similarity & Incompatible Content Detection
        incompatible_genre = None
        edit_type_score = 0.5
        is_action_edit = any(
            t in u_edit_type for t in ["action montage", "vfx edit", "cinematic", "fight"]
        )

        incompatible_terms = ["cosplay", "review", "reaction", "meme", "news", "vlog", "unboxing", "interview", "podcast"]
        for term in incompatible_terms:
            if term in c_text_combined or term == c_edit_type:
                incompatible_genre = term
                break

        if is_action_edit and incompatible_genre:
            edit_type_score = 0.0
        elif u_edit_type and c_edit_type:
            if u_edit_type == c_edit_type:
                edit_type_score = 1.0
            elif (is_action_edit and c_edit_type in ["action montage", "vfx edit", "cinematic edit"]):
                edit_type_score = 0.8
            else:
                edit_type_score = 0.4
        else:
            edit_type_score = 0.5

        # 7. Transcript / Text similarity if available
        u_trans = str(uploaded_analysis.get("transcript") or "").lower()
        c_trans = str(competitor_analysis.get("transcript") or "").lower()
        trans_sim = None
        if u_trans and c_trans:
            u_tw = set(u_trans.split()[:40])
            c_tw = set(c_trans.split()[:40])
            t_overlap = u_tw & c_tw
            trans_sim = round(len(t_overlap) / max(1, len(u_tw)), 3) if u_tw else 0.0

        # Weighted aggregate score
        if u_sub_niche or u_edit_type:
            weights = {"entity": 0.30, "niche": 0.15, "sub_niche": 0.15, "topic": 0.20, "edit_type": 0.10, "format": 0.10}
            total_score = (
                weights["entity"] * entity_score
                + weights["niche"] * niche_score
                + weights["sub_niche"] * sub_niche_score
                + weights["topic"] * topic_score
                + weights["edit_type"] * edit_type_score
                + weights["format"] * format_score
            )
        else:
            weights = {"entity": 0.35, "niche": 0.25, "topic": 0.25, "format": 0.15}
            total_score = (
                weights["entity"] * entity_score
                + weights["niche"] * niche_score
                + weights["topic"] * topic_score
                + weights["format"] * format_score
            )
        total_score = round(min(1.0, max(0.0, total_score)), 4)

        # Incompatibility penalty
        if is_action_edit and incompatible_genre:
            total_score = min(total_score, 0.35)

        # Section 11: Guard against entity-alone match ("spider-man matched spider-man" alone is not sufficient)
        # Requires multimodal semantic alignment
        is_eligible = (
            not (is_action_edit and incompatible_genre)
            and (
                (entity_score >= 0.7 and (niche_score >= 0.3 or topic_score >= 0.3))
                or (niche_score >= 0.7 and topic_score >= 0.35)
                or (total_score >= 0.50 and (entity_score > 0.4 or niche_score > 0.4))
            )
        )

        tl_evidence = []
        if entity_score >= 0.7:
            tl_evidence.append(f"Entity alignment: '{u_entity}' ~ '{c_entity}' (score: {entity_score:.2f})")
        if niche_score >= 0.7:
            tl_evidence.append(f"Niche alignment: '{u_niche}' ~ '{c_niche}' (score: {niche_score:.2f})")
        if sub_niche_score >= 0.7:
            tl_evidence.append(f"Sub-niche alignment: '{u_sub_niche}' ~ '{c_sub_niche}' (score: {sub_niche_score:.2f})")
        if topic_score >= 0.3:
            tl_evidence.append(f"Topic alignment: '{u_topic}' ~ '{c_topic}' (score: {topic_score:.2f})")
        if edit_type_score >= 0.7:
            tl_evidence.append(f"Edit-type alignment: '{u_edit_type}' ~ '{c_edit_type}' (score: {edit_type_score:.2f})")
        if format_score >= 0.7:
            tl_evidence.append(f"Format alignment: '{u_format}' ~ '{c_format}' (score: {format_score:.2f})")
        if is_action_edit and incompatible_genre:
            tl_evidence.append(f"Rejected incompatible content type '{incompatible_genre}' for uploaded action montage")

        # Explicit labeling
        if is_action_edit and incompatible_genre:
            evidence_label = f"Incompatible content type ({incompatible_genre})"
        elif entity_score >= 0.7 and niche_score < 0.4 and topic_score < 0.3:
            evidence_label = "Entity-based relevance evidence"
        elif is_eligible:
            evidence_label = "Multimodal semantic competitor match"
        else:
            evidence_label = "Insufficient semantic overlap"

        status = "verified_relevant" if is_eligible else "analyzed_not_relevant"
        reasons = []
        if is_eligible:
            reasons.append(evidence_label)
            reasons.extend(tl_evidence)
        elif is_action_edit and incompatible_genre:
            reasons.append(f"Rejected incompatible content type '{incompatible_genre}' (action montage uploaded)")
        else:
            reasons.append(f"Insufficient semantic overlap across niche/topic/entity (score: {total_score:.2f})")

        relevance_evidence = {
            "entity_similarity": entity_score,
            "topic_similarity": topic_score,
            "niche_similarity": niche_score,
            "sub_niche_similarity": sub_niche_score,
            "edit_type_similarity": edit_type_score,
            "semantic_video_similarity": total_score,
            "visual_format_similarity": format_score,
            "transcript_similarity": trans_sim,
            "caption_hashtag_similarity": None,
            "twelvelabs_evidence": tl_evidence,
            "evidence_label": evidence_label,
            "entity_match": entity_score,
            "topic_match": topic_score,
            "niche_match": niche_score,
            "sub_niche_match": sub_niche_score,
            "edit_type_match": edit_type_score,
            "format_match": format_score,
            "overall_relevance": total_score,
        }

        return {
            "video_relevance_eligible": is_eligible,
            "video_relevance_score": total_score,
            "relevance_status": status,
            "video_verification_status": status,
            "evidence_label": evidence_label,
            "video_relevance_reason": "; ".join(reasons),
            "relevance_evidence": relevance_evidence,
            "video_relevance_dimensions": {
                "entity_similarity": entity_score,
                "niche_similarity": niche_score,
                "sub_niche_similarity": sub_niche_score,
                "topic_similarity": topic_score,
                "edit_type_similarity": edit_type_score,
                "format_similarity": format_score,
                "entity_match": entity_score,
                "topic_match": topic_score,
                "niche_match": niche_score,
                "sub_niche_match": sub_niche_score,
                "edit_type_match": edit_type_score,
                "format_match": format_score,
                "overall_relevance": total_score,
            },
            "entity_match": entity_score,
            "topic_match": topic_score,
            "niche_match": niche_score,
            "sub_niche_match": sub_niche_score,
            "edit_type_match": edit_type_score,
            "format_match": format_score,
            "overall_relevance": total_score,
        }
