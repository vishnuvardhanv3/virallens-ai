"""Master Agent - Complete End-to-End Orchestrator for ViralLens AI.

Enforces:
1. Twelve Labs native MP4 analysis for uploaded Reel.
2. Human Attention Model inference (laboratory-derived visual attention potential).
3. Specialized multi-agent evaluation:
   - Human Attention Agent
   - Human Behavior Agent
   - Emotion Agent
   - Hook Agent
   - Editing Agent
   - Communication Agent
4. Query Generation: 3-5 multi-word, niche-anchored queries via Discovery Agent.
5. Apify Instagram Reel Discovery: multi-word preservation, normalization, deduplication.
6. Metadata relevance ranking & selection (capped at 10 competitors).
7. Real MP4 download & validation.
8. Twelve Labs video analysis for EVERY downloaded competitor.
9. Semantic relevance verification (verified_relevant vs analyzed_not_relevant).
10. Attention Model inference for EVERY downloaded verified competitor MP4.
11. Competitor Attention Benchmark Comparison.
12. Pattern Discovery exclusively on verified competitors.
13. Strategy Agent evidence-backed action items.
14. Multi-agent conflict resolution using evidence density, confidence, and source reliability.
15. Final structured report generation with strict non-causal terminology.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

from config import settings
from providers.twelvelabs_provider import TwelveLabsProvider
from utils.query_builder import QueryBuilder
from instagram.provider_router import InstagramProviderRouter
from services.discovery_profile import build_discovery_profile
from services.relevance import filter_relevant_candidates, score_video_relevance
from services.media_service import acquire, validate_video_file
from services.canonical_schema import build_canonical_analysis, normalize_score_0_to_1
from services.competitor_dataset_service import save_competitor_dataset
from services.report_service import save_report
from services.human_attention_service import HumanAttentionService
from services.attention_comparison_service import AttentionComparisonService
from services.attention_intelligence import AttentionIntelligenceService, render_attention_visualization
from services.profiler import PipelineProfiler, TimingCategory
from services.video_asset_service import SharedVideoAsset
from services.competitor_cache import CompetitorArtifactCache, get_canonical_reel_id

from agents.video_agent import analyze_video
from agents.audio_agent import analyze_audio
from agents.speech_agent import transcribe
from agents.ocr_agent import extract_ocr
from agents.competitor_agent import rank_candidates, select_final_competitors, cluster_competitors
from agents.comparison_agent import compare
from agents.trend_agent import analyze_trends
from agents.virality_agent import ViralityAgent
from agents.human_attention_agent import HumanAttentionAgent
from agents.hook_agent import HookAgent
from agents.human_behavior_agent import HumanBehaviorAgent
from agents.emotion_agent import EmotionAgent
from agents.editing_agent import EditingAgent
from agents.communication_agent import CommunicationAgent
from agents.discovery_agent import DiscoveryAgent
from agents.pattern_agent import PatternAgent
from agents.strategy_agent import StrategyAgent
from ml.pattern_discovery import PatternDiscovery


from core.startup_check import validate_system_health, HealthState


class MasterAgent:
    """Master pipeline coordinator."""

    def __init__(
        self,
        progress_callback: Callable[[str], None] | None = None,
        on_target_ready: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.on_target_ready = on_target_ready
        self.health_report = validate_system_health()
        self.analyzer = TwelveLabsProvider()
        self.router = InstagramProviderRouter()
        self.ml = PatternDiscovery()
        self.competitor_cache = CompetitorArtifactCache()
        try:
            self.attention_service = HumanAttentionService()
        except Exception:
            self.attention_service = None
        try:
            from services.shadow_spatial_service import ShadowSpatialService
            self.shadow_spatial_service = ShadowSpatialService()
        except Exception:
            self.shadow_spatial_service = None
        try:
            self.attention_intelligence_service = AttentionIntelligenceService()
        except Exception:
            self.attention_intelligence_service = None

    def _log(self, message: str) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(f"{datetime.now().strftime('%H:%M:%S')}  {message}")
            except Exception as exc:
                logger.debug("Progress callback error: %s", exc)

    def analyze_one(
        self,
        path: str | Path,
        stage: str = "Uploaded Reel",
        shared_asset: Any = None,
        profiler: Any = None,
        parent: str | None = None,
    ) -> dict[str, Any]:
        """Runs Twelve Labs video analysis plus supplemental objective local signals."""
        p = Path(path)
        self._log(f"{stage}: starting Twelve Labs multimodal analysis")
        tl_res = self.analyzer.analyze_video(p, shared_asset=shared_asset, profiler=profiler, parent=parent)

        if tl_res.get("success"):
            self._log(f"{stage}: Twelve Labs analysis complete")
        else:
            err = str(tl_res.get("error", "Analysis failed"))[:150]
            self._log(f"{stage}: Twelve Labs analysis failed: {err}")

        # Local technical signals
        self._log(f"{stage}: extracting OpenCV visual signals")
        video_tech = analyze_video(p, shared_asset=shared_asset)

        self._log(f"{stage}: extracting librosa audio signals")
        audio_tech = analyze_audio(p)

        self._log(f"{stage}: transcribing speech locally with Whisper")
        speech_tech = transcribe(p)

        self._log(f"{stage}: extracting on-screen text with OCR")
        ocr_tech = extract_ocr(p, shared_asset=shared_asset, profiler=profiler)

        analysis_dict = tl_res.get("analysis") or {}

        # Canonical assembly
        canonical = build_canonical_analysis(
            analysis=analysis_dict,
            video=video_tech,
            audio={**audio_tech, "transcript": speech_tech.get("speech_text")},
            visual=analysis_dict.get("visual", {}),
            hook=analysis_dict.get("hook", {}),
            structure=analysis_dict.get("structure", {}),
            engagement=analysis_dict.get("engagement", {}),
            transcript=speech_tech.get("speech_text"),
            ocr_text=ocr_tech.get("ocr_text"),
            ocr=ocr_tech,
        )

        return {
            "niche": canonical.get("niche") or "creative",
            "sub_niche": canonical.get("sub_niche") or "",
            "topic": canonical.get("topic") or "general content",
            "primary_entity": canonical.get("primary_entity") or "",
            "format": canonical.get("format") or "cinematic edit",
            "visual_style": canonical.get("style") or "",
            "editing_style": canonical.get("editing_style") or "",
            "tone": canonical.get("tone") or "dynamic",
            "language": canonical.get("language") or "English",
            "hook": canonical.get("hook") or {},
            "visual": canonical.get("visual") or {},
            "structure": canonical.get("structure") or {},
            "audio": canonical.get("audio") or {},
            "speech": canonical.get("speech") or {},
            "ocr": canonical.get("ocr") or ocr_tech,
            "onscreen_text": canonical.get("onscreen_text") or ocr_tech.get("ocr_text", ""),
            "keywords": canonical.get("keywords") or [],
            "analysis": analysis_dict if tl_res.get("success") else None,
            "analysis_status": "success" if tl_res.get("success") else "failed",
            "analysis_error": tl_res.get("error", ""),
            "canonical": canonical,
            "technical_signals": {
                "video": video_tech,
                "audio": audio_tech,
                "speech": speech_tech,
                "ocr": ocr_tech,
            },
        }

    def resolve_agent_conflicts(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Systematically resolves disagreements among specialized agents based on empirical weighting."""
        resolutions = []

        # Example conflict: Hook Agent vs Human Attention Agent on opening strength
        hook_res = agent_results.get("hook", {})
        att_res = agent_results.get("human_attention", {})
        hook_att = hook_res.get("attention_score")
        att_curve = att_res.get("attention_curve", [])

        if hook_att is not None and att_curve:
            opening_mean = sum(att_curve[: min(6, len(att_curve))]) / max(1, min(6, len(att_curve)))
            discrepancy = abs(hook_att - opening_mean)
            if discrepancy > 0.15:
                resolutions.append({
                    "agents": ["hook", "human_attention"],
                    "conflict": "Opening attention score divergence between hook window and regression curve.",
                    "resolution": "Adopting empirical regression curve mean (source: EyeLink gaze model) over heuristic hook classification.",
                    "consensus_score": round(opening_mean, 4),
                })

        # Example conflict: Editing cut rate vs Visual pace
        edit_res = agent_results.get("editing", {})
        if edit_res.get("cut_rate", 0) > 1.2 and edit_res.get("pacing") != "kinetic_rapid_fire":
            resolutions.append({
                "agents": ["editing", "video"],
                "conflict": "Cut rate exceeds 1.2 cuts/sec while visual pace was tagged moderate.",
                "resolution": "Prioritizing objective OpenCV pixel difference cut detection; pace resolved as kinetic.",
            })

        return {
            "conflict_count": len(resolutions),
            "resolutions": resolutions,
        }

    def run(self, uploaded_video_path: str | Path) -> dict[str, Any]:
        """Execute the full end-to-end ViralLens AI intelligence pipeline with unified profiling."""
        import concurrent.futures

        video_path = Path(uploaded_video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        profiler = PipelineProfiler(video_path)
        self._log(f"Pipeline: started for {video_path.name}")

        # Spawn true non-blocking shadow spatial analysis in an isolated background thread
        main_pipeline_start = time.perf_counter()
        shadow_future = None
        shadow_executor = None
        if getattr(self, "shadow_spatial_service", None) and self.shadow_spatial_service.is_available:
            try:
                shadow_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="ShadowSpatialWorker"
                )
                shadow_future = shadow_executor.submit(
                    self.shadow_spatial_service.analyze_video, video_path
                )
                self._log("Shadow Spatial Saliency: background execution started")
            except Exception as e:
                self._log(f"Shadow Spatial Saliency: failed to spawn worker: {e}")

        # Step 0: Video Probing & Analysis Asset Creation
        with profiler.profile_stage("stage_a_asset_preparation"):
            self._log(f"Asset Preparation: probing {video_path.name} ({round(video_path.stat().st_size / (1024*1024), 2)} MB)")
            with profiler.profile_stage("stage_a_probing", category=TimingCategory.LOCAL_CPU, parent="stage_a_asset_preparation"):
                shared_asset = SharedVideoAsset(video_path)
            with profiler.profile_stage("stage_a_derivative", category=TimingCategory.LOCAL_IO, parent="stage_a_asset_preparation"):
                shared_asset.generate_derivative()
            with profiler.profile_stage("stage_a_audio", category=TimingCategory.LOCAL_IO, parent="stage_a_asset_preparation"):
                shared_asset.extract_audio()
            if shared_asset.has_derivative:
                self._log(
                    f"Asset Preparation: created analysis derivative "
                    f"({round(Path(shared_asset.analysis_video_path).stat().st_size / (1024*1024), 2)} MB, "
                    f"ratio: {shared_asset.meta.compression_ratio:.1%})"
                )

        # Step 1: Twelve Labs Multimodal Video Understanding
        with profiler.profile_stage("stage_b_twelvelabs_analysis"):
            self._log("Uploaded Reel: starting Twelve Labs multimodal analysis")
            tl_res = self.analyzer.analyze_video(
                video_path, shared_asset=shared_asset, profiler=profiler, parent="stage_b_twelvelabs_analysis"
            )
            if tl_res.get("success"):
                self._log("Uploaded Reel: Twelve Labs analysis complete")
            else:
                err = str(tl_res.get("error", "Analysis failed"))[:150]
                self._log(f"Uploaded Reel: Twelve Labs analysis failed: {err}")

        # Step 2: Concurrent Local Technical Signals & Human Attention Inference
        with profiler.profile_stage("stage_c_local_signals_and_attention", concurrency_level=3):
            self._log("Uploaded Reel: extracting local video, audio, OCR, and attention model concurrently")

            def _run_ocr():
                with profiler.profile_stage("stage_c_ocr", category=TimingCategory.LOCAL_CPU, parent="stage_c_local_signals_and_attention"):
                    return extract_ocr(video_path, shared_asset=shared_asset, profiler=profiler)

            def _run_attention():
                with profiler.profile_stage("stage_c_attention", category=TimingCategory.LOCAL_CPU, parent="stage_c_local_signals_and_attention"):
                    if self.attention_service:
                        try:
                            return self.attention_service.predict_attention(
                                video_path, audio_wav_path=shared_asset.audio_wav_path
                            )
                        except Exception as e:
                            self._log(f"Human Attention Model: inference failed: {e}")
                            return []
                    return []

            def _run_media_tech():
                with profiler.profile_stage("stage_c_opencv", category=TimingCategory.LOCAL_CPU, parent="stage_c_local_signals_and_attention"):
                    v = analyze_video(video_path, shared_asset=shared_asset)
                with profiler.profile_stage("stage_c_librosa", category=TimingCategory.LOCAL_CPU, parent="stage_c_local_signals_and_attention"):
                    a = analyze_audio(video_path)
                with profiler.profile_stage("stage_c_whisper", category=TimingCategory.LOCAL_CPU, parent="stage_c_local_signals_and_attention"):
                    s = transcribe(video_path)
                return v, a, s

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_ocr = executor.submit(_run_ocr)
                future_att = executor.submit(_run_attention)
                future_tech = executor.submit(_run_media_tech)

                ocr_tech = future_ocr.result()
                attention_predictions = future_att.result()
                video_tech, audio_tech, speech_tech = future_tech.result()

            analysis_dict = tl_res.get("analysis") or {}
            canonical = build_canonical_analysis(
                analysis=analysis_dict,
                video=video_tech,
                audio={**audio_tech, "transcript": speech_tech.get("speech_text")},
                visual=analysis_dict.get("visual", {}),
                hook=analysis_dict.get("hook", {}),
                structure=analysis_dict.get("structure", {}),
                engagement=analysis_dict.get("engagement", {}),
                transcript=speech_tech.get("speech_text"),
                ocr_text=ocr_tech.get("ocr_text"),
                ocr=ocr_tech,
            )

            your_reel = {
                "niche": canonical.get("niche") or "creative",
                "sub_niche": canonical.get("sub_niche") or "",
                "topic": canonical.get("topic") or "general content",
                "primary_entity": canonical.get("primary_entity") or "",
                "format": canonical.get("format") or "cinematic edit",
                "visual_style": canonical.get("style") or "",
                "editing_style": canonical.get("editing_style") or "",
                "tone": canonical.get("tone") or "dynamic",
                "language": canonical.get("language") or "English",
                "hook": canonical.get("hook") or {},
                "visual": canonical.get("visual") or {},
                "structure": canonical.get("structure") or {},
                "audio": canonical.get("audio") or {},
                "speech": canonical.get("speech") or {},
                "ocr": canonical.get("ocr") or ocr_tech,
                "onscreen_text": canonical.get("onscreen_text") or ocr_tech.get("ocr_text", ""),
                "keywords": canonical.get("keywords") or [],
                "analysis": analysis_dict if tl_res.get("success") else None,
                "analysis_status": "success" if tl_res.get("success") else "failed",
                "analysis_error": tl_res.get("error", ""),
                "canonical": canonical,
                "technical_signals": {
                    "video": video_tech,
                    "audio": audio_tech,
                    "speech": speech_tech,
                    "ocr": ocr_tech,
                },
                "attention_predictions": attention_predictions,
            }

            # Retrieve shadow spatial analysis results (non-blocking if already finished)
            dhf1k_spatial_signal = None
            if shadow_future is not None:
                try:
                    dhf1k_spatial_signal = shadow_future.result(timeout=15.0)
                    self._log("Shadow Spatial Saliency: background execution completed successfully")
                except Exception as e:
                    dhf1k_spatial_signal = {
                        "status": "FAILED",
                        "error": f"Shadow spatial worker error: {e}",
                        "samples": [],
                        "telemetry": {}
                    }
                    self._log(f"Shadow Spatial Saliency: error retrieving result: {e}")
                if shadow_executor:
                    shadow_executor.shutdown(wait=False)
                shadow_future = None  # prevent duplicate retrieval

            # Stage C2: Attention Intelligence Synthesis
            attention_intelligence = None
            if getattr(self, "attention_intelligence_service", None):
                try:
                    attention_intelligence = self.attention_intelligence_service.build_intelligence(
                        nemar_predictions=attention_predictions,
                        spatial_data=dhf1k_spatial_signal or {},
                        context=your_reel,
                        source_video_path=video_path,
                    )
                    self._log("Attention Intelligence: unified temporal-spatial intelligence synthesized")
                except Exception as e:
                    self._log(f"Attention Intelligence: synthesis error: {e}")
                    attention_intelligence = {
                        "schema_version": "1.0.0",
                        "status": "FAILED",
                        "error": str(e),
                    }

            your_reel["nemar_attention_signal"] = attention_predictions
            your_reel["dhf1k_spatial_saliency_signal"] = dhf1k_spatial_signal
            your_reel["attention_intelligence"] = attention_intelligence

        # Step 3: Run specialized domain agents on uploaded Reel
        with profiler.profile_stage("stage_d_domain_agents"):
            self._log("Multi-Agent System: executing specialized analytical agents")
            human_attention_res = HumanAttentionAgent.analyze(
                attention_predictions, your_reel.get("analysis"), your_reel.get("technical_signals")
            )
            hook_res = HookAgent.analyze(
                attention_predictions, your_reel.get("analysis"), your_reel.get("technical_signals")
            )
            behavior_res = HumanBehaviorAgent.analyze(
                your_reel.get("analysis"), your_reel.get("technical_signals"), attention_predictions
            )
            emotion_res = EmotionAgent.analyze(
                your_reel.get("analysis"), your_reel.get("technical_signals"), attention_predictions
            )
            editing_res = EditingAgent.analyze(
                your_reel.get("technical_signals"), attention_predictions, your_reel.get("analysis")
            )
            comm_res = CommunicationAgent.analyze(
                your_reel.get("technical_signals"), your_reel.get("analysis"), attention_predictions
            )

            agent_results = {
                "human_attention": human_attention_res,
                "hook": hook_res,
                "human_behavior": behavior_res,
                "emotion": emotion_res,
                "editing": editing_res,
                "communication": comm_res,
                "ocr": your_reel.get("ocr", {}),
                "attention_intelligence": attention_intelligence,
            }

        # Step 4: Build discovery profile & Discovery Agent multi-word queries
        with profiler.profile_stage("stage_e_query_generation"):
            disc_profile = build_discovery_profile(your_reel)
            discovery_res = DiscoveryAgent.generate_queries(disc_profile, max_queries=5)
            queries = discovery_res.get("queries", [])
            agent_results["discovery"] = discovery_res

            self._log(f"Discovery Agent: {len(queries)} niche-anchored multi-word queries generated")
            for idx, q in enumerate(queries, 1):
                self._log(f"  Query {idx}/{len(queries)}: '{q['query']}' [{q.get('search_level', 'CORE')}]")

            # Expose Target Video Intelligence as soon as ready before discovery (Section 4)
            if self.on_target_ready:
                try:
                    self.on_target_ready({
                        "your_reel": your_reel,
                        "discovery_profile": disc_profile,
                        "queries": queries,
                        "agent_results": agent_results,
                    })
                except Exception as e:
                    self._log(f"Target Ready Callback warning: {e}")

        # Step 5: Apify discovery
        with profiler.profile_stage("stage_f_instagram_discovery"):
            self._log(f"Discovery: executing Apify Instagram search across {len(queries)} queries")
            raw_candidates, discovery_diag = self.router.search_with_diagnostics(
                queries,
                limit=settings.INSTAGRAM_MAX_RESULTS,
                profiler=profiler,
                progress_callback=self._log,
            )
            self._log(f"Discovery: raw candidates = {discovery_diag.get('raw_candidates', len(raw_candidates))}")

            disc_health = discovery_diag.get("discovery_health")
            if disc_health == "failed":
                if discovery_diag.get("blocked_queries"):
                    self._log(
                        "Instagram discovery was blocked for all attempted queries. No verified competitors were generated."
                    )
                else:
                    self._log("Instagram discovery returned zero results across all attempted queries.")
            elif disc_health in {"partial", "degraded"}:
                blocked_count = len(discovery_diag.get("blocked_queries", []))
                self._log(
                    f"Discovery: {disc_health.upper()} health ({blocked_count} queries blocked, {len(raw_candidates)} candidates retrieved from successful queries)"
                )

        # Step 6-11: Competitor filtering, download, and multi-modal analysis
        with profiler.profile_stage("stage_g_competitor_download_and_analysis", concurrency_level=2):
            if len(raw_candidates) == 0:
                self._log("Competitor discovery did not produce usable results; competitor analysis skipped.")
                relevant_candidates = []
                relevance_diag = {"total_candidates": 0, "relevant_candidates": 0, "rejected_candidates": 0}
                ranked_candidates = []
                selected_candidates = []
                downloaded_count = 0
                analyzed_competitors = []
                verified_competitors = []
                rejected_competitors = []
                final_verified = []
                diversity_diag = {"unique_creators": 0, "selection_counts": {}}
                competitor_clusters = cluster_competitors([], user_edit_type="action montage")
                duplicate_tracker = {}
            else:
                # Step 6: Metadata relevance filtering
                with profiler.profile_stage("stage_g_metadata_filter", category=TimingCategory.LOCAL_CPU, parent="stage_g_competitor_download_and_analysis"):
                    relevant_candidates, relevance_diag = filter_relevant_candidates(
                        raw_candidates, disc_profile, minimum_score=0.10
                    )
                self._log(
                    f"Metadata filter: kept {len(relevant_candidates)} candidates; rejected {relevance_diag['rejected_candidates']}"
                )

                # Step 7: Ranking & Selection with Canonical Deduplication
                with profiler.profile_stage("stage_g_ranking", category=TimingCategory.LOCAL_CPU, parent="stage_g_competitor_download_and_analysis"):
                    ranked_candidates = rank_candidates(relevant_candidates)
                    
                    # Canonical Reel Deduplication before selection
                    seen_cids: set[str] = set()
                    deduped_ranked: list[dict[str, Any]] = []
                    for c in ranked_candidates:
                        cid = get_canonical_reel_id(c)
                        if cid and cid != "unknown" and cid in seen_cids:
                            continue
                        if cid and cid != "unknown":
                            seen_cids.add(cid)
                        deduped_ranked.append(c)

                    max_comps = min(getattr(settings, "TWELVELABS_MAX_COMPETITORS", 10), 10)
                    selected_candidates = deduped_ranked[:max_comps]
                self._log(f"Ranking: selected top {len(selected_candidates)} candidate(s) (cap={max_comps})")

                # Step 8, 9, 10, 11: Competitor Processing with bounded concurrency (max_workers=2)
                downloaded_count = 0
                duplicate_tracker: Dict[str, Dict[str, int]] = {}
                for cand in selected_candidates:
                    cid = get_canonical_reel_id(cand)
                    duplicate_tracker[cid] = {
                        "download_count": 0,
                        "upload_count": 0,
                        "analysis_count": 0,
                        "ocr_count": 0,
                        "verification_count": 0,
                    }

                def _process_competitor_task(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
                    idx, cand = item
                    cid = get_canonical_reel_id(cand) or f"cand_{idx}"
                    creator = str(cand.get("creator") or cand.get("username") or "unknown")
                    self._log(f"Competitor {idx}/{len(selected_candidates)} [{cid}]: starting acquisition/analysis")

                    # Download with bounded timing
                    t_dl_start = time.perf_counter()
                    media_info = acquire(cand, settings.MEDIA_DIR)
                    t_dl_end = time.perf_counter()
                    local_path = media_info.get("local_path")
                    dl_success = bool(media_info.get("video_download_status") == "DOWNLOADED" and local_path and os.path.exists(local_path))
                    file_size = int(media_info.get("size_bytes", 0))

                    profiler.record_stage_direct(
                        stage_name=f"competitor_{idx}_{cid}_download",
                        duration=round(t_dl_end - t_dl_start, 3),
                        category=TimingCategory.COMPETITOR_DOWNLOAD,
                        parent="stage_g_competitor_download_and_analysis",
                        metadata={
                            "canonical_id": cid,
                            "creator": creator,
                            "file_size": file_size,
                            "cached": media_info.get("cached", False),
                            "success": dl_success,
                        },
                        start_time=t_dl_start,
                        end_time=t_dl_end,
                    )

                    if not dl_success:
                        cand["status"] = "FAILED"
                        cand["download_status"] = "DOWNLOAD_FAILED"
                        cand["error"] = media_info.get("error", "Download failed")
                        self._log(f"Download: competitor {idx}/{len(selected_candidates)} [{cid}] failed: {cand['error']}")
                        return cand

                    if not media_info.get("cached", False):
                        duplicate_tracker[cid]["download_count"] += 1
                    cand["status"] = "DOWNLOADED"
                    cand["download_status"] = "DOWNLOADED"
                    cand["local_media_path"] = local_path

                    # Check Competitor Artifact Cache
                    cached_artifact = self.competitor_cache.get(cid, profiler=profiler)
                    if (
                        cached_artifact
                        and cached_artifact.get("comp_analysis")
                        and cached_artifact.get("comp_analysis", {}).get("analysis_status") == "success"
                        and cached_artifact.get("attention_predictions") is not None
                    ):
                        comp_analysis = cached_artifact["comp_analysis"]
                        comp_att = cached_artifact["attention_predictions"]
                        relevance_res = cached_artifact.get("relevance_res")
                        if not relevance_res:
                            relevance_res = score_video_relevance(
                                your_reel.get("analysis") or your_reel,
                                comp_analysis.get("analysis") or comp_analysis,
                            )
                        cand["status"] = "ANALYZED"
                        cand["analysis"] = comp_analysis
                        cand["hook"] = comp_analysis.get("hook")
                        cand["visual"] = comp_analysis.get("visual")
                        cand["audio"] = comp_analysis.get("audio")
                        cand["niche"] = comp_analysis.get("niche")
                        cand["topic"] = comp_analysis.get("topic")
                        cand["attention_predictions"] = comp_att
                        cand["video_relevance_score"] = relevance_res.get("video_relevance_score")
                        cand["video_relevance_reason"] = relevance_res.get("video_relevance_reason")
                        cand["video_verification_status"] = relevance_res.get("relevance_status")
                        if relevance_res.get("video_relevance_eligible"):
                            cand["status"] = "VERIFIED"
                            self._log(f"Relevance: competitor {idx}/{len(selected_candidates)} [{cid}] VERIFIED (cached)")
                        else:
                            cand["status"] = "REJECTED"
                            self._log(f"Relevance: competitor {idx}/{len(selected_candidates)} [{cid}] REJECTED (cached)")
                        return cand

                    # SharedVideoAsset for competitor (derivatives + pre-extracted audio WAV)
                    comp_asset = None
                    try:
                        comp_asset = SharedVideoAsset(local_path)
                        comp_asset.generate_derivative()
                        comp_asset.extract_audio()
                    except Exception:
                        comp_asset = None

                    # Twelve Labs Analysis
                    t_tl_start = time.perf_counter()
                    self._log(f"Twelve Labs: competitor {idx}/{len(selected_candidates)} [{cid}] video analysis started")
                    comp_analysis = self.analyze_one(
                        local_path,
                        stage=f"Competitor {idx}/{len(selected_candidates)}",
                        shared_asset=comp_asset,
                        profiler=profiler,
                        parent="stage_g_competitor_download_and_analysis",
                    )
                    t_tl_end = time.perf_counter()
                    tl_ok = bool(comp_analysis.get("analysis_status") == "success")

                    profiler.record_stage_direct(
                        stage_name=f"competitor_{idx}_{cid}_twelvelabs",
                        duration=round(t_tl_end - t_tl_start, 3),
                        category=TimingCategory.TWELVE_LABS,
                        parent="stage_g_competitor_download_and_analysis",
                        metadata={
                            "canonical_id": cid,
                            "success": tl_ok,
                            "error": comp_analysis.get("analysis_error", ""),
                        },
                        start_time=t_tl_start,
                        end_time=t_tl_end,
                    )

                    if not tl_ok:
                        cand["status"] = "FAILED"
                        cand["analysis_status"] = "VIDEO_ANALYSIS_FAILED"
                        cand["error"] = comp_analysis.get("analysis_error", "Twelve Labs analysis failed")
                        self._log(f"Twelve Labs: competitor {idx}/{len(selected_candidates)} [{cid}] failed: {cand['error']}")
                        return cand

                    duplicate_tracker[cid]["upload_count"] += 1
                    duplicate_tracker[cid]["analysis_count"] += 1
                    duplicate_tracker[cid]["ocr_count"] += 1

                    cand["status"] = "ANALYZED"
                    cand["analysis"] = comp_analysis
                    cand["hook"] = comp_analysis.get("hook")
                    cand["visual"] = comp_analysis.get("visual")
                    cand["audio"] = comp_analysis.get("audio")
                    cand["niche"] = comp_analysis.get("niche")
                    cand["topic"] = comp_analysis.get("topic")

                    # Attention model inference
                    t_att_start = time.perf_counter()
                    comp_att = []
                    if self.attention_service:
                        try:
                            audio_path = comp_asset.audio_wav_path if comp_asset else None
                            comp_att = self.attention_service.predict_attention(local_path, audio_wav_path=audio_path)
                        except Exception:
                            comp_att = []
                    t_att_end = time.perf_counter()

                    profiler.record_stage_direct(
                        stage_name=f"competitor_{idx}_{cid}_attention",
                        duration=round(t_att_end - t_att_start, 3),
                        category=TimingCategory.LOCAL_CPU,
                        parent="stage_g_competitor_download_and_analysis",
                        metadata={"canonical_id": cid},
                        start_time=t_att_start,
                        end_time=t_att_end,
                    )
                    cand["attention_predictions"] = comp_att

                    # Semantic relevance verification
                    t_rel_start = time.perf_counter()
                    self._log(f"Relevance: verifying semantic relation to uploaded Reel for [{cid}]")
                    relevance_res = score_video_relevance(
                        your_reel.get("analysis") or your_reel,
                        comp_analysis.get("analysis") or comp_analysis,
                    )
                    t_rel_end = time.perf_counter()
                    duplicate_tracker[cid]["verification_count"] += 1

                    profiler.record_stage_direct(
                        stage_name=f"competitor_{idx}_{cid}_relevance",
                        duration=round(t_rel_end - t_rel_start, 3),
                        category=TimingCategory.LOCAL_CPU,
                        parent="stage_g_competitor_download_and_analysis",
                        metadata={
                            "canonical_id": cid,
                            "eligible": relevance_res.get("video_relevance_eligible", False),
                            "score": relevance_res.get("video_relevance_score", 0.0),
                        },
                        start_time=t_rel_start,
                        end_time=t_rel_end,
                    )

                    cand["video_relevance_score"] = relevance_res.get("video_relevance_score")
                    cand["video_relevance_reason"] = relevance_res.get("video_relevance_reason")
                    cand["video_verification_status"] = relevance_res.get("relevance_status")

                    if relevance_res.get("video_relevance_eligible"):
                        cand["status"] = "VERIFIED"
                        self._log(f"Relevance: competitor {idx}/{len(selected_candidates)} [{cid}] VERIFIED relevant")
                    else:
                        cand["status"] = "REJECTED"
                        self._log(f"Relevance: competitor {idx}/{len(selected_candidates)} [{cid}] REJECTED (unrelated)")

                    # Write to Competitor Artifact Cache
                    self.competitor_cache.set(
                        cid,
                        {
                            "comp_analysis": comp_analysis,
                            "attention_predictions": comp_att,
                            "relevance_res": relevance_res,
                            "local_media_path": local_path,
                        },
                        profiler=profiler,
                    )
                    return cand

                # Bounded concurrency: strictly 2 worker threads
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    processed_items = list(executor.map(
                        _process_competitor_task,
                        list(enumerate(selected_candidates, 1))
                    ))

                analyzed_competitors = []
                verified_competitors = []
                rejected_competitors = []
                competitor_attentions = {}
                downloaded_count = sum(1 for c in processed_items if c.get("download_status") == "DOWNLOADED")

                for cand in processed_items:
                    cid = get_canonical_reel_id(cand)
                    if cand.get("status") in {"ANALYZED", "VERIFIED", "REJECTED"}:
                        analyzed_competitors.append(cand)
                    if cand.get("status") == "VERIFIED":
                        verified_competitors.append(cand)
                    elif cand.get("status") == "REJECTED":
                        rejected_competitors.append(cand)
                    if cand.get("attention_predictions"):
                        competitor_attentions[cid] = cand["attention_predictions"]

                # Step 11b: Final Competitor Selection enforcing Creator Diversity (Section 12 & 13)
                with profiler.profile_stage("stage_g_final_selection", parent="stage_g_competitor_download_and_analysis"):
                    final_verified, diversity_diag = select_final_competitors(
                        verified_competitors, max_competitors=10, max_per_creator=2
                    )
                user_edit_type = (
                    (your_reel.get("discovery_profile") or {}).get("edit_type")
                    or (your_reel.get("analysis") or {}).get("edit_type")
                    or your_reel.get("edit_type")
                    or "action montage"
                )
                competitor_clusters = cluster_competitors(final_verified, user_edit_type=user_edit_type)
                self._log(
                    f"Competitor Selection: {len(final_verified)} verified Reels selected across "
                    f"{diversity_diag['unique_creators']} unique creators (diversity cap: max 2/creator); "
                    f"Primary cluster '{competitor_clusters['primary_cluster_name']}': {competitor_clusters['primary_count']} Reels"
                )

                # Diagnostic duplicate work verification log
                self._log(f"Competitor Duplicate Audit: {len(duplicate_tracker)} unique competitors tracked")
                for cid_k, counts in duplicate_tracker.items():
                    if any(v > 1 for v in counts.values()):
                        self._log(f"[WARN] Duplicate work detected for [{cid_k}]: {counts}")

            # Update diagnostics and profiler metrics
            profiler.record_metric("candidate_count", len(raw_candidates))
            profiler.record_metric("verified_competitor_count", len(final_verified))
            profiler.record_metric("download_count", downloaded_count)
            profiler.record_metric("twelvelabs_upload_count", len(analyzed_competitors) + 1)
            profiler.record_metric("twelvelabs_analysis_count", len(analyzed_competitors) + 1)
            profiler.record_metric("ocr_processing_count", len(analyzed_competitors) + 1)

            discovery_diag.update({
                "raw_candidates": discovery_diag.get("raw_candidates", len(raw_candidates)),
                "normalized_candidates": len(raw_candidates),
                "ranked_candidates": len(ranked_candidates),
                "selected_competitors": len(selected_candidates),
                "downloaded_competitors": downloaded_count,
                "twelvelabs_analyzed_competitors": len(analyzed_competitors),
                "verified_relevant_competitors": len(final_verified),
                "raw_verified_pool": len(verified_competitors),
                "rejected_competitors": len(rejected_competitors),
                "unique_creators": diversity_diag["unique_creators"],
                "competitor_clusters": competitor_clusters,
                "duplicate_work_audit": duplicate_tracker,
            })

        # Step 12: Attention Curve Benchmark Comparison & Reach-Attention Association
        with profiler.profile_stage("stage_h_attention_benchmark"):
            if len(final_verified) > 0:
                self._log(f"Attention Benchmark: comparing uploaded Reel against {len(final_verified)} verified competitors")
                verified_attentions = {
                    c.get("shortcode") or c.get("id"): c.get("attention_predictions", [])
                    for c in final_verified
                    if c.get("attention_predictions")
                }
                attention_comparison = AttentionComparisonService.compare(
                    attention_predictions, verified_attentions, verified_competitors=final_verified
                )
            else:
                self._log("Attention Benchmark: zero verified competitors; skipping competitor benchmark comparison")
                attention_comparison = {
                    "status": "SKIPPED_NO_COMPETITORS",
                    "message": "Competitor attention benchmark skipped: zero competitors discovered.",
                    "benchmark": {"competitor_count": 0},
                    "insights": [],
                    "reach_attention_association": None,
                }

        # Step 13-14: Pattern Discovery & Strategy Synthesis
        with profiler.profile_stage("stage_i_strategy_and_patterns"):
            if len(final_verified) > 0:
                self._log(f"Pattern Agent: extracting empirical recurring patterns from {len(final_verified)} verified competitors")
                pattern_agent_res = PatternAgent.analyze(final_verified, your_reel)
            else:
                self._log("Pattern Agent: zero verified competitors; skipping competitor pattern discovery")
                pattern_agent_res = {
                    "agent": "pattern",
                    "status": "SKIPPED_NO_COMPETITORS",
                    "verified_competitor_count": 0,
                    "findings": ["[AGGREGATED_PATTERN] No verified competitors available for empirical recurring pattern discovery."],
                    "patterns": [],
                    "pattern_counts": {},
                }
            agent_results["pattern"] = pattern_agent_res
            agent_results["competitor_intelligence"] = pattern_agent_res.get("competitor_intelligence") or {}
            your_reel["competitor_intelligence"] = agent_results["competitor_intelligence"]

            # Legacy ML cluster discovery
            patterns = self.ml.discover(final_verified) if len(final_verified) > 0 else {}

            # Step 14: Strategy Agent Synthesis
            self._log("Strategy Agent: generating evidence-backed strategic action items")
            strategy_res = StrategyAgent.synthesize(
                your_reel, agent_results, attention_comparison, pattern_agent_res
            )
            agent_results["strategy"] = strategy_res
            agent_results["recommendation_intelligence"] = strategy_res.get("recommendation_intelligence") or {}
            your_reel["recommendation_intelligence"] = agent_results["recommendation_intelligence"]

            # Disagreement resolution
            conflict_res = self.resolve_agent_conflicts(agent_results)

            # Comparison against verified competitors
            comparison = compare(your_reel, final_verified)
            recommendations = analyze_trends(final_verified, your_reel)
            virality_analysis = ViralityAgent.evaluate(your_reel, final_verified, comparison)

        # Step 15: Save Competitor Dataset & Generate Audit Report
        with profiler.profile_stage("stage_j_report_generation"):
            save_competitor_dataset(analyzed_competitors)

            # Ensure shadow spatial result is present (if not already collected)
            if shadow_future is not None and dhf1k_spatial_signal is None:
                try:
                    dhf1k_spatial_signal = shadow_future.result(timeout=15.0)
                except Exception as e:
                    dhf1k_spatial_signal = {
                        "status": "FAILED",
                        "error": f"Shadow spatial worker error: {e}",
                        "samples": [],
                        "telemetry": {}
                    }
                if shadow_executor:
                    shadow_executor.shutdown(wait=False)

            main_pipeline_end = time.perf_counter()

            # Store separated signals: NEMAR production attention vs. DHF1K shadow spatial signal
            your_reel["nemar_attention_signal"] = attention_predictions
            your_reel["dhf1k_spatial_saliency_signal"] = dhf1k_spatial_signal
            your_reel["attention_intelligence"] = attention_intelligence

            # Render non-obscuring visualization if intelligence is available
            vis_path = None
            if attention_intelligence and attention_intelligence.get("status") in {"SUCCESS", "PARTIAL"}:
                vis_out = settings.REPORTS_DIR / f"virallens_attention_{video_path.stem}.png"
                vis_path = render_attention_visualization(video_path, attention_intelligence, vis_out)

            full_payload = {
                "your_reel": your_reel,
                "nemar_attention_signal": attention_predictions,
                "dhf1k_spatial_saliency_signal": dhf1k_spatial_signal,
                "attention_intelligence": attention_intelligence,
                "attention_visualization_path": vis_path,
                "main_pipeline_telemetry": {
                    "main_pipeline_start": round(main_pipeline_start, 4),
                    "main_pipeline_end": round(main_pipeline_end, 4),
                    "main_pipeline_wall_time": round(main_pipeline_end - main_pipeline_start, 4),
                },
                "discovery_profile": disc_profile,
                "queries": queries,
                "discovery_diagnostics": discovery_diag,
                "selected_candidates": selected_candidates,
                "analyzed_competitors": analyzed_competitors,
                "verified_competitors": final_verified,
                "competitor_clusters": competitor_clusters,
                "raw_verified_pool": verified_competitors,
                "rejected_competitors": rejected_competitors,
                "patterns": patterns,
                "comparison": comparison,
                "virality_analysis": virality_analysis,
                "recommendations": recommendations,
                "agent_results": agent_results,
                "attention_comparison": attention_comparison,
                "conflict_resolution": conflict_res,
            }

            json_path, md_path = save_report(full_payload, base_name=video_path.stem)
            self._log(f"Report: saved audit report to {json_path.name}")
            self._log("Pipeline: complete")

            full_payload["report_json_path"] = str(json_path)
            full_payload["report_md_path"] = str(md_path)

        perf_report = profiler.save_report()
        full_payload["performance_report"] = perf_report
        self._log(f"Performance: report saved to {perf_report.get('report_path', '')}")
        print("\n" + profiler.summary_log())

        return full_payload
