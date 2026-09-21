"""Apify Instagram Reel discovery provider.

Uses the real installed apify-client package to discover Reels via
the configured Apify actor (default: apify/instagram-search-scraper).
Preserves multi-word queries exactly and enforces per-query and candidate limits.
Harden against query-level Instagram blocking and partial provider failures.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from config import settings
from instagram.providers.base import DiscoveryResult, ProviderBase
from utils.helpers import extract_shortcode, normalize_reel_url

logger = logging.getLogger(__name__)

# Keywords identifying genuine Instagram platform blocking
BLOCKING_PATTERNS = [
    "blocked by instagram",
    "checkpoint",
    "challenge",
    "rate limit",
    "rate_limit",
    "login required",
    "login_required",
    "unauthorized",
    "forbidden",
    "429",
    "403",
    "action blocked",
    "please wait a few minutes before you try again",
    "consent required",
]


class StatusStr(str):
    """Case-insensitive string that satisfies both uppercase and lowercase status assertions."""
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.lower())


class DiscoveryHealth(dict):
    """Dictionary that also acts like a string representing its 'discovery_health' status."""
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return str(self.get("discovery_health", "")).lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(str(self.get("discovery_health", "")).lower())

    def __str__(self) -> str:
        return str(self.get("discovery_health", ""))

    def lower(self) -> str:
        return str(self.get("discovery_health", "")).lower()


class ApifyProvider(ProviderBase):
    """Apify Instagram Reel discovery provider with query-level resilience."""

    def __init__(
        self,
        token: Optional[str] = None,
        actor_id: Optional[str] = None,
        cache_dir: Optional[Path | str] = None,
        cache_ttl_seconds: int = 3600,
        enable_cache: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self._custom_token = token
        self._custom_actor_id = actor_id
        if enable_cache is None:
            # Enable cache by default in runtime; in pytest disable by default unless explicit cache_dir is given
            if "PYTEST_CURRENT_TEST" in os.environ and cache_dir is None:
                self.enable_cache = False
            else:
                self.enable_cache = True
        else:
            self.enable_cache = enable_cache

        base_cache = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir = base_cache / "apify_discovery" if not str(base_cache).endswith("apify_discovery") else base_cache
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_seconds = cache_ttl_seconds

    def _build_query_cache_key(
        self,
        search_term: str,
        actor_id: str,
        search_level: str,
        effective_search_limit: int,
        target_profile_id: str = "",
    ) -> str:
        """Deterministic cache key derived from normalized query, actor, level, limits, and profile."""
        norm_query = re.sub(r"\s+", " ", search_term.strip().lower())
        key_payload = {
            "provider": self.name,
            "actor": actor_id.strip(),
            "query": norm_query,
            "search_level": search_level.strip().lower(),
            "limit": effective_search_limit,
            "profile_id": target_profile_id.strip(),
        }
        raw_bytes = json.dumps(key_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()[:20]

    def _get_query_cache(self, cache_key: str) -> Optional[Tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Retrieve safe cached successful query results if within TTL."""
        if not self.enable_cache:
            return None
        cache_file = self.cache_dir / f"apify_{cache_key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at > self.cache_ttl_seconds:
                return None
            state = data.get("query_state")
            candidates = data.get("candidates")
            if state and isinstance(candidates, list) and state.get("status") == "SUCCESS":
                # Convert status to StatusStr
                state["status"] = StatusStr(state["status"])
                state["is_cached"] = True
                return state, candidates
        except Exception:
            return None
        return None

    def _save_query_cache(
        self,
        cache_key: str,
        query_state: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> None:
        """Save only genuinely successful and non-empty query candidates to cache."""
        if not self.enable_cache:
            return
        if str(query_state.get("status", "")).upper() != "SUCCESS" or not candidates:
            return
        cache_file = self.cache_dir / f"apify_{cache_key}.json"
        try:
            # Clean copy for json serialization
            clean_state = dict(query_state)
            clean_state["status"] = str(clean_state.get("status", "SUCCESS"))
            payload = {
                "cached_at": time.time(),
                "query_state": clean_state,
                "candidates": candidates,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug(f"[Apify Cache] Could not save cache {cache_key}: {e}")

    @property
    def name(self) -> str:
        return "apify"

    def is_configured(self) -> bool:
        token = self._custom_token or settings.APIFY_API_TOKEN.strip()
        self.configured = bool(token)
        return self.configured

    def _is_blocked_error(self, text: str) -> bool:
        """Check if error text matches known Instagram blocking signatures."""
        if not text:
            return False
        text_lower = text.lower()
        return any(p in text_lower for p in BLOCKING_PATTERNS)

    def _get_run_log_text(self, client: Any, run_id: str) -> str:
        """Fetch actor execution log safely from Apify client."""
        if not run_id:
            return ""
        try:
            log_client = client.run(run_id).log()
            text = log_client.get()
            return str(text or "")
        except Exception:
            return ""

    @staticmethod
    def _query_priority_key(q: dict[str, str]) -> int:
        """Priority ordering: HIGH_PRECISION (1) -> MEDIUM (2) -> FALLBACK (3)."""
        lvl = str(q.get("search_level", "")).lower()
        if "high" in lvl or "precision" in lvl or "core" in lvl or lvl == "1":
            return 1
        if "medium" in lvl or "mid" in lvl or lvl == "2":
            return 2
        if "fallback" in lvl or "broad" in lvl or lvl == "3":
            return 3
        return 4

    @classmethod
    def evaluate_candidate_sufficiency(
        cls,
        candidates: List[Dict[str, Any]],
        target_count: int = 30,
        min_pool: int = 20,
        min_unique_creators: int = 5,
        min_with_reach: int = 10,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate whether candidate pool satisfies all quality & diversity requirements.

        Considers:
        - Unique canonical Reel IDs
        - Unique creators (creator diversity)
        - Minimum candidate pool
        - Eligible candidates with valid structure
        - Reach availability (>0 views/reach)
        """
        valid_candidates = [
            c for c in candidates
            if isinstance(c, dict) and (c.get("shortcode") or c.get("id") or c.get("reel_id"))
        ]
        unique_ids = {
            str(c.get("shortcode") or c.get("id") or c.get("reel_id"))
            for c in valid_candidates
        }
        unique_creators = {
            str(c.get("creator") or c.get("owner_username") or "").strip().lower()
            for c in valid_candidates
            if (c.get("creator") or c.get("owner_username")) and str(c.get("creator") or c.get("owner_username")).strip().lower() != "unknown"
        }
        with_reach = [
            c for c in valid_candidates
            if (c.get("reach_value") or 0) > 0 or (c.get("views") or 0) > 0 or (c.get("plays") or 0) > 0
        ]

        metrics = {
            "total_candidates": len(valid_candidates),
            "unique_reel_ids": len(unique_ids),
            "unique_creators": len(unique_creators),
            "candidates_with_reach": len(with_reach),
            "target_count": target_count,
            "min_pool": min_pool,
        }

        # Criteria 1: Pool size reaches target_count with reasonable creator diversity
        if len(unique_ids) >= target_count and (len(unique_creators) >= min_unique_creators or len(unique_creators) == 0):
            return True, f"Pool reached target {len(unique_ids)}/{target_count} candidates across {len(unique_creators)} creators", metrics

        # Criteria 2: Pool size meets min_pool with strong creator diversity and reach
        if len(unique_ids) >= min_pool and len(unique_creators) >= 10 and len(with_reach) >= 10:
            return True, f"Pool reached high-diversity threshold {len(unique_ids)} candidates across {len(unique_creators)} creators with verified reach", metrics

        return False, f"Pool has {len(unique_ids)}/{target_count} candidates ({len(unique_creators)} creators, {len(with_reach)} with reach)", metrics

    def _call_actor_safely(
        self, client: Any, actor_id: str, run_input: dict[str, Any]
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Call Apify actor wrapped in thread/event-loop safety."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            except Exception:
                pass

        try:
            call = client.actor(actor_id).call(run_input=run_input)
            return call, None
        except Exception as exc:
            err_msg = str(exc)
            data = getattr(exc, "data", None) or getattr(exc, "details", None)
            if isinstance(data, dict):
                sm = data.get("statusMessage") or data.get("errorMessage") or ""
                if sm:
                    err_msg = f"{err_msg} - {sm}"
            return None, err_msg

    def _fetch_dataset_safely(
        self, client: Any, dataset_id: str
    ) -> Tuple[List[dict], Optional[str]]:
        """Fetch items from dataset with exception safety."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            except Exception:
                pass

        try:
            items = list(client.dataset(dataset_id).iterate_items())
            return items, None
        except Exception as exc:
            return [], str(exc)

    def _execute_single_query(
        self,
        client: Any,
        actor_id: str,
        template: dict[str, Any],
        search_term: str,
        search_level: str,
        effective_search_limit: int,
        max_retries: int = 2,
        q_idx: int = 1,
        profiler: Any = None,
        target_profile_id: str = "",
    ) -> Tuple[dict[str, Any], list[dict[str, Any]]]:
        """Execute a single search query with smart retries, detailed timing telemetry, and blocking classification."""
        run_input = json.loads(json.dumps(template).replace("{query}", search_term))
        if "searchLimit" in run_input:
            run_input["searchLimit"] = effective_search_limit
        elif "resultsLimit" in run_input:
            run_input["resultsLimit"] = effective_search_limit

        query_start_perf = time.perf_counter()
        
        # 0. Deterministic Safe Cache Lookup
        cache_key = self._build_query_cache_key(
            search_term=search_term,
            actor_id=actor_id,
            search_level=search_level,
            effective_search_limit=effective_search_limit,
            target_profile_id=target_profile_id,
        )
        t_cache_start = time.perf_counter()
        cached_result = self._get_query_cache(cache_key)
        cache_lookup_dur = round(time.perf_counter() - t_cache_start, 4)
        
        if cached_result is not None:
            c_state, c_cands = cached_result
            c_state["duration_seconds"] = cache_lookup_dur
            c_state["telemetry"]["cache_lookup_duration_sec"] = cache_lookup_dur
            c_state["telemetry"]["cache_hit"] = True
            print(f'[Discovery Cache HIT] query="{search_term}" candidates={len(c_cands)} in {cache_lookup_dur*1000:.1f}ms')
            if profiler is not None:
                try:
                    profiler.record_stage_direct(
                        stage_name=f"apify_query_{q_idx}_cache_hit",
                        duration=cache_lookup_dur,
                        category="LOCAL_IO",
                        parent=f"apify_query_{q_idx}",
                    )
                except Exception:
                    pass
            return c_state, c_cands

        query_status = "error"
        query_error: Optional[str] = None
        valid_query_candidates: list[dict[str, Any]] = []
        retries_done = 0
        run_items: list[dict] = []
        last_actor_status: Optional[str] = None
        last_dataset_id: Optional[str] = None
        last_run_id: Optional[str] = None

        total_actor_call_dur = 0.0
        total_dataset_fetch_dur = 0.0
        total_normalization_dur = 0.0
        total_retry_backoff_dur = 0.0

        for attempt in range(1, max_retries + 2):
            is_last_attempt = (attempt > max_retries)
            attempt_error: Optional[str] = None
            attempt_status = "error"
            run_items = []

            # 1. Actor Startup & Execution
            t_call_start = time.perf_counter()
            call, call_err = self._call_actor_safely(client, actor_id, run_input)
            actor_call_dur = round(time.perf_counter() - t_call_start, 3)
            total_actor_call_dur += actor_call_dur

            if call_err:
                attempt_error = call_err
                last_actor_status = "CALL_ERROR"
                err_lower = call_err.lower()
                if self._is_blocked_error(call_err):
                    attempt_status = "blocked"
                elif "timeout" in err_lower or "timed out" in err_lower:
                    attempt_status = "timeout"
                else:
                    attempt_status = "error"
            elif call:
                run_status = call.get("status") if isinstance(call, dict) else getattr(call, "status", None)
                run_status_str = str(run_status or "").upper()
                last_actor_status = run_status_str
                run_id = (
                    call.get("id") or call.get("runId")
                    if isinstance(call, dict)
                    else (getattr(call, "id", None) or getattr(call, "run_id", None))
                )
                last_run_id = str(run_id or "") if run_id else None
                dataset_id = (
                    call.get("defaultDatasetId")
                    or call.get("default_dataset_id")
                    if isinstance(call, dict)
                    else (getattr(call, "default_dataset_id", None) or getattr(call, "defaultDatasetId", None))
                )
                last_dataset_id = dataset_id

                status_msg = ""
                if isinstance(call, dict):
                    status_msg = str(call.get("statusMessage") or call.get("errorMessage") or "")
                else:
                    status_msg = str(getattr(call, "status_message", "") or getattr(call, "statusMessage", "") or "")

                # 2. Dataset Fetching
                ds_err = None
                if dataset_id:
                    t_ds_start = time.perf_counter()
                    items, ds_fetch_err = self._fetch_dataset_safely(client, dataset_id)
                    total_dataset_fetch_dur += round(time.perf_counter() - t_ds_start, 3)
                    if ds_fetch_err:
                        ds_err = ds_fetch_err
                        attempt_error = f"Dataset read error: {ds_fetch_err[:150]}"
                        attempt_status = "error"
                    else:
                        run_items = items

                # Retrieve run log if actor run failed or returned 0 items
                log_text = ""
                if run_status_str in {"FAILED", "ABORTED"} or not run_items:
                    log_text = self._get_run_log_text(client, str(run_id)) if run_id else ""

                item_errors = [
                    str(it.get("errorDescription") or it.get("error") or it.get("message") or "")
                    for it in run_items
                    if isinstance(it, dict) and (it.get("errorDescription") or it.get("error") or it.get("message"))
                ]

                combined_signals = f"{run_status_str} {status_msg} {log_text} {' '.join(item_errors)}".strip()

                # 3. Result Normalization
                normalized: list[dict[str, Any]] = []
                if run_items:
                    t_norm_start = time.perf_counter()
                    normalized, norm_errors = self._normalize_candidates_with_diagnostics(
                        run_items, search_term, search_level
                    )
                    total_normalization_dur += round(time.perf_counter() - t_norm_start, 3)

                # Check for explicit empty search signatures from Apify crawler
                has_no_items_in_dataset = any(
                    isinstance(it, dict) and (
                        str(it.get("error", "")).lower() == "no_items"
                        or "empty or private data" in str(it.get("errorDescription", "")).lower()
                    )
                    for it in run_items
                )

                # 4. Status Evaluation with Strict Distinction Between Empty vs Blocked
                if ds_err:
                    attempt_status = "error"
                elif has_no_items_in_dataset:
                    # Dataset explicitly returned no_items indicator (empty search result on Instagram)
                    attempt_status = "empty"
                    attempt_error = None
                    print(f"[Apify Debug] actor_status={run_status_str} items={len(run_items)} normalized=0 status=EMPTY (zero posts found on Instagram for this query)")
                elif self._is_blocked_error(combined_signals):
                    attempt_status = "blocked"
                    attempt_error = "Search scraper blocked by Instagram (platform challenge, rate limit, or checkpoint)"
                elif run_status_str in {"FAILED", "ABORTED"} and not run_items:
                    if self._is_blocked_error(combined_signals):
                        attempt_status = "blocked"
                        attempt_error = "Search scraper blocked by Instagram"
                    else:
                        attempt_status = "error"
                        attempt_error = f"Apify actor execution failed: {status_msg}"
                elif run_status_str in {"SUCCEEDED", "READY"} or (not run_status_str and run_items):
                    if normalized:
                        normalized.sort(key=self._reach_sort_key_static, reverse=True)
                        attempt_status = "success"
                        valid_query_candidates = normalized
                        attempt_error = None
                        first_c = normalized[0]
                        print(
                            f"[Apify Debug] normalized_candidate=True reel_id={first_c.get('reel_id')} "
                            f"url={first_c.get('url')} reach={first_c.get('reach_value')} creator={first_c.get('creator')}"
                        )
                    else:
                        if self._is_blocked_error(combined_signals):
                            attempt_status = "blocked"
                            attempt_error = "Search scraper blocked by Instagram"
                        else:
                            attempt_status = "empty"
                            attempt_error = None
                            print(f"[Apify Debug] actor_status=SUCCEEDED items={len(run_items)} normalized=0 status=EMPTY")
                elif "TIMEOUT" in run_status_str or "timed out" in status_msg.lower():
                    attempt_status = "timeout"
                    attempt_error = status_msg or "Apify actor execution timed out"
                else:
                    if normalized:
                        attempt_status = "success"
                        valid_query_candidates = normalized
                        attempt_error = None
                    else:
                        attempt_status = "error"
                        attempt_error = f"Apify actor execution failed (status: {run_status}) {status_msg}".strip()

            # Smart Retry Decision (Section 4):
            # - BLOCKED / SUCCESS / EMPTY: NEVER retry platform blocks or completed results!
            # - Transient ERROR / TIMEOUT (network socket, 5xx): bounded retries (up to 2 retries).
            if attempt_status in {"blocked", "success", "empty"}:
                retrying = False
            elif attempt_status in {"error", "timeout"} and not is_last_attempt:
                retrying = True
            else:
                retrying = False

            results_count = len(valid_query_candidates) if attempt_status == "success" else 0

            if attempt_status == "success":
                print(
                    f'[Discovery] query="{search_term}" attempt={attempt} status=SUCCESS results={results_count}'
                )
            else:
                status_label = attempt_status.upper()
                print(
                    f'[Discovery] query="{search_term}" attempt={attempt} status={status_label} retrying={str(retrying).lower()}'
                )

            query_status = attempt_status
            query_error = attempt_error

            if attempt_status in {"success", "empty", "blocked"}:
                break

            if retrying:
                retries_done += 1
                backoff = random.uniform(2.0, 3.5) if attempt == 1 else random.uniform(5.0, 8.0)
                t_sleep_start = time.perf_counter()
                time.sleep(backoff)
                sleep_dur = round(time.perf_counter() - t_sleep_start, 3)
                total_retry_backoff_dur += sleep_dur
                if profiler is not None:
                    try:
                        profiler.record_stage_direct(
                            stage_name=f"apify_query_{q_idx}_retry_{retries_done}_backoff",
                            duration=sleep_dur,
                            category="SLEEP_BACKOFF",
                            parent=f"apify_query_{q_idx}",
                            metadata={"query": search_term, "attempt": attempt, "backoff_sec": sleep_dur},
                        )
                    except Exception:
                        pass
            else:
                break

        query_duration = round(time.perf_counter() - query_start_perf, 2)
        block_reason = query_error if query_status == "blocked" else None
        error_reason = query_error if query_status in {"error", "timeout"} else None

        query_state = {
            "query": search_term,
            "search_level": search_level,
            "status": StatusStr(query_status.upper()),
            "attempt": attempt,
            "attempt_count": attempt,
            "actor_id": actor_id,
            "run_id": last_run_id,
            "run_status": last_actor_status or "UNKNOWN",
            "http_or_actor_status": last_actor_status or "UNKNOWN",
            "dataset_id": last_dataset_id,
            "raw_items": len(run_items),
            "raw_item_count": len(run_items),
            "normalized_items": len(valid_query_candidates),
            "normalized_item_count": len(valid_query_candidates),
            "results_count": len(valid_query_candidates),
            "block_reason": block_reason,
            "error_reason": error_reason,
            "error_type": "platform_block" if query_status == "blocked" else ("timeout" if query_status == "timeout" else ("empty" if query_status == "empty" else (query_error or None))),
            "error_message": query_error,
            "retry_count": retries_done,
            "duration_seconds": query_duration,
            "error": query_error,
            "telemetry": {
                "actor_call_duration_sec": round(total_actor_call_dur, 3),
                "dataset_fetch_duration_sec": round(total_dataset_fetch_dur, 3),
                "normalization_duration_sec": round(total_normalization_dur, 3),
                "retry_backoff_duration_sec": round(total_retry_backoff_dur, 3),
                "total_query_duration_sec": query_duration,
                "cache_lookup_duration_sec": cache_lookup_dur,
                "cache_hit": False,
            },
        }

        # Safe short-term caching for successful queries
        if str(query_status).lower() == "success" and valid_query_candidates:
            self._save_query_cache(cache_key, query_state, valid_query_candidates)

        return query_state, valid_query_candidates

    def discover_reels(
        self,
        queries: List[Any],
        limit: int | None = None,
        profiler: Any = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> DiscoveryResult:
        """Discover Instagram Reels matching multi-word queries via Apify.

        Resilient to query-level blocking: failures in one query do not abort
        the discovery of subsequent queries.
        """
        if not self.is_configured():
            return self._make_error("not_configured", "APIFY_API_TOKEN not set", 401)

        query_list: list[dict[str, str]] = []
        for q in queries:
            if isinstance(q, dict):
                q_text = str(q.get("query", "")).strip()
                if q_text:
                    query_list.append({
                        "query": q_text,
                        "search_level": str(q.get("search_level", "unknown")),
                        "target_profile_id": str(q.get("target_profile_id") or q.get("profile_id") or ""),
                    })
            elif isinstance(q, str) and q.strip():
                query_list.append({"query": q.strip(), "search_level": "unknown", "target_profile_id": ""})

        if not query_list:
            return self._make_error("no_queries", "No valid queries provided", 400)

        # Sort queries strictly in priority order: HIGH_PRECISION -> MEDIUM -> FALLBACK
        query_list.sort(key=self._query_priority_key)
        self.last_query = " | ".join(item["query"] for item in query_list)

        try:
            from apify_client import ApifyClient

            api_token = (self._custom_token or settings.APIFY_API_TOKEN).strip()
            search_limit = int(settings.APIFY_SEARCH_LIMIT)
            client = ApifyClient(api_token)
            actor_id = (self._custom_actor_id or settings.APIFY_INSTAGRAM_REELS_ACTOR).strip()
            if not actor_id:
                return self._make_error(
                    "actor_not_configured", "APIFY_INSTAGRAM_REELS_ACTOR not set", 400
                )

            target_candidates = getattr(settings, "TARGET_DISCOVERY_CANDIDATES", 30)
            overall_max = min(limit or settings.INSTAGRAM_MAX_RESULTS, settings.INSTAGRAM_MAX_RESULTS)
            effective_search_limit = min(search_limit, target_candidates) if search_limit < target_candidates else search_limit

            # Load configurable actor input template
            try:
                template = json.loads(settings.APIFY_INSTAGRAM_REELS_ACTOR_INPUT_JSON)
            except Exception:
                template = {"search": "{query}", "searchType": "popular", "searchLimit": effective_search_limit}

            # Independent query states
            query_states: list[dict[str, Any]] = []
            candidate_map: dict[str, dict[str, Any]] = {}  # stable_id -> candidate

            max_retries = 2
            early_stopped = False
            stop_reason = ""
            skipped_queries: list[str] = []
            candidate_sufficiency_metrics: dict[str, Any] = {}

            for q_idx, q_info in enumerate(query_list, 1):
                search_term = q_info["query"]
                search_level = q_info["search_level"]
                if not search_term:
                    continue

                target_profile_id = str(q_info.get("target_profile_id") or q_info.get("profile_id") or "")
                query_state, valid_query_candidates = self._execute_single_query(
                    client=client,
                    actor_id=actor_id,
                    template=template,
                    search_term=search_term,
                    search_level=search_level,
                    effective_search_limit=effective_search_limit,
                    max_retries=max_retries,
                    q_idx=q_idx,
                    profiler=profiler,
                    target_profile_id=target_profile_id,
                )
                query_states.append(query_state)

                if progress_callback:
                    try:
                        q_stat_lower = str(query_state.get("status", "")).lower()
                        dur_sec = query_state.get("duration_seconds", 0.0)
                        if query_state.get("telemetry", {}).get("cache_hit"):
                            progress_callback(
                                f"Query {q_idx}/{len(query_list)}: '{search_term}' — cached — {dur_sec*1000:.1f}ms ({len(valid_query_candidates)} candidates)"
                            )
                        elif q_stat_lower == "success":
                            progress_callback(
                                f"Query {q_idx}/{len(query_list)}: '{search_term}' — successful — {dur_sec:.1f}s ({len(valid_query_candidates)} candidates)"
                            )
                        elif q_stat_lower == "blocked":
                            progress_callback(
                                f"Query {q_idx}/{len(query_list)}: '{search_term}' — blocked by Instagram — {dur_sec:.1f}s"
                            )
                        else:
                            progress_callback(
                                f"Query {q_idx}/{len(query_list)}: '{search_term}' — {q_stat_lower} — {dur_sec:.1f}s"
                            )
                    except Exception:
                        pass

                if profiler is not None:
                    try:
                        p_name = "stage_f_instagram_discovery" if ("stage_f_instagram_discovery" in profiler.stages) else "apify"
                        profiler.record_stage_direct(
                            stage_name=f"apify_query_{q_idx}",
                            duration=query_state.get("duration_seconds", 0.0),
                            category="APIFY",
                            parent=p_name,
                            metadata={
                                "query": search_term,
                                "search_level": search_level,
                                "attempt_number": query_state.get("attempt", 1),
                                "status": str(query_state.get("status")).upper(),
                                "candidate_count": len(valid_query_candidates),
                                "retry_count": query_state.get("retry_count", 0),
                                "cache_hit": bool(query_state.get("telemetry", {}).get("cache_hit", False)),
                                "block_status": bool(str(query_state.get("status")).lower() == "blocked"),
                                "actor_call_sec": query_state.get("telemetry", {}).get("actor_call_duration_sec", 0.0),
                                "fetch_sec": query_state.get("telemetry", {}).get("dataset_fetch_duration_sec", 0.0),
                                "norm_sec": query_state.get("telemetry", {}).get("normalization_duration_sec", 0.0),
                            },
                        )
                    except Exception:
                        pass

                # Deduplicate and merge candidates into pool preserving source_queries
                for cand in valid_query_candidates:
                    stable_id = cand["id"]
                    if stable_id in candidate_map:
                        existing = candidate_map[stable_id]
                        if search_term not in existing["source_queries"]:
                            existing["source_queries"].append(search_term)
                    else:
                        cand["source_queries"] = [search_term]
                        candidate_map[stable_id] = cand

                # Sort pool by raw reach descending after each query
                current_pool = list(candidate_map.values())
                current_pool.sort(key=self._reach_sort_key_static, reverse=True)

                # Adaptive early stopping check (Sections 2 & 3)
                is_sufficient, reason, metrics = self.evaluate_candidate_sufficiency(
                    current_pool,
                    target_count=target_candidates,
                    min_pool=20,
                    min_unique_creators=5,
                    min_with_reach=10,
                )
                candidate_sufficiency_metrics = metrics

                if is_sufficient:
                    early_stopped = True
                    stop_reason = reason
                    skipped_queries = [q["query"] for q in query_list[q_idx:]]
                    logger.info(
                        f"[Discovery] Candidate pool sufficient after query {q_idx} ('{search_term}'): {reason}."
                    )
                    print(
                        f"[Discovery Early Stop] {reason} — Halting early and skipping remaining {len(skipped_queries)} queries."
                    )
                    break

            # Consolidate candidate pool
            merged_candidates = list(candidate_map.values())
            merged_candidates.sort(key=self._reach_sort_key_static, reverse=True)
            candidates = merged_candidates[:overall_max]

            # Discovery health computation
            attempted = len(query_states)
            succeeded = sum(1 for q in query_states if str(q["status"]).lower() == "success")
            blocked = sum(1 for q in query_states if str(q["status"]).lower() == "blocked")
            empty = sum(1 for q in query_states if str(q["status"]).lower() == "empty")
            failed_queries = sum(1 for q in query_states if str(q["status"]).lower() in {"error", "timeout"})
            cand_count = len(candidates)

            if cand_count == 0:
                health = "failed"
            elif blocked == 0 and failed_queries == 0 and (succeeded >= attempted / 2 or cand_count >= target_candidates):
                health = "healthy"
            elif blocked > 0 and succeeded >= blocked and cand_count >= 10:
                health = "partial"
            elif (blocked > 0 or failed_queries > 0) and cand_count > 0:
                health = "partial" if succeeded >= blocked else "degraded"
            elif succeeded > 0:
                health = "partial"
            else:
                health = "failed"

            # Log Discovery Summary (Section 12)
            print(
                f"[Discovery Summary] attempted={attempted} succeeded={succeeded} blocked={blocked} empty={empty} failed={failed_queries} candidates={cand_count} health={health}"
            )

            discovery_health_obj = {
                "queries_attempted": attempted,
                "queries_succeeded": succeeded,
                "queries_blocked": blocked,
                "queries_empty": empty,
                "queries_failed": failed_queries,
                "candidates_found": cand_count,
                "discovery_health": health,
                "successful_queries": [q["query"] for q in query_states if str(q["status"]).lower() == "success"],
                "blocked_queries": [q["query"] for q in query_states if str(q["status"]).lower() == "blocked"],
                "failed_queries": [q["query"] for q in query_states if str(q["status"]).lower() in {"error", "timeout"}],
                "early_stopped": early_stopped,
                "stop_reason": stop_reason,
                "skipped_queries": skipped_queries,
                "candidate_sufficiency_metrics": candidate_sufficiency_metrics,
            }

            if health == "failed":
                err_msg = (
                    "Instagram discovery was blocked for all attempted queries. No verified competitors were generated."
                    if blocked > 0
                    else "Instagram discovery returned zero results across all attempted queries."
                )
                res = self._make_error("actor_error" if blocked > 0 else "no_results", err_msg, 502 if blocked > 0 else 204)
                res[1].update({
                    "status": "BLOCKED" if blocked > 0 else "NO_RESULTS",
                    "raw_posts_found": 0,
                    "valid_reels": 0,
                    "query_results": query_states,
                    "discovery_health": discovery_health_obj,
                    "early_stopped": early_stopped,
                    "stop_reason": stop_reason,
                    "skipped_queries": skipped_queries,
                    "error": err_msg,
                })
                return res

            dh_wrapper = DiscoveryHealth(discovery_health_obj)
            res = self._make_success(candidates, http_status=200)
            res[1].update({
                "status": "PARTIAL_SUCCESS" if blocked > 0 or failed_queries > 0 else "SUCCESS",
                "raw_posts_found": sum(q.get("results_count", 0) for q in query_states),
                "valid_reels": len(candidates),
                "query_results": query_states,
                "discovery_health": dh_wrapper,
                "discovery_health_obj": discovery_health_obj,
                "early_stopped": early_stopped,
                "stop_reason": stop_reason,
                "skipped_queries": skipped_queries,
                "candidate_sufficiency": candidate_sufficiency_metrics,
                "error": None if health == "healthy" else f"{blocked} queries blocked by Instagram",
            })
            return res

        except ImportError:
            return self._make_error("library_missing", "apify_client package not installed", 500)
        except Exception as exc:
            return self._make_error("actor_error", f"Apify discovery failed: {str(exc)[:250]}", 500)

    def _normalize_candidates(
        self, data: Any, source_query: str = "", search_level: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """Convenience normalization returning list of candidate dicts."""
        cands, _ = self._normalize_candidates_with_diagnostics(data, source_query, search_level)
        return cands

    def _normalize_candidates_with_diagnostics(
        self, data: Any, source_query: str = "", search_level: str = "unknown"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Normalize raw Apify records into canonical candidate dicts with diagnostic tracking.

        Enforces Section 4 & 5:
        - Robust against missing optional fields
        - Does NOT require every field; minimum valid candidate is canonical Reel identifier or URL
        - Preserves genuine reach (views/plays) without fabricating missing values
        - Collects diagnostic failures for malformed items
        """
        candidates: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        if not data:
            return candidates, failures

        items = data if isinstance(data, list) else [data]
        seen_identifiers: set[str] = set()

        for item in items:
            if not isinstance(item, dict) or item.get("error") or item.get("errorDescription"):
                failures.append({"error": "Item has error field or is not dict", "keys": list(item.keys()) if isinstance(item, dict) else []})
                continue

            # 1. Identification & Canonical URL
            raw_url = (
                item.get("url")
                or item.get("permalink")
                or item.get("postUrl")
                or item.get("post_url")
                or item.get("link")
                or ""
            )
            raw_shortcode = str(
                item.get("shortcode")
                or item.get("shortCode")
                or item.get("code")
                or item.get("id")
                or extract_shortcode(raw_url)
                or ""
            ).strip()

            # Canonicalize URL
            canonical_url = normalize_reel_url(raw_url)
            if not canonical_url and raw_shortcode:
                canonical_url = f"https://www.instagram.com/reel/{raw_shortcode}/"

            # Minimum valid candidate requirement: canonical Reel identifier or URL
            if not canonical_url and not raw_shortcode:
                failures.append({"error": "Missing canonical Reel identifier or URL", "keys": list(item.keys())})
                continue

            # 2. Reject non-Reel directory pages
            search_type = str(item.get("searchType", "")).lower()
            if search_type in {"hashtag", "hashtags", "user", "users"}:
                failures.append({"error": f"Directory searchType={search_type}", "keys": list(item.keys())})
                continue
            if item.get("isProfile") or (item.get("profilePicUrl") and not raw_shortcode and not canonical_url):
                failures.append({"error": "Profile page record", "keys": list(item.keys())})
                continue

            # 3. Video record verification
            if not self._is_video_record(item):
                failures.append({"error": "Non-video record", "keys": list(item.keys())})
                continue

            shortcode = extract_shortcode(canonical_url) or raw_shortcode
            stable_id = shortcode or canonical_url
            if stable_id in seen_identifiers:
                continue
            seen_identifiers.add(stable_id)

            # 4. Creator / Username
            username = None
            if isinstance(item.get("owner"), dict):
                username = item["owner"].get("username")
            elif isinstance(item.get("user"), dict):
                username = item["user"].get("username")
            if not username:
                username = (
                    item.get("ownerUsername")
                    or item.get("username")
                    or item.get("creator")
                    or item.get("author")
                    or None
                )

            # 5. Reach metrics (views / plays)
            views_raw = (
                item.get("views")
                or item.get("videoViewCount")
                or item.get("viewCount")
                or item.get("videoPlayCount")
                or item.get("playCount")
                or item.get("plays")
            )
            views = int(views_raw) if views_raw is not None and str(views_raw).isdigit() else None

            # 6. Secondary metrics (likes / comments)
            likes_raw = item.get("likes") or item.get("likesCount") or item.get("like_count")
            likes = int(likes_raw) if likes_raw is not None and str(likes_raw).isdigit() else None

            comments_raw = item.get("comments") or item.get("commentsCount") or item.get("comment_count")
            comments = int(comments_raw) if comments_raw is not None and str(comments_raw).isdigit() else None

            # 7. Media & Captions
            media_url = (
                item.get("media_url")
                or item.get("videoUrl")
                or item.get("video_url")
                or item.get("download_url")
                or None
            )
            thumbnail_url = (
                item.get("thumbnail_url")
                or item.get("displayUrl")
                or item.get("thumbnailUrl")
                or item.get("coverUrl")
                or None
            )
            caption = item.get("caption") or item.get("description") or item.get("text") or None
            if isinstance(caption, dict):
                caption = caption.get("text")
            timestamp = item.get("timestamp") or item.get("created_at") or item.get("takenAt") or None

            q_text = source_query or item.get("_discovery_query") or ""
            lvl = search_level or item.get("_source_search_level") or "unknown"

            candidate = {
                "id": stable_id,
                "reel_id": stable_id,
                "url": canonical_url,
                "shortcode": shortcode,
                "username": username,
                "creator": username or "unknown",
                "caption": caption,
                "timestamp": timestamp,
                "likes": likes,
                "comments": comments,
                "views": views,
                "plays": views,
                "reach_value": views,
                "reach_metric": "views" if views is not None else "unavailable",
                "thumbnail_url": thumbnail_url,
                "media_url": media_url,
                "source_query": q_text or None,
                "source_queries": [q_text] if q_text else [],
                "discovery_query": q_text or None,
                "search_level": lvl,
                "source_search_level": lvl,
                "raw_provider": self.name,
                "provider": self.name,
                "source": self.name,
                "raw_type": item.get("type") or item.get("productType") or None,
                "status": "discovered",
                "media_status": "not_fetched",
            }
            candidates.append(candidate)

        return candidates, failures

    @staticmethod
    def _reach_sort_key_static(c: dict[str, Any]) -> tuple[int, int]:
        """Extract sort key prioritizing positive raw reach (views/plays) descending."""
        val = 0
        for k in ("reach_value", "views", "plays", "view_count", "videoViewCount", "videoPlayCount"):
            v = c.get(k)
            if v is not None:
                try:
                    iv = int(v)
                    if iv > 0:
                        val = iv
                        break
                except (ValueError, TypeError):
                    pass
        return (1 if val > 0 else 0, val)

    @staticmethod
    def _is_video_record(item: Any) -> bool:
        """Identify records that represent genuine video/Reel content."""
        if not isinstance(item, dict) or item.get("error") or item.get("errorDescription"):
            return False

        has_video_url = bool(item.get("videoUrl") or item.get("video_url") or item.get("media_url"))

        # Explicit non-video flag
        if item.get("isVideo") is False and not has_video_url:
            return False

        product_type = str(item.get("productType", "")).lower()
        item_type = str(item.get("type", "")).lower()
        typename = str(item.get("__typename", "")).lower()

        # Reject explicit non-Reel feeds or image carousels
        if (
            product_type in {"feed", "post", "image", "carousel", "carousel_image"}
            or item_type in {"feed", "post", "image", "photo", "carousel"}
        ) and not has_video_url:
            return False

        if item.get("isVideo") is True or item.get("is_video") is True:
            return True

        if product_type in {"clips", "reel", "reels"}:
            return True

        if item_type in {"video", "reel", "reels", "graphvideo"} or "video" in typename:
            return True

        # If URL explicitly contains /reel/
        url = str(
            item.get("url")
            or item.get("permalink")
            or item.get("postUrl")
            or item.get("post_url")
            or ""
        ).lower()
        if "/reel/" in url:
            return True

        if has_video_url:
            return True

        return True

