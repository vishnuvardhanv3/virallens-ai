"""Centralized application configuration.

Single source of truth for all environment variables, limits, and defaults.
Never expose API keys in logs, exceptions, or error messages.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Twelve Labs Multimodal AI API
TWELVELABS_API_KEY: str = os.getenv("TWELVELABS_API_KEY", "").strip()
TWELVELABS_INDEX_ID: str = os.getenv("TWELVELABS_INDEX_ID", "").strip()
TWELVELABS_ENGINE: str = os.getenv("TWELVELABS_ENGINE", "pegasus1.5").strip()

# Apify Instagram Discovery API
APIFY_API_TOKEN: str = os.getenv("APIFY_API_TOKEN", "").strip()
APIFY_INSTAGRAM_REELS_ACTOR: str = os.getenv(
    "APIFY_INSTAGRAM_REELS_ACTOR", "apify/instagram-search-scraper"
).strip()
APIFY_INSTAGRAM_REELS_ACTOR_INPUT_JSON: str = os.getenv(
    "APIFY_INSTAGRAM_REELS_ACTOR_INPUT_JSON",
    '{"search":"{query}","searchType":"popular","searchLimit":25}',
).strip()

# Discovery configuration
DISCOVERY_MODE: str = os.getenv("DISCOVERY_MODE", "provider").strip()
DISCOVERY_PROVIDER: str = os.getenv("DISCOVERY_PROVIDER", "apify").strip().lower()
INSTAGRAM_PROVIDER_ORDER: str = os.getenv("INSTAGRAM_PROVIDER_ORDER", "apify").strip()

# Discovery & Processing Limits (independent controls)
INSTAGRAM_MAX_RESULTS: int = int(os.getenv("INSTAGRAM_MAX_RESULTS", "100"))
APIFY_SEARCH_LIMIT: int = int(os.getenv("APIFY_SEARCH_LIMIT", "25"))
TARGET_DISCOVERY_CANDIDATES: int = int(os.getenv("TARGET_DISCOVERY_CANDIDATES", "30"))
TWELVELABS_MAX_COMPETITORS: int = int(os.getenv("TWELVELABS_MAX_COMPETITORS", "10"))

# Twelve Labs Retries and Timeouts
TWELVELABS_RETRY_ATTEMPTS: int = int(os.getenv("TWELVELABS_RETRY_ATTEMPTS", "2"))
TWELVELABS_REQUEST_DELAY_SECONDS: float = float(
    os.getenv("TWELVELABS_REQUEST_DELAY_SECONDS", "1")
)
TWELVELABS_REQUEST_TIMEOUT_SECONDS: int = int(
    os.getenv("TWELVELABS_REQUEST_TIMEOUT_SECONDS", "180")
)

# Local Tool Paths & Model Configurations
FFMPEG_BIN: str = os.getenv("FFMPEG_BIN", "").strip() or shutil.which("ffmpeg") or "ffmpeg"


def _resolve_tesseract() -> str:
    env_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if env_cmd and os.path.exists(env_cmd):
        return env_cmd
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return which_cmd
    standard_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in standard_paths:
        if os.path.exists(p):
            return p
    return "tesseract"


TESSERACT_CMD: str = _resolve_tesseract()
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base").strip()
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()

# Application Storage Directories
MEDIA_DIR: Path = BASE_DIR / "media"
REPORTS_DIR: Path = BASE_DIR / "reports"
CACHE_DIR: Path = BASE_DIR / "cache"
DATA_DIR: Path = BASE_DIR / "data"
COMPETITORS_DATA_DIR: Path = DATA_DIR / "competitors"

# Create required runtime directories
for d in (MEDIA_DIR, REPORTS_DIR, CACHE_DIR, DATA_DIR, COMPETITORS_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)
