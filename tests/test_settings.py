"""Tests for configuration settings and defaults."""
from config import settings


def test_settings_defaults():
    assert settings.DISCOVERY_MODE == "provider"
    assert settings.INSTAGRAM_PROVIDER_ORDER == "apify"
    assert settings.INSTAGRAM_MAX_RESULTS == 100
    assert settings.APIFY_SEARCH_LIMIT == 25
    assert settings.TWELVELABS_MAX_COMPETITORS == 10
    assert settings.TWELVELABS_RETRY_ATTEMPTS == 2
    assert settings.TWELVELABS_REQUEST_DELAY_SECONDS == 1.0
    assert settings.TWELVELABS_REQUEST_TIMEOUT_SECONDS == 180
    assert settings.TWELVELABS_ENGINE == "pegasus1.5"
    assert settings.WHISPER_MODEL == "base"
    assert settings.EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.APIFY_INSTAGRAM_REELS_ACTOR == "apify/instagram-search-scraper"


def test_runtime_directories_exist():
    assert settings.MEDIA_DIR.exists()
    assert settings.REPORTS_DIR.exists()
    assert settings.CACHE_DIR.exists()
    assert settings.COMPETITORS_DATA_DIR.exists()
