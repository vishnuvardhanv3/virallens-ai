"""Release Hardening Test Suite for ViralLens AI (Phase 18).

Comprehensive verification of:
1. Multi-format input validation (valid, 0-byte, corrupt, missing, unsupported)
2. Boundary failure handling & graceful degradation (TwelveLabs, Apify, OCR, TinySalNet)
3. Partial-result preservation (optional components fail without fake zeros or crashes)
4. Epistemic language audit (zero causal overclaiming in generated analysis)
5. Secret exposure audit (zero credentials in logs, reports, or telemetry)
6. Startup health validation
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
import pytest

from config import settings
from core.startup_check import HealthCheckItem, HealthState, SystemHealthReport, validate_system_health
from services.media_service import validate_video_file
from services.report_service import save_report
from services.recommendation_intelligence import RecommendationIntelligenceService
from services.competitor_intelligence import CompetitorIntelligenceService
from services.attention_intelligence import AttentionIntelligenceService


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def test_input_validation_missing_file():
    """Verify missing video file returns structured error rather than crashing."""
    res = validate_video_file("C:/nonexistent_video_path.mp4")
    assert res["valid"] is False
    assert "does not exist" in res["error"].lower()


def test_input_validation_zero_byte_file(temp_dir):
    """Verify zero-byte video file returns structured error."""
    zero_file = temp_dir / "zero.mp4"
    zero_file.write_bytes(b"")
    res = validate_video_file(zero_file)
    assert res["valid"] is False
    assert "zero bytes" in res["error"].lower()


def test_input_validation_corrupted_file(temp_dir):
    """Verify corrupted video file returns structured error."""
    corrupt_file = temp_dir / "corrupted.mp4"
    corrupt_file.write_bytes(b"NOT_A_VALID_MP4_HEADER_GARBAGE_BYTES_1234567890")
    res = validate_video_file(corrupt_file)
    assert res["valid"] is False
    assert "could not" in res["error"].lower() or "validation" in res["error"].lower()


def test_input_validation_unsupported_extension(temp_dir):
    """Verify unsupported extension returns structured error."""
    txt_file = temp_dir / "video.txt"
    txt_file.write_text("hello world")
    res = validate_video_file(txt_file)
    assert res["valid"] is False
    assert "unsupported" in res["error"].lower()


def test_input_validation_valid_benchmark_video():
    """Verify valid benchmark MP4 passes validation with decoded properties."""
    real_path = Path("htdyr.mp4")
    if not real_path.exists():
        real_path = Path("media/htdyr.mp4")
    if real_path.exists():
        res = validate_video_file(real_path)
        assert res["valid"] is True
        assert res["size_bytes"] > 0
        assert res["duration_sec"] > 0
        assert res["width"] > 0
        assert res["height"] > 0


def test_startup_health_check_execution():
    """Verify startup health check runs quickly and produces a structured report."""
    report = validate_system_health()
    assert report.overall_state in {HealthState.READY, HealthState.DEGRADED, HealthState.FAILED}
    assert report.python_version.startswith("3.")
    assert report.discovery_provider == "apify"
    assert len(report.checks) >= 7

    d = report.to_dict()
    assert "overall_state" in d
    assert "checks" in d


def test_optional_component_failure_graceful_degradation():
    """Verify optional components (OCR, Spatial Saliency) degrade cleanly without fake predictions."""
    # Target reel with NO OCR text and NO spatial saliency data
    empty_target = {
        "id": "test_target",
        "primary_entity": "fitness",
        "niche": "fitness",
        "ocr": {"text_present": False, "ocr_text": "", "segments": []},
        "spatial_saliency": None,  # Optional component absent
        "attention": {"visual_attention_score": 0.72},
        "hook": {"hook_type": "question", "confidence": 0.8},
    }

    report = RecommendationIntelligenceService.generate(
        target_profile=empty_target,
        attention_intelligence=None,
        competitor_intelligence=None,
    )

    assert report is not None
    assert "recommendations" in report
    assert isinstance(report["recommendations"], list)
    # When competitor baseline is absent, it cleanly returns status indicating unverified/no actionable recommendation
    assert report.get("status") in {"NO_ACTIONABLE_RECOMMENDATION", "INSUFFICIENT_COMPETITORS", "SUCCESS"}
    if report.get("status") == "NO_ACTIONABLE_RECOMMENDATION":
        assert report["recommendations"][0]["recommendation_id"] == "rec_null_no_competitors"


def test_epistemic_language_in_production_services():
    """Verify that production analysis services do NOT contain prohibited causal virality claims."""
    forbidden_claims = [
        "guaranteed viral",
        "will go viral",
        "guarantees retention",
        "will increase retention",
        "predicts instagram retention",
        "predicts scroll stops",
        "causes engagement",
        "proves viewers stopped scrolling",
    ]

    service_files = [
        Path("services/recommendation_intelligence.py"),
        Path("services/competitor_intelligence.py"),
        Path("services/attention_intelligence.py"),
        Path("services/report_service.py"),
    ]

    for sf in service_files:
        if sf.exists():
            text = sf.read_text(encoding="utf-8").lower()
            for claim in forbidden_claims:
                assert claim not in text, f"Forbidden claim '{claim}' found in {sf.name}"


def test_secret_exposure_audit_in_reports_and_diagnostics(temp_dir, monkeypatch):
    """Verify that report generation and telemetry never leak sensitive API credentials."""
    monkeypatch.setattr("config.settings.REPORTS_DIR", temp_dir)
    
    fake_payload = {
        "your_reel": {
            "id": "target_123",
            "primary_entity": "fitness",
            "niche": "workout",
            "url": "https://instagram.com/reel/C2_test/",
        },
        "discovery_diagnostics": {
            "provider": "apify",
            "status": "SUCCESS",
            "discovery_health": {"discovery_health": "healthy"},
            "query_results": [{"query": "workout motivation", "status": "success", "results_count": 5}],
        },
    }

    json_path, md_path = save_report(fake_payload, base_name="test_exposure")
    json_text = json_path.read_text(encoding="utf-8").lower()
    md_text = md_path.read_text(encoding="utf-8").lower()

    sensitive_keys = ["apify_api_token", "twelvelabs_api_key", "password", "bearer ", "sessionid="]
    for key in sensitive_keys:
        assert key not in json_text, f"Sensitive credential pattern '{key}' detected in final report JSON"
        assert key not in md_text, f"Sensitive credential pattern '{key}' detected in final report MD"


def test_regression_startup_health_normal_import_succeeds():
    """1. Normal import succeeds -> critical packages check is READY."""
    report = validate_system_health()
    crit_check = next(c for c in report.checks if c.name == "critical_packages")
    assert crit_check.state == HealthState.READY


def test_regression_startup_health_import_error_handling(monkeypatch):
    """2. ImportError on critical package -> correct health state FAILED."""
    import importlib
    real_import = importlib.import_module

    def mock_import(name, *args, **kwargs):
        if name == "requests":
            raise ImportError("No module named 'requests'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import)
    report = validate_system_health()
    crit_check = next(c for c in report.checks if c.name == "critical_packages")
    assert crit_check.state == HealthState.FAILED
    assert report.overall_state == HealthState.FAILED


def test_regression_startup_health_oserror_winerror_1114_no_crash(monkeypatch):
    """3. OSError WinError 1114 -> no Streamlit / startup crash, marked DEGRADED."""
    import importlib
    real_import = importlib.import_module

    err = OSError("A dynamic link library (DLL) initialization routine failed. Error loading: c10.dll")
    err.winerror = 1114

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise err
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import)
    report = validate_system_health()
    torch_check = next(c for c in report.checks if c.name == "torch_runtime")
    assert torch_check.state == HealthState.DEGRADED
    assert "WinError 1114" in torch_check.message
    assert torch_check.details.get("winerror") == 1114
    assert any("WinError 1114" in r for r in report.degraded_reasons)


def test_regression_startup_health_arbitrary_native_dll_exception_no_crash(monkeypatch):
    """4. Arbitrary native DLL exception -> no Streamlit / startup crash, marked DEGRADED."""
    import importlib
    real_import = importlib.import_module

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("Fatal native C++ runtime error during DLL entry point")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import)
    report = validate_system_health()
    torch_check = next(c for c in report.checks if c.name == "torch_runtime")
    assert torch_check.state == HealthState.DEGRADED
    assert torch_check.details.get("exception_type") == "RuntimeError"


def test_regression_startup_health_required_dependency_failure_sets_failed(monkeypatch):
    """5. Required dependency failure -> FAILED."""
    import importlib
    real_import = importlib.import_module

    def mock_import(name, *args, **kwargs):
        if name == "cv2":
            raise OSError("DLL initialization routine failed for cv2.pyd")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import)
    report = validate_system_health()
    assert report.overall_state == HealthState.FAILED
    crit_check = next(c for c in report.checks if c.name == "critical_packages")
    assert crit_check.state == HealthState.FAILED
    assert any("opencv-python" in r for r in report.failed_reasons)


def test_regression_startup_health_runtime_dependency_failure_sets_degraded(monkeypatch):
    """6. Optional / runtime dependency failure -> DEGRADED, not FAILED."""
    import importlib
    real_import = importlib.import_module

    err = OSError("A dynamic link library (DLL) initialization routine failed. Error loading: c10.dll")
    err.winerror = 1114

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise err
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import)
    report = validate_system_health()
    torch_check = next(c for c in report.checks if c.name == "torch_runtime")
    assert torch_check.state == HealthState.DEGRADED
    crit_check = next(c for c in report.checks if c.name == "critical_packages")
    assert crit_check.state == HealthState.READY
    assert not any("torch" in r for r in report.failed_reasons)
    assert any("torch" in r.lower() or "winerror 1114" in r.lower() for r in report.degraded_reasons)


def test_regression_startup_health_report_remains_serializable():
    """7. Health report remains serializable via to_dict and json.dumps."""
    report = SystemHealthReport(
        overall_state=HealthState.DEGRADED,
        python_version="3.12.4",
        discovery_provider="apify",
        checks=[
            HealthCheckItem(
                name="torch_runtime",
                category="dependencies",
                state=HealthState.DEGRADED,
                message="PyTorch native DLL initialization failed (WinError 1114).",
                details={"winerror": 1114, "exception_type": "OSError"},
            ),
            HealthCheckItem(
                name="critical_packages",
                category="dependencies",
                state=HealthState.READY,
                message="All critical packages installed and verified.",
            ),
        ],
        failed_reasons=[],
        degraded_reasons=["PyTorch native DLL initialization failed in the current process (WinError 1114)."],
    )

    d = report.to_dict()
    assert isinstance(d, dict)
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["overall_state"] == "DEGRADED"
    assert len(deserialized["checks"]) == 2
    assert deserialized["checks"][0]["details"]["winerror"] == 1114
    assert len(deserialized["degraded_reasons"]) == 1

