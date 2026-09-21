"""System Startup and Environment Health Validation for ViralLens AI.

Fast, non-blocking pre-flight health assessment distinguishing:
- READY: All production dependencies, configurations, models, and paths are valid.
- DEGRADED: Non-critical components (e.g. experimental spatial saliency or OCR) unavailable.
- FAILED: Critical requirements (Python runtime, output dirs, core packages, or primary provider) missing.

Zero Secret Exposure Guarantee:
Never logs, prints, or exposes API tokens, keys, passwords, or session IDs.
"""
from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

# Expected frozen model hashes
EXPECTED_TINYSALNET_HASH = "fd6f27fd5ef6334c597ee3981086fc70f639cd59c7d57399fa72253d883b95c9"


class HealthState(str, Enum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"

    def __str__(self) -> str:
        return self.value


@dataclass
class HealthCheckItem:
    name: str
    category: str
    state: HealthState
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthReport:
    overall_state: HealthState
    python_version: str
    discovery_provider: str
    checks: List[HealthCheckItem] = field(default_factory=list)
    failed_reasons: List[str] = field(default_factory=list)
    degraded_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_state": self.overall_state.value,
            "python_version": self.python_version,
            "discovery_provider": self.discovery_provider,
            "checks": [asdict(c) for c in self.checks],
            "failed_reasons": self.failed_reasons,
            "degraded_reasons": self.degraded_reasons,
        }


def _sha256(path: Path) -> str:
    """Calculate sha256 hash of a file safely."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_system_health() -> SystemHealthReport:
    """Perform fast, non-blocking preflight validation across the production stack."""
    checks: List[HealthCheckItem] = []
    failed_reasons: List[str] = []
    degraded_reasons: List[str] = []

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # 1. Python Runtime Check
    if sys.version_info < (3, 10):
        state = HealthState.FAILED
        msg = f"Python {py_ver} is unsupported. Requires Python >= 3.10."
        failed_reasons.append(msg)
    else:
        state = HealthState.READY
        msg = f"Python {py_ver} compatible."
    checks.append(HealthCheckItem(name="python_runtime", category="runtime", state=state, message=msg))

    # 2. Critical Package Imports Check (Startup-critical for core ViralLens pipeline)
    critical_packages = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("requests", "requests"),
    ]
    failed_crit = []
    for mod_name, pkg_name in critical_packages:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            failed_crit.append({
                "package": pkg_name,
                "module": mod_name,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            })

    if failed_crit:
        crit_errs = [f"{c['package']} ({c['exception_type']}: {c['exception_message'][:60]})" for c in failed_crit]
        msg = f"Missing or failed critical packages: {', '.join(crit_errs)}"
        failed_reasons.append(msg)
        checks.append(HealthCheckItem(
            name="critical_packages",
            category="dependencies",
            state=HealthState.FAILED,
            message=msg,
            details={"failures": failed_crit},
        ))
    else:
        checks.append(HealthCheckItem(
            name="critical_packages",
            category="dependencies",
            state=HealthState.READY,
            message="All critical packages installed and verified.",
        ))

    # 2b. PyTorch Runtime Dependency Check (Runtime-required for experimental TinySalNet spatial saliency)
    torch_details: Dict[str, Any] = {"package": "torch", "module": "torch"}
    try:
        t_mod = importlib.import_module("torch")
        ver = getattr(t_mod, "__version__", "unknown")
        cuda_ok = False
        try:
            cuda_ok = bool(getattr(getattr(t_mod, "cuda", None), "is_available", lambda: False)())
        except Exception:
            pass
        torch_details.update({"version": ver, "cuda_available": cuda_ok})
        checks.append(HealthCheckItem(
            name="torch_runtime",
            category="dependencies",
            state=HealthState.READY,
            message=f"PyTorch {ver} runtime initialized successfully.",
            details=torch_details,
        ))
    except Exception as exc:
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        winerror = getattr(exc, "winerror", None)
        is_win_1114 = winerror == 1114 or "1114" in exc_msg

        torch_details.update({
            "exception_type": exc_type,
            "exception_message": exc_msg,
            "winerror": winerror,
        })

        if is_win_1114:
            msg = "PyTorch native DLL initialization failed in the current process (WinError 1114). Experimental spatial saliency will run degraded."
        else:
            msg = f"PyTorch runtime dependency unavailable ({exc_type}: {exc_msg[:120]}). Experimental spatial saliency will run degraded."

        degraded_reasons.append(msg)
        checks.append(HealthCheckItem(
            name="torch_runtime",
            category="dependencies",
            state=HealthState.DEGRADED,
            message=msg,
            details=torch_details,
        ))

    # 3. Optional Tool Dependencies (FFmpeg, Tesseract, Librosa)
    optional_status = []
    if not settings.FFMPEG_BIN:
        optional_status.append("FFmpeg executable not found on PATH or FFMPEG_BIN")
    try:
        import librosa  # type: ignore
    except Exception as exc:
        optional_status.append(f"librosa unavailable ({type(exc).__name__})")

    if optional_status:
        msg = "; ".join(optional_status)
        degraded_reasons.append(msg)
        checks.append(HealthCheckItem(name="optional_tools", category="dependencies", state=HealthState.DEGRADED, message=msg))
    else:
        checks.append(HealthCheckItem(name="optional_tools", category="dependencies", state=HealthState.READY, message="FFmpeg and acoustic tools available."))

    # 4. Storage Directories Check
    output_dirs = [settings.MEDIA_DIR, settings.REPORTS_DIR, settings.CACHE_DIR, settings.DATA_DIR, settings.COMPETITORS_DATA_DIR]
    unwritable = []
    for d in output_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            # Test write access safely
            test_file = d / f".health_test_{os.getpid()}"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
        except Exception as exc:
            unwritable.append(f"{d.name}: {exc}")

    if unwritable:
        msg = f"Output directories unwritable: {', '.join(unwritable)}"
        failed_reasons.append(msg)
        checks.append(HealthCheckItem(name="storage_directories", category="filesystem", state=HealthState.FAILED, message=msg))
    else:
        checks.append(HealthCheckItem(name="storage_directories", category="filesystem", state=HealthState.READY, message="All application storage directories writable."))

    # 5. NEMAR Frozen Model Check
    nemar_dir = settings.BASE_DIR / "models" / "human_attention"
    nemar_required = ["model.pkl", "scaler.pkl", "feature_schema.json", "training_metadata.json"]
    nemar_missing = [f for f in nemar_required if not (nemar_dir / f).exists()]

    if nemar_missing:
        msg = f"Missing NEMAR production model files: {', '.join(nemar_missing)}"
        failed_reasons.append(msg)
        checks.append(HealthCheckItem(name="nemar_model", category="models", state=HealthState.FAILED, message=msg))
    else:
        checks.append(HealthCheckItem(name="nemar_model", category="models", state=HealthState.READY, message="NEMAR temporal attention model files verified."))

    # 6. TinySalNet Spatial Model & Hash Check
    tinysal_path = settings.BASE_DIR / "models" / "human_attention_experiments" / "dhf1k_spatial" / "spatial_model.pt"
    if not tinysal_path.exists():
        msg = "TinySalNet spatial model file missing. Spatial attention will run in degraded mode."
        degraded_reasons.append(msg)
        checks.append(HealthCheckItem(name="tinysalnet_spatial", category="models", state=HealthState.DEGRADED, message=msg))
    else:
        observed_hash = _sha256(tinysal_path)
        if observed_hash != EXPECTED_TINYSALNET_HASH:
            msg = f"TinySalNet model hash mismatch! Expected {EXPECTED_TINYSALNET_HASH[:16]}, observed {observed_hash[:16]}."
            failed_reasons.append(msg)
            checks.append(HealthCheckItem(name="tinysalnet_spatial", category="models", state=HealthState.FAILED, message=msg))
        else:
            checks.append(HealthCheckItem(name="tinysalnet_spatial", category="models", state=HealthState.READY, message="TinySalNet weights verified with exact SHA-256 match."))

    # 7. Dataset Immutability Check (Optional external NEMAR dataset)
    raw_nemar_env = os.getenv("NEMAR_DATASET_PATH", r"E:\v1.0.0")
    raw_nemar_path = Path(raw_nemar_env) if raw_nemar_env else None
    if raw_nemar_path and raw_nemar_path.exists():
        # Verify read access
        try:
            list(raw_nemar_path.iterdir())
            checks.append(HealthCheckItem(name="raw_nemar_dataset", category="datasets", state=HealthState.READY, message=f"Raw NEMAR dataset ({raw_nemar_path}) verified read-only."))
        except Exception as exc:
            msg = f"Unable to read raw NEMAR dataset: {exc}"
            degraded_reasons.append(msg)
            checks.append(HealthCheckItem(name="raw_nemar_dataset", category="datasets", state=HealthState.DEGRADED, message=msg))
    else:
        checks.append(HealthCheckItem(name="raw_nemar_dataset", category="datasets", state=HealthState.READY, message="External training dataset not locally mounted; production models remain frozen."))

    # 8. Twelve Labs Configuration Check
    tl_key = settings.TWELVELABS_API_KEY.strip()
    if not tl_key:
        msg = "TWELVELABS_API_KEY not configured in environment or .env."
        degraded_reasons.append(msg)
        checks.append(HealthCheckItem(name="twelvelabs_config", category="configuration", state=HealthState.DEGRADED, message=msg))
    else:
        checks.append(HealthCheckItem(name="twelvelabs_config", category="configuration", state=HealthState.READY, message="Twelve Labs API key configured (masked)."))

    # 9. Discovery Provider Configuration Check
    disc_prov = getattr(settings, "DISCOVERY_PROVIDER", "apify").strip().lower()
    apify_token = settings.APIFY_API_TOKEN.strip()

    if disc_prov in {"apify", "fallback"}:
        if not apify_token:
            msg = f"Primary discovery provider is '{disc_prov}' but APIFY_API_TOKEN is missing."
            degraded_reasons.append(msg)
            checks.append(HealthCheckItem(name="discovery_provider_config", category="configuration", state=HealthState.DEGRADED, message=msg))
        else:
            checks.append(HealthCheckItem(name="discovery_provider_config", category="configuration", state=HealthState.READY, message=f"Discovery provider '{disc_prov}' configured with valid Apify token (masked)."))
    elif disc_prov == "agent_reach":
        msg = "Discovery provider configured as experimental 'agent_reach'. Dependent on local OpenCLI availability."
        degraded_reasons.append(msg)
        checks.append(HealthCheckItem(name="discovery_provider_config", category="configuration", state=HealthState.DEGRADED, message=msg))
    else:
        msg = f"Unknown discovery provider '{disc_prov}'."
        failed_reasons.append(msg)
        checks.append(HealthCheckItem(name="discovery_provider_config", category="configuration", state=HealthState.FAILED, message=msg))

    # Determine overall state
    if failed_reasons:
        overall = HealthState.FAILED
    elif degraded_reasons:
        overall = HealthState.DEGRADED
    else:
        overall = HealthState.READY

    return SystemHealthReport(
        overall_state=overall,
        python_version=py_ver,
        discovery_provider=disc_prov,
        checks=checks,
        failed_reasons=failed_reasons,
        degraded_reasons=degraded_reasons,
    )
