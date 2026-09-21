"""Base interface for Instagram Reel discovery providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

DiscoveryResult = Tuple[List[Dict[str, Any]], Dict[str, Any]]


class ProviderBase(ABC):
    """Abstract base class for Reel discovery providers."""

    def __init__(self) -> None:
        self.configured = False
        self.last_query: str | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        raise NotImplementedError

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required credentials / environment are present."""
        raise NotImplementedError

    @abstractmethod
    def discover_reels(
        self, queries: List[Any], limit: int
    ) -> DiscoveryResult:
        """Discover Instagram Reels matching queries."""
        raise NotImplementedError

    def _make_error(
        self, error_type: str, message: str, http_status: int = 500
    ) -> DiscoveryResult:
        return (
            [],
            {
                "success": False,
                "error_type": error_type,
                "error": message,
                "http_status": http_status,
                "provider": self.name,
                "last_query": self.last_query,
            },
        )

    def _make_success(
        self, candidates: List[Dict[str, Any]], http_status: int = 200
    ) -> DiscoveryResult:
        return (
            candidates,
            {
                "success": True,
                "candidates_count": len(candidates),
                "http_status": http_status,
                "provider": self.name,
                "last_query": self.last_query,
            },
        )
