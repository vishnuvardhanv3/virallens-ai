"""Tests for presentation formatting in MetricsService."""
from services.metrics_service import format_percentage, format_number, calculate_engagement_rate


def test_percentage_formatting():
    # Canonical 0.8 displays as 80% (never 8% or 800%)
    assert format_percentage(0.8) == "80%"
    assert format_percentage(8) == "80%"
    assert format_percentage(80) == "80%"
    assert format_percentage(0.73) == "73%"
    assert format_percentage(None) == "Unavailable"
    assert format_percentage("", fallback="N/A") == "N/A"


def test_engagement_rate():
    assert calculate_engagement_rate(100, 20, 1000) == 0.12
    assert calculate_engagement_rate(None, None, 1000) is None
    assert calculate_engagement_rate(10, 5, 0) is None
    assert calculate_engagement_rate(10, 5, None) is None


def test_number_formatting():
    assert format_number(1234567) == "1,234,567"
    assert format_number(0) == "0"
    assert format_number(None) == "Unavailable"
