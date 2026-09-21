"""Test attention timeline chronological sorting.

Verifies:
1. Input [0.0, 2.0, 0.5, 2.5, 1.0, 3.0] sorts to [0.0, 0.5, 1.0, 2.0, 2.5, 3.0].
2. Field preservation (start_time, end_time, score, event metadata).
3. Handling of missing or string timestamps without error.
4. Markdown report includes chronologically ordered timeline events.
"""
import pytest
from services.report_service import sort_timeline_events, _build_markdown_report


def test_timeline_numeric_sorting_exact_requirement():
    """Verify exact requirement: input [0.0, 2.0, 0.5, 2.5, 1.0, 3.0] -> [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]."""
    test_input = [
        {"start_time": 0.0, "end_time": 0.5, "event": "ev0"},
        {"start_time": 2.0, "end_time": 2.5, "event": "ev2"},
        {"start_time": 0.5, "end_time": 1.0, "event": "ev0.5"},
        {"start_time": 2.5, "end_time": 3.0, "event": "ev2.5"},
        {"start_time": 1.0, "end_time": 1.5, "event": "ev1"},
        {"start_time": 3.0, "end_time": 3.5, "event": "ev3"},
    ]

    sorted_res = sort_timeline_events(test_input)
    actual_starts = [e["start_time"] for e in sorted_res]
    expected_starts = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    assert actual_starts == expected_starts


def test_timeline_field_preservation():
    """Verify all fields and metadata are completely preserved during sorting."""
    event = {
        "timestamp": "1.5s–2.0s",
        "start_time": 1.5,
        "end_time": 2.0,
        "attention_score": 0.62,
        "confidence": 0.89,
        "major_observed_event": "on-screen textual graphics",
        "attention_mechanism": "Temporal window characterized by text.",
        "baseline_delta": 0.08,
        "is_local_peak": True,
        "peak_prominence": 0.05,
    }
    events = [
        {"start_time": 3.0, "attention_score": 0.45},
        event,
        {"start_time": 0.0, "attention_score": 0.50},
    ]

    res = sort_timeline_events(events)
    assert len(res) == 3
    assert res[1]["start_time"] == 1.5
    assert res[1]["attention_score"] == 0.62
    assert res[1]["major_observed_event"] == "on-screen textual graphics"
    assert res[1]["peak_prominence"] == 0.05
    assert res[1]["is_local_peak"] is True


def test_timeline_string_and_missing_values_handled_gracefully():
    """Verify string numbers or missing values sort safely without crashing."""
    events = [
        {"start_time": "2.0", "name": "second"},
        {"start_time": None, "name": "null"},
        {"start_time": 0.5, "name": "first"},
    ]
    res = sort_timeline_events(events)
    assert len(res) == 3
    assert res[0]["name"] == "null"  # 0.0
    assert res[1]["start_time"] == 0.5
    assert res[2]["name"] == "second"


def test_markdown_report_chronological_ordering():
    """Verify markdown report outputs timeline events in strictly chronological order."""
    data = {
        "your_reel": {},
        "agent_results": {
            "human_attention": {
                "timeline_events": [
                    {"start_time": 2.0, "timestamp": "2.0s–2.5s", "attention_score": 0.55, "major_observed_event": "event2"},
                    {"start_time": 0.0, "timestamp": "0.0s–0.5s", "attention_score": 0.52, "major_observed_event": "event0"},
                    {"start_time": 1.0, "timestamp": "1.0s–1.5s", "attention_score": 0.53, "major_observed_event": "event1"},
                ]
            }
        }
    }
    md = _build_markdown_report(data)
    idx0 = md.find("0.0s–0.5s")
    idx1 = md.find("1.0s–1.5s")
    idx2 = md.find("2.0s–2.5s")
    assert idx0 != -1 and idx1 != -1 and idx2 != -1
    assert idx0 < idx1 < idx2
