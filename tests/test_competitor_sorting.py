"""Tests proving highest-reach competitor sorting and order preservation.

Verifies:
1. highest reach appears first
2. descending ordering is preserved in JSON
3. descending ordering is preserved in Markdown
4. Streamlit receives already-ranked competitors
5. missing reach values do not break sorting
6. verified competitor selection still respects creator diversity
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agents.competitor_agent import rank_candidates, select_final_competitors, _safe_int_reach
from services.report_service import save_report, _build_markdown_report


# 1. Highest reach appears first
def test_highest_reach_appears_first():
    candidates = [
        {"id": "c1", "views": 1630558, "creator": "a"},
        {"id": "c2", "views": 11945712, "creator": "b"},
        {"id": "c3", "views": 11527029, "creator": "c"},
    ]
    ranked = rank_candidates(candidates)
    reaches = [_safe_int_reach(c) for c in ranked]
    assert reaches == [11945712, 11527029, 1630558]
    assert ranked[0]["id"] == "c2"
    assert ranked[1]["id"] == "c3"
    assert ranked[2]["id"] == "c1"


# 2. Descending ordering is preserved in JSON
def test_descending_ordering_preserved_in_json(tmp_path: Path):
    candidates = [
        {"id": "c1", "shortcode": "c1", "views": 1630558, "creator": "a", "reach_value": 1630558},
        {"id": "c2", "shortcode": "c2", "views": 11945712, "creator": "b", "reach_value": 11945712},
        {"id": "c3", "shortcode": "c3", "views": 11527029, "creator": "c", "reach_value": 11527029},
    ]
    # Intentionally shuffle/unsort before passing into save_report
    payload = {
        "your_reel": {"niche": "superhero"},
        "verified_competitors": [candidates[0], candidates[2], candidates[1]],
    }
    json_path, _ = save_report(payload, base_name="test_sort_order")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    verified = data.get("verified_competitors", [])
    reaches = [_safe_int_reach(c) for c in verified]
    assert reaches == [11945712, 11527029, 1630558]


# 3. Descending ordering is preserved in Markdown
def test_descending_ordering_preserved_in_markdown():
    candidates = [
        {"id": "c1", "shortcode": "c1", "views": 1630558, "creator": "jenkz.p", "reach_value": 1630558},
        {"id": "c2", "shortcode": "c2", "views": 11945712, "creator": "matiko.tt", "reach_value": 11945712},
        {"id": "c3", "shortcode": "c3", "views": 11527029, "creator": "ghxz.edit", "reach_value": 11527029},
    ]
    payload = {
        "your_reel": {"niche": "superhero"},
        "verified_competitors": [candidates[0], candidates[1], candidates[2]],  # unsorted
    }
    md = _build_markdown_report(payload)
    
    # Verify section 6 order
    pos_c2 = md.find("11,945,712")
    pos_c3 = md.find("11,527,029")
    pos_c1 = md.find("1,630,558")
    
    assert pos_c2 != -1 and pos_c3 != -1 and pos_c1 != -1
    assert pos_c2 < pos_c3 < pos_c1, "Markdown does not display competitors in descending reach order!"


# 4. Streamlit receives already-ranked competitors
def test_streamlit_receives_already_ranked_competitors():
    from core.master_agent import MasterAgent
    # Simulate candidates processed through selection pipeline
    raw_verified = [
        {"id": "c1", "views": 1630558, "creator": "a", "video_relevance_score": 0.85},
        {"id": "c2", "views": 11945712, "creator": "b", "video_relevance_score": 0.85},
        {"id": "c3", "views": 11527029, "creator": "c", "video_relevance_score": 0.85},
    ]
    final_verified, _ = select_final_competitors(raw_verified, max_competitors=10, max_per_creator=2)
    
    # Streamlit dashboard consumes final_verified directly from results dictionary
    reaches = [_safe_int_reach(c) for c in final_verified]
    assert reaches == [11945712, 11527029, 1630558], "Competitors received by UI are not pre-sorted!"


# 5. Missing reach values do not break sorting
def test_missing_reach_values_do_not_break_sorting():
    candidates = [
        {"id": "none_reach", "views": None, "plays": None, "creator": "x"},
        {"id": "high", "views": 11945712, "creator": "y"},
        {"id": "invalid_reach", "views": "not_a_number", "creator": "z"},
        {"id": "low", "plays": 1000, "creator": "w"},
        {"id": "missing_key", "creator": "v"},
    ]
    ranked = rank_candidates(candidates)
    reaches = [_safe_int_reach(c) for c in ranked]
    
    # Positive reaches sorted descending, zero/missing placed at bottom safely
    assert reaches[0] == 11945712
    assert reaches[1] == 1000
    assert reaches[2] == 0
    assert reaches[3] == 0
    assert reaches[4] == 0


# 6. Verified competitor selection still respects creator diversity
def test_verified_competitor_selection_respects_creator_diversity():
    candidates = [
        {"id": "c1_a", "creator": "creator_popular", "views": 20000000, "video_relevance_score": 0.9},
        {"id": "c2_a", "creator": "creator_popular", "views": 18000000, "video_relevance_score": 0.9},
        {"id": "c3_a", "creator": "creator_popular", "views": 16000000, "video_relevance_score": 0.9},  # exceeds 2/creator
        {"id": "c1_b", "creator": "creator_b", "views": 15000000, "video_relevance_score": 0.9},
        {"id": "c1_c", "creator": "creator_c", "views": 12000000, "video_relevance_score": 0.9},
        {"id": "c1_d", "creator": "creator_d", "views": 10000000, "video_relevance_score": 0.9},
        {"id": "c1_e", "creator": "creator_e", "views": 8000000, "video_relevance_score": 0.9},
        {"id": "c1_f", "creator": "creator_f", "views": 6000000, "video_relevance_score": 0.9},
    ]
    selected, diag = select_final_competitors(candidates, max_competitors=10, max_per_creator=2)
    
    # Creator 'creator_popular' must have at most 2 reels
    creator_counts = {}
    for c in selected:
        creator_counts[c["creator"]] = creator_counts.get(c["creator"], 0) + 1
    assert creator_counts["creator_popular"] == 2
    
    # Still sorted descending by reach
    reaches = [_safe_int_reach(c) for c in selected]
    for i in range(len(reaches) - 1):
        assert reaches[i] >= reaches[i + 1]
    
    assert reaches[0] == 20000000
    assert reaches[1] == 18000000
    assert reaches[2] == 15000000
