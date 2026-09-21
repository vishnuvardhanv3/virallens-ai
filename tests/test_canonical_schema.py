"""Tests for canonical schema and score normalization."""
from services.canonical_schema import normalize_score_0_to_1, build_canonical_analysis


def test_hook_normalization_rules():
    # 0.8 remains 0.8
    assert normalize_score_0_to_1(0.8) == 0.8
    # 8 on 1-10 scale becomes 0.8
    assert normalize_score_0_to_1(8) == 0.8
    assert normalize_score_0_to_1("8") == 0.8
    # 80 on percentage scale becomes 0.8
    assert normalize_score_0_to_1(80) == 0.8
    assert normalize_score_0_to_1("80") == 0.8
    # Missing values remain None (never fake zero)
    assert normalize_score_0_to_1(None) is None
    assert normalize_score_0_to_1("") is None
    assert normalize_score_0_to_1("invalid") is None


def test_build_canonical_analysis_structure():
    canonical = build_canonical_analysis(
        analysis={"niche": "superhero", "topic": "spiderman fight", "primary_entity": "spiderman"},
        hook={"type": "visual hook", "strength": 8, "first_3_seconds": "Spiderman flips onto car"},
        visual={"visual_impact": 75, "scene_count": 12},
        engagement={"curiosity": 9, "views": 150000},
    )
    assert canonical["niche"] == "superhero"
    assert canonical["primary_entity"] == "spiderman"
    assert canonical["hook"]["strength"] == 0.8
    assert canonical["visual"]["visual_impact"] == 0.75
    assert canonical["engagement"]["curiosity"] == 0.9
    assert canonical["engagement"]["views"] == 150000
    # Missing fields remain None
    assert canonical["structure"]["cta"] is None
