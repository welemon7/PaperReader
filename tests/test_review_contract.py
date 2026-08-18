"""100-point contract: verdict computation, dimension extraction, normalization."""

from __future__ import annotations

from src.agents.reviewer import extract_review_dimensions
from src.schemas.review import (
    DIMENSION_WEIGHTS,
    ReviewDimensions,
    compute_verdict,
)


def _full_dims(**overrides) -> ReviewDimensions:
    base = dict(
        layout_hierarchy=8.0, readability_overflow=8.0, figures_storytelling=8.0,
        content_coverage_facts=8.0, color_accessibility=8.0,
    )
    base.update(overrides)
    return ReviewDimensions(**base)


def test_weighted_total_matches_weights():
    dims = ReviewDimensions(
        layout_hierarchy=10.0, readability_overflow=10.0, figures_storytelling=10.0,
        content_coverage_facts=10.0, color_accessibility=10.0,
    )
    assert dims.weighted_total() == 100.0
    assert sum(DIMENSION_WEIGHTS.values()) == 100


def test_weighted_total_partial():
    dims = _full_dims(layout_hierarchy=5.0)
    # 5/10*30 + 8/10*70 = 15 + 56 = 71
    assert dims.weighted_total() == 71.0


def test_compute_verdict_passes_when_all_gates_ok():
    verdict = compute_verdict(_full_dims(layout_hierarchy=9.0, readability_overflow=9.0,
                                         figures_storytelling=9.0, content_coverage_facts=9.0,
                                         color_accessibility=9.0),
                              hard_error_count=0, qa_accuracy=0.9)
    assert verdict.passed is True
    assert verdict.total_score == 90.0
    assert len(verdict.gates) == 4
    assert all(g.passed for g in verdict.gates)
    assert verdict.reasons == []


def test_compute_verdict_fails_on_hard_errors_even_with_high_score():
    dims = _full_dims(layout_hierarchy=10.0, readability_overflow=10.0,
                      figures_storytelling=10.0, content_coverage_facts=10.0,
                      color_accessibility=10.0)
    verdict = compute_verdict(dims, hard_error_count=1, qa_accuracy=0.95)
    assert verdict.passed is False
    assert "no_hard_errors" in verdict.reasons


def test_compute_verdict_fails_below_total_threshold():
    verdict = compute_verdict(_full_dims(layout_hierarchy=2.0), hard_error_count=0, qa_accuracy=1.0)
    # total = 2/10*30 + 56 = 62
    assert verdict.passed is False
    assert "total_score" in verdict.reasons
    assert "dimension_minimums" in verdict.reasons


def test_compute_verdict_fails_on_low_qa():
    verdict = compute_verdict(_full_dims(), hard_error_count=0, qa_accuracy=0.5)
    assert verdict.passed is False
    assert "image_qa" in verdict.reasons


def test_compute_verdict_not_run_qa_is_fail_gate():
    verdict = compute_verdict(_full_dims(), hard_error_count=0, qa_accuracy=None)
    assert verdict.passed is False
    assert any(g.name == "image_qa" and not g.passed for g in verdict.gates)


def test_compute_verdict_respects_config_thresholds():
    verdict = compute_verdict(
        _full_dims(layout_hierarchy=7.0), hard_error_count=0, qa_accuracy=0.75,
        total_threshold=70.0, dim_fraction=0.5, qa_threshold=0.7,
    )
    # total = 7/10*30 + 56 = 77 >= 70; dim 7 >= 30*0.5=15? 7/10*30=21 >= 15 ok; qa 0.75 >= 0.7
    assert verdict.passed is True


def test_extract_review_dimensions_canonical():
    raw = {
        "dimension_scores": {
            "layout_hierarchy": 9, "readability_overflow": 8,
            "figures_storytelling": 7, "content_coverage_facts": 6,
            "color_accessibility": 10,
        }
    }
    dims = extract_review_dimensions(raw)
    assert dims.layout_hierarchy == 9.0
    assert dims.color_accessibility == 10.0
    # 9/10*30 + 8/10*25 + 7/10*20 + 6/10*20 + 10/10*5 = 27+20+14+12+5 = 78
    assert dims.weighted_total() == 78.0


def test_extract_review_dimensions_aliases():
    raw = {"dimension_scores": {"layout": 8, "readability": 7, "figures": 6, "content": 5, "color": 9}}
    dims = extract_review_dimensions(raw)
    assert dims.layout_hierarchy == 8.0
    assert dims.readability_overflow == 7.0
    assert dims.figures_storytelling == 6.0
    assert dims.content_coverage_facts == 5.0


def test_extract_review_dimensions_partial_falls_back_to_quality():
    # 只有 2 个维度时不能把其余 3 个当 0 分 —— 回退到 quality_score
    raw = {"quality_score": 7, "dimension_scores": {"layout": 8, "typography": 7}}
    dims = extract_review_dimensions(raw)
    assert dims.layout_hierarchy == 7.0
    assert dims.readability_overflow == 7.0
    assert dims.color_accessibility == 7.0


def test_extract_review_dimensions_empty_scores_zero():
    dims = extract_review_dimensions({"summary": "ok"})
    assert dims.weighted_total() == 0.0
