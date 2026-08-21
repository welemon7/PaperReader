"""Harness v2 loop: 100-point contract path (review mocked, no browser)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.agents.poster_harness import run_poster_harness
from src.schemas.analysis import Contribution, ExperimentSummary, PaperAnalysis
from src.schemas.paper import Author, Figure, PaperDocument, Section
from src.schemas.poster_harness import HarnessConfig
from src.schemas.poster_v2 import PosterComment, PosterReview


def _make_doc() -> PaperDocument:
    return PaperDocument(
        paper_id="v2-1", arxiv_id="9999.99997", title="V2 Paper",
        authors=[Author(name="Alice")], abstract="A.",
        sections=[Section(section_id="s1", title="Intro", level=1, text="x", raw_latex="x")],
        figures=[Figure(figure_id="fig-1", caption="Architecture", section_id="s1")],
        raw_markdown="# x",
    )


def _make_analysis() -> PaperAnalysis:
    return PaperAnalysis(
        paper_id="v2-1", arxiv_id="9999.99997", title_zh="",
        problem_statement="We solve classification.",
        contributions=[Contribution(text="Better backbone", category="method")],
        method_overview="New architecture.",
        key_formulas=[], key_figures=[],
        experiments=ExperimentSummary(main_results="99%", takeaways=["Works"]),
        conclusion="Done.", full_analysis_md="# A",
    )


def _review_with_verdict(total: float, hard_errors: int = 0, passed: bool | None = None) -> PosterReview:
    passed = (total >= 85.0 and hard_errors == 0) if passed is None else passed
    return PosterReview(
        quality_score=int(round(total / 10)),
        total_score=total,
        hard_error_count=hard_errors,
        needs_improvement=not passed,
        issues=[],
        summary="ok",
        verdict={
            "computed": True,
            "passed": passed,
            "total_score": total,
            "hard_error_count": hard_errors,
            "gates": [{"name": "total_score", "passed": passed, "required": ">= 85", "actual": str(total)}],
            "reasons": [] if passed else ["total_score"],
        },
        dimension_breakdown={"layout_hierarchy": 30.0, "readability_overflow": 25.0,
                             "figures_storytelling": 20.0, "content_coverage_facts": 20.0,
                             "color_accessibility": 5.0},
        dimension_scores={"layout_hierarchy": 10.0, "readability_overflow": 10.0,
                          "figures_storytelling": 10.0, "content_coverage_facts": 10.0,
                          "color_accessibility": 10.0},
    )


def _config(**kw) -> HarnessConfig:
    defaults = dict(
        threshold=8, max_rounds=5, zoom_crops=False, enable_qa_eval=False,
        advanced_visual=True, pass_total=85.0, pass_dim_fraction=0.6,
        plateau_rounds=2, improvement_delta=2.0, max_figure_crops=0,
    )
    defaults.update(kw)
    return HarnessConfig(**defaults)


def _render_initial(output_dir: Path) -> Path:
    html = output_dir / "poster_draft.html"
    html.write_text("<html><body><div class='poster-container'></div></body></html>", encoding="utf-8")
    return html


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_harness_v2_passes_with_100pt_contract(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    from src.agents.poster_planner import generate_blueprint
    blueprint = generate_blueprint(doc, analysis)
    initial = _render_initial(tmp_path)

    with patch("src.agents.poster_harness.review_rendered_poster_v2",
               side_effect=[_review_with_verdict(72.0), _review_with_verdict(90.0, passed=True)]):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=_config())

    assert result.passed is True
    assert result.stop_reason == "passed"
    assert len(result.rounds) == 2
    assert [r.total_score for r in result.rounds] == [72.0, 90.0]
    assert result.best_round_no == 2
    assert result.stop_label == "已达标：通过全部门禁"


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_harness_v2_stopped_not_passing_on_plateau(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    from src.agents.poster_planner import generate_blueprint
    blueprint = generate_blueprint(doc, analysis)
    initial = _render_initial(tmp_path)

    reviews = [_review_with_verdict(t) for t in (60.0, 61.0, 61.0)]
    with patch("src.agents.poster_harness.review_rendered_poster_v2", side_effect=reviews):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=_config())

    assert result.passed is False
    assert result.stop_reason == "stopped_not_passing"
    assert "未达标" in result.stop_label
    assert len(result.rounds) == 3
    # 最优轮保留
    assert result.best_round_no == 2


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_harness_v2_hard_errors_block_pass(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    from src.agents.poster_planner import generate_blueprint
    blueprint = generate_blueprint(doc, analysis)
    initial = _render_initial(tmp_path)

    review = _review_with_verdict(92.0, hard_errors=1)
    review.hard_failures = ["broken_images"]
    review.issues.append(PosterComment(issue="Broken image", severity="error", target="", action="replace_figure"))

    with patch("src.agents.poster_harness.review_rendered_poster_v2", side_effect=[review, review]):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path,
                                    config=_config(max_rounds=2, plateau_rounds=5))

    assert result.passed is False
    assert result.best_total == 92.0


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_harness_v2_vision_unavailable_stops_cleanly(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    from src.agents.poster_planner import generate_blueprint
    blueprint = generate_blueprint(doc, analysis)
    initial = _render_initial(tmp_path)

    with patch("src.agents.poster_harness.review_rendered_poster_v2", return_value=None):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=_config())

    assert result.passed is False
    assert result.stop_reason == "vision_unavailable"
    assert result.rounds == []


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_harness_v2_records_artifacts(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    from src.agents.poster_planner import generate_blueprint
    blueprint = generate_blueprint(doc, analysis)
    initial = _render_initial(tmp_path)

    review = _review_with_verdict(66.0)
    review.artifact_paths = {
        "full_png": str(tmp_path / "poster_full.png"),
        "grid_png": str(tmp_path / "grid.png"),
        "diff_png": str(tmp_path / "diff.png"),
        "sections": '{"sec-motivation": "sections/sec-motivation.png"}',
        "figures": '{"fig_00": "figures/fig_00.png"}',
    }
    with patch("src.agents.poster_harness.review_rendered_poster_v2", side_effect=[review, _review_with_verdict(70.0)]):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=_config(max_rounds=2))

    round1 = result.rounds[0]
    assert round1.grid_png
    assert round1.section_crops.get("sec-motivation")
    assert round1.figure_crops.get("fig_00")
    report = __import__("json").loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["total_scores"] == [66.0, 70.0]
    assert report["rounds"][0]["verdict"]["computed"] is True


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=True)
@patch("src.agents.poster_harness.LLMClient.chat")
def test_harness_v2_blank_section_creates_svg_supplement(mock_chat, _mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    from src.agents.poster_planner import generate_blueprint
    blueprint = generate_blueprint(doc, analysis)
    initial = _render_initial(tmp_path)

    motivation = next(sec for sec in blueprint.sections if sec.type == "motivation")
    motivation.content_md = ""

    mock_chat.return_value = """<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 320 320\">
<rect width=\"320\" height=\"320\" fill=\"white\"/>
<circle cx=\"160\" cy=\"160\" r=\"72\" fill=\"#16324f\"/>
<text x=\"160\" y=\"172\" text-anchor=\"middle\" font-size=\"80\" fill=\"white\">◎</text>
</svg>"""

    review = _review_with_verdict(60.0)
    review.issues.append(PosterComment(issue="Blank area in motivation", severity="warning", target="sec-motivation", action="supplement"))
    review.deterministic_checks = {
        "section_blank_reports": [
            {"section_id": "sec-motivation", "blank_ratio": 0.72, "content_ratio": 0.28, "width": 0, "height": 0}
        ]
    }

    normal_config = HarnessConfig(threshold=8, max_rounds=2, zoom_crops=False, enable_qa_eval=False)

    with patch("src.agents.poster_harness.review_rendered_poster", side_effect=[review, _review_with_verdict(70.0)]):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=normal_config)

    round1 = result.rounds[0]
    assert "supplement sec-motivation (svg)" in round1.applied_actions
    figure_dir = tmp_path / "harness" / "round_1" / "figures"
    supplement_dir = tmp_path / "harness" / "round_1" / "supplement"
    svg_files = list(figure_dir.glob("*.svg"))
    supplement_files = list(supplement_dir.glob("*.svg"))
    assert svg_files, "expected generated SVG supplement asset"
    assert supplement_files, "expected raw supplement asset copy"
    assert svg_files[0].name == supplement_files[0].name
    assert "<svg" in supplement_files[0].read_text(encoding="utf-8")
