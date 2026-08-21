from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.poster_harness import (
    _apply_css_patches,
    _apply_feedback,
    _css_for_section,
    _heuristic_density_issues,
    _match_section,
    _normalize_dimension_scores,
    _normalize_issue,
    _normalize_quality_score,
    _normalize_severity,
    _rewrite_section,
    _analyze_blank_regions,
    _normalize_core_blank_review,
    _size_supplement_svg,
    _supplement_overlay_html,
    _should_stop,
    evaluate_poster_visual_qa,
    run_poster_harness,
)
from src.agents.content_policy import count_words, section_budget
from src.agents.poster_harness import review_rendered_poster
from src.schemas.analysis import Contribution, ExperimentSummary, PaperAnalysis
from src.schemas.paper import Author, Figure, PaperDocument, Section
from src.schemas.poster import PosterBlueprint
from src.schemas.poster_harness import HarnessConfig
from src.schemas.poster_v2 import PosterComment, PosterReview


def _make_doc() -> PaperDocument:
    return PaperDocument(
        paper_id="test-999",
        arxiv_id="9999.99999",
        title="Test Paper",
        authors=[Author(name="Alice")],
        abstract="Abstract.",
        sections=[Section(section_id="sec-001", title="Intro", level=1, text="Intro", raw_latex="Intro")],
        figures=[Figure(figure_id="fig-001", caption="Framework overview", section_id="sec-001")],
        raw_markdown="# Test",
    )


def _make_analysis() -> PaperAnalysis:
    return PaperAnalysis(
        paper_id="test-999",
        arxiv_id="9999.99999",
        title_zh="",
        problem_statement="We solve classification.",
        contributions=[Contribution(text="A better backbone", category="method")],
        method_overview="We use a new architecture.",
        key_formulas=[],
        key_figures=[],
        experiments=ExperimentSummary(main_results="99% accuracy", takeaways=["It works well"]),
        conclusion="Done.",
        full_analysis_md="# Analysis",
    )


def _make_blueprint() -> PosterBlueprint:
    from src.agents.poster_planner import generate_blueprint

    return generate_blueprint(_make_doc(), _make_analysis())


def _review(score: int, needs_improvement: bool = False, issues: list[PosterComment] | None = None) -> PosterReview:
    return PosterReview(
        quality_score=score,
        needs_improvement=needs_improvement,
        issues=issues or [],
        summary="ok",
        dimension_scores={"layout": float(score), "typography": float(score)},
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_quality_score_ranges():
    assert _normalize_quality_score(9) == 9
    assert _normalize_quality_score(8.4) == 8
    assert _normalize_quality_score(85) == 8  # 0-100 scale -> 8.5 -> banker's rounding
    assert _normalize_quality_score("7") == 7
    assert _normalize_quality_score(None) == 0
    assert _normalize_quality_score(-3) == 0
    assert _normalize_quality_score(200) == 10


def test_normalize_severity_mapping():
    assert _normalize_severity("error") == "error"
    assert _normalize_severity("critical") == "error"
    assert _normalize_severity("medium") == "warning"
    assert _normalize_severity("minor") == "info"
    assert _normalize_severity("???") == "warning"


def test_normalize_issue():
    issue = _normalize_issue({
        "description": "Text overflows",
        "severity": "high",
        "target": "sec-motivation",
        "suggestion": "Shorten",
        "action": "rewrite",
    })
    assert issue is not None
    assert issue.severity == "error"
    assert issue.target == "sec-motivation"
    assert issue.action == "rewrite"

    # 非法 action 会被归一化为 rewrite（只要 description 存在）
    issue2 = _normalize_issue({"action": "bogus_action", "description": "x"})
    assert issue2 is not None
    assert issue2.action == "rewrite"
    assert _normalize_issue({"description": ""}) is None
    assert _normalize_issue(None) is None


def test_normalize_dimension_scores():
    scores = _normalize_dimension_scores({"layout": 7, "typography": "8", "figures": None})
    assert scores["layout"] == 7.0
    assert scores["typography"] == 8.0
    assert scores["figures"] == 0.0


# ---------------------------------------------------------------------------
# review_rendered_poster (VLM mocked)
# ---------------------------------------------------------------------------


def test_review_rendered_poster_parses_vlm_json(tmp_path):
    html_path = tmp_path / "poster.html"
    html_path.write_text("<html><head></head><body>poster</body></html>", encoding="utf-8")
    blueprint = _make_blueprint()

    fake_review = {
        "quality_score": 6,
        "dimension_scores": {"layout": 6, "typography": 5},
        "needs_improvement": True,
        "summary": "Dense text.",
        "issues": [
            {
                "description": "Text overflow in motivation",
                "severity": "warning",
                "target": "sec-motivation",
                "suggestion": "Shorten bullet points",
                "action": "resize",
            }
        ],
    }

    def _fake_capture(html_path, png_path, selectors, width=1200, height=1697):
        png_path.write_bytes(b"fake-png")
        return {}

    captured_images: list[tuple[str, str]] = []

    def _fake_vlm(system, images, user_text="", model=None):
        captured_images.extend(images)
        return fake_review

    with patch("src.agents.poster_harness.capture_poster_full_and_sections", side_effect=_fake_capture), \
         patch("src.agents.poster_harness.downscale_image", side_effect=lambda p, max_width=1400: p), \
         patch("src.agents.poster_harness.inspect_rendered_poster", return_value={"available": True, "hard_failures": []}), \
         patch("src.agents.poster_harness.multimodal_analyze_labeled", side_effect=_fake_vlm):
        review = review_rendered_poster(html_path, tmp_path, HarnessConfig(), blueprint)

    assert review is not None
    assert review.quality_score == 6
    assert review.needs_improvement is True
    assert len(review.issues) == 1
    assert review.issues[0].target == "sec-motivation"
    assert review.issues[0].action == "resize"
    assert review.dimension_scores["layout"] == 6.0
    # Regression test for the former (label, path) ordering bug: the actual
    # local PNG path must be first so the multimodal client can open it.
    assert captured_images == [(str(tmp_path / "poster.png"), "poster (full view)")]


def test_review_rendered_poster_returns_none_when_vlm_fails(tmp_path):
    html_path = tmp_path / "poster.html"
    html_path.write_text("<html><head></head><body>poster</body></html>", encoding="utf-8")
    blueprint = _make_blueprint()

    def _fake_capture(html_path, png_path, selectors, width=1200, height=1697):
        png_path.write_bytes(b"fake-png")
        return {}

    with patch("src.agents.poster_harness.capture_poster_full_and_sections", side_effect=_fake_capture), \
         patch("src.agents.poster_harness.downscale_image", side_effect=lambda p, max_width=1400: p), \
         patch("src.agents.poster_harness.inspect_rendered_poster", return_value={"available": True, "hard_failures": []}), \
         patch("src.agents.poster_harness.multimodal_analyze_labeled", return_value=None):
        review = review_rendered_poster(html_path, tmp_path, HarnessConfig(), blueprint)
    assert review is None


# ---------------------------------------------------------------------------
# Feedback application
# ---------------------------------------------------------------------------


def test_match_section_by_type_title_and_id():
    blueprint = _make_blueprint()
    assert _match_section("sec-motivation", blueprint) is not None
    assert _match_section("motivation", blueprint) is not None
    assert _match_section("Motivation", blueprint) is not None
    assert _match_section("nope", blueprint) is None
    assert _match_section("", blueprint) is None


def test_apply_feedback_density_css_patch():
    blueprint = _make_blueprint()
    css_patches: list[str] = []
    review = _review(6, needs_improvement=True, issues=[
        PosterComment(issue="Text is too dense and overflows", severity="warning",
                      target="sec-motivation", suggestion="", action="resize"),
    ])
    applied = _apply_feedback(blueprint, review, llm=None, css_patches=css_patches)
    assert applied and any("density" in a for a in applied)
    assert any("#sec-motivation" in p for p in css_patches)


def test_apply_feedback_rewrite_trims_without_llm():
    blueprint = _make_blueprint()
    sec = next(s for s in blueprint.sections if s.section_id == "sec-motivation")
    long_text = " ".join(["word"] * 200)
    sec.content_md = long_text
    css_patches: list[str] = []
    review = _review(5, needs_improvement=True, issues=[
        PosterComment(issue="Too much text", severity="warning",
                      target="sec-motivation", suggestion="", action="rewrite"),
    ])
    _apply_feedback(blueprint, review, llm=None, css_patches=css_patches)
    assert len((sec.content_md or "").split()) <= 90


def test_apply_feedback_replace_figure_moves_placement():
    blueprint = _make_blueprint()
    css_patches: list[str] = []
    review = _review(6, needs_improvement=True, issues=[
        PosterComment(issue="Figures should be closer to method", severity="info",
                      target="sec-main-method", suggestion="", action="replace_figure"),
    ])
    _apply_feedback(blueprint, review, llm=None, css_patches=css_patches)
    moved = any(fp.section_id == "sec-main-method" for fp in blueprint.figure_placements)
    assert moved


def test_apply_feedback_remove_figure():
    blueprint = _make_blueprint()
    css_patches: list[str] = []
    target_sec = next(s.section_id for s in blueprint.sections if s.type == "method_overview")
    review = _review(6, needs_improvement=True, issues=[
        PosterComment(issue="Figure is broken/irrelevant", severity="error",
                      target=target_sec, suggestion="", action="remove"),
    ])
    _apply_feedback(blueprint, review, llm=None, css_patches=css_patches)
    assert all(fp.section_id != target_sec for fp in blueprint.figure_placements)


def test_apply_feedback_supplement_injects_visual_html():
    blueprint = _make_blueprint()
    sec = next(s for s in blueprint.sections if s.section_id == "sec-motivation")
    review = _review(6, needs_improvement=True, issues=[
        PosterComment(issue="Large blank area", severity="warning",
                      target="sec-motivation", suggestion="fill", action="supplement"),
    ])
    review.deterministic_checks = {
        "section_blank_reports": [
            {"section_id": "sec-motivation", "blank_ratio": 0.72, "content_ratio": 0.28, "width": 0, "height": 0}
        ]
    }
    css_patches: list[str] = []
    applied = _apply_feedback(blueprint, review, llm=None, css_patches=css_patches)
    assert any("supplement sec-motivation" in a for a in applied)
    assert "Blank ratio" not in sec.supplement_html
    assert "figure-card" in sec.supplement_html or sec.supplement_html == ""


def test_blank_png_ratio_probe_detects_white_space(tmp_path):
    from PIL import Image

    from src.agents.poster_harness import _measure_png_blank_ratio

    path = tmp_path / "blank.png"
    img = Image.new("RGB", (20, 20), (255, 255, 255))
    for x in range(6):
        for y in range(6):
            img.putpixel((x, y), (0, 0, 0))
    img.save(path)
    ratio = _measure_png_blank_ratio(path)
    assert ratio is not None
    assert 0.9 <= ratio <= 0.95


def test_blank_region_analysis_returns_bbox_and_rejects_low_variance_false_positive(tmp_path):
    from PIL import Image, ImageDraw

    path = tmp_path / "section.png"
    image = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(image)
    for y in range(40):
        shade = 240 if y % 2 else 255
        draw.line((0, y, 99, y), fill=(shade, shade, shade))
    draw.rectangle((0, 60, 99, 99), fill="white")
    draw.rectangle((15, 68, 85, 72), fill=(0, 0, 0))
    image.save(path)

    regions = _analyze_blank_regions(path)

    assert regions
    largest = max(regions, key=lambda item: item["area_pixels"])
    assert largest["x"] == 0
    assert largest["y"] >= 60
    assert largest["width"] == 100
    assert largest["height"] <= 40
    assert largest["gray_variance"] < 10
    assert all(region["y"] >= 60 for region in regions)


def test_svg_supplement_uses_detected_region_geometry():
    from src.agents.poster_harness import BlankRegionCandidate

    candidate = BlankRegionCandidate(
        section_id="sec-motivation",
        section_type="motivation",
        section_title="Motivation",
        blank_ratio=0.4,
        content_ratio=0.6,
        width=100,
        height=100,
        text_words=20,
        figure_count=0,
        has_figures=False,
        local_context="context",
        nearby_context="",
        global_context="",
        blank_regions=[{"x": 0, "y": 60, "width": 100, "height": 40, "area_pixels": 4000}],
    )
    svg = _size_supplement_svg('<svg viewBox="0 0 10 10"></svg>', candidate)
    overlay = _supplement_overlay_html("figures/sec-motivation_supplement.svg", candidate, "Motivation")

    assert 'width="100"' in svg
    assert 'height="40"' in svg
    assert 'left:0px;top:6px;width:64px;height:24px' in overlay


def test_core_blank_vlm_requires_explicit_positive_review():
    from src.agents.poster_harness import SectionBlankReport, _should_supplement_report

    sec = next(s for s in _make_blueprint().sections if s.type == "main_method")
    report = SectionBlankReport(
        section_id=sec.section_id,
        section_type=sec.type,
        section_title=sec.title,
        blank_ratio=0.8,
        content_ratio=0.2,
        width=100,
        height=100,
        text_words=50,
        figure_count=0,
        has_figures=False,
        core_blank_review={"has_invalid_blank": False},
    )
    assert _should_supplement_report(report, sec) is False

    report.core_blank_review = {
        "has_invalid_blank": True,
        "location": "bottom_right",
        "confidence": 0.9,
    }
    assert _should_supplement_report(report, sec) is True


def test_core_blank_vlm_normalizes_location_and_region_hint():
    review = _normalize_core_blank_review({
        "has_invalid_blank": True,
        "location": "bottom_right",
        "description": "Almost no text or graphics in the lower-right area.",
        "confidence": 0.92,
        "region_hint": {"x": 0.55, "y": 0.58, "width": 0.5, "height": 0.5},
    })
    assert review is not None
    assert review.has_invalid_blank is True
    assert review.location == "bottom_right"
    assert review.region_hint == {"x": 0.55, "y": 0.58, "width": 0.45, "height": 0.42}

    negative = _normalize_core_blank_review({
        "has_invalid_blank": True,
        "location": "bottom_right",
        "confidence": 0.2,
    })
    assert negative is not None
    assert negative.has_invalid_blank is False

    chinese = _normalize_core_blank_review({
        "has_invalid_blank": True,
        "location": "右下角",
        "confidence": 0.9,
    })
    assert chinese is not None
    assert chinese.location == "bottom_right"


# ---------------------------------------------------------------------------
# Loop control
# ---------------------------------------------------------------------------


def test_should_stop():
    assert _should_stop(_review(9, needs_improvement=False), 1, HarnessConfig(threshold=8), [9]) == "passed"
    assert _should_stop(_review(6, needs_improvement=True), 5, HarnessConfig(max_rounds=5), [6]) == "max_rounds"
    assert _should_stop(_review(6, needs_improvement=True), 3, HarnessConfig(max_rounds=5), [6, 6, 6]) == "plateau"
    assert _should_stop(_review(7, needs_improvement=True), 3, HarnessConfig(max_rounds=5), [6, 7, 8]) is None


def test_should_stop_rejects_high_score_with_hard_failure():
    review = _review(10, needs_improvement=False)
    review.hard_failures = ["broken_images"]
    assert _should_stop(review, 1, HarnessConfig(threshold=9), [10]) is None


def test_image_grounded_qa_uses_poster_png_only(tmp_path):
    poster_png = tmp_path / "poster.png"
    poster_png.write_bytes(b"fake-image")
    seen: list[tuple[str, str]] = []

    def _fake_vlm(system, images, user_text="", model=None):
        seen.extend(images)
        return {
            "answers": [
                {"question_id": "q-problem", "answer": "We solve classification."},
                {"question_id": "q-method", "answer": "We use a new architecture."},
                {"question_id": "q-contrib-1", "answer": "A better backbone"},
                {"question_id": "q-result", "answer": "99% accuracy"},
                {"question_id": "q-takeaway", "answer": "It works well"},
            ]
        }

    with patch("src.agents.poster_harness.multimodal_analyze_labeled", side_effect=_fake_vlm):
        qa = evaluate_poster_visual_qa(_make_doc(), _make_analysis(), poster_png)

    assert qa is not None
    assert qa.accuracy == 1.0
    assert seen == [(str(poster_png), "final candidate poster")]


def test_apply_css_patches_injects_before_head_close():
    html = "<html><head><title>t</title></head><body>x</body></html>"
    out = _apply_css_patches(html, ["#sec-a { font-size: 12px !important; }"])
    assert 'id="harness-css-patch"' in out
    assert out.index("</head>") > out.index("harness-css-patch")


# ---------------------------------------------------------------------------
# 字数预算 / 密度控制（参考 poster-defaults.md 250-500 词标准）
# ---------------------------------------------------------------------------


def test_heuristic_density_issues_flag_wordy_sections():
    blueprint = _make_blueprint()
    sec = next(s for s in blueprint.sections if s.section_id == "sec-motivation")
    sec.content_md = " ".join(["word"] * 300)
    issues = _heuristic_density_issues(blueprint)
    condense_targets = {i.target for i in issues if i.action == "condense"}
    assert "sec-motivation" in condense_targets


def test_apply_feedback_condense_trims_to_budget():
    blueprint = _make_blueprint()
    sec = next(s for s in blueprint.sections if s.section_id == "sec-highlights")
    sec.content_md = " ".join(["word"] * 300)
    css_patches: list[str] = []
    review = _review(5, needs_improvement=True, issues=[
        PosterComment(issue="Text too dense", severity="warning",
                      target="sec-highlights", suggestion="", action="condense"),
    ])
    _apply_feedback(blueprint, review, llm=None, css_patches=css_patches)
    assert count_words(sec.content_md) <= section_budget("highlights")


def test_rewrite_section_prompt_includes_word_target():
    blueprint = _make_blueprint()
    sec = next(s for s in blueprint.sections if s.section_id == "sec-motivation")
    recorded: dict = {}

    class _FakeLLM:
        def chat(self, system, user):
            recorded["user"] = user
            return "Shorter motivation text with all facts."

    ok = _rewrite_section(_FakeLLM(), sec, "too dense", "shorten", max_words=40)
    assert ok
    assert "at most 40 words" in recorded["user"]
    assert count_words(sec.content_md) <= 40


# ---------------------------------------------------------------------------
# run_poster_harness loop
# ---------------------------------------------------------------------------


def _config(**kwargs) -> HarnessConfig:
    defaults = dict(threshold=8, max_rounds=5, zoom_crops=False, enable_qa_eval=False)
    defaults.update(kwargs)
    return HarnessConfig(**defaults)


def _mock_reviews(scores):
    """Return a side-effect list of PosterReview, one per round."""
    reviews = []
    for score in scores:
        reviews.append(_review(score, needs_improvement=score < 8))
    return reviews


def _render_initial(output_dir: Path, blueprint) -> Path:
    from src.renderers.html_renderer import HtmlPosterRenderer

    html_path = output_dir / "poster_draft.html"
    HtmlPosterRenderer().render_to_file(blueprint, _make_doc(), html_path)
    return html_path


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_run_poster_harness_passes_after_improvements(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    blueprint = _make_blueprint()
    initial = _render_initial(tmp_path, blueprint)

    with patch("src.agents.poster_harness.review_rendered_poster", side_effect=_mock_reviews([6, 8])):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=_config())

    assert result.passed is True
    assert result.stop_reason == "passed"
    assert len(result.rounds) == 2
    assert [r.quality_score for r in result.rounds] == [6, 8]
    assert result.best_round_no == 2
    # 产物落盘
    assert (tmp_path / "harness" / "round_1" / "poster.html").exists()
    assert (tmp_path / "harness" / "round_2" / "review.json").exists()
    assert (tmp_path / "harness_report.json").exists()
    assert (tmp_path / "poster_final.html").exists()


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_run_poster_harness_plateau_early_stop(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    blueprint = _make_blueprint()
    initial = _render_initial(tmp_path, blueprint)

    with patch("src.agents.poster_harness.review_rendered_poster", side_effect=_mock_reviews([6, 6, 6, 6])):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=_config())

    assert result.passed is False
    assert result.stop_reason == "plateau"
    assert len(result.rounds) == 3
    assert result.best_round_no == 1  # 保留最优轮
    assert result.best_score == 6


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_run_poster_harness_max_rounds_bound(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    blueprint = _make_blueprint()
    initial = _render_initial(tmp_path, blueprint)

    with patch("src.agents.poster_harness.review_rendered_poster", side_effect=_mock_reviews([5, 5])):
        result = run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=_config(max_rounds=2))

    assert result.passed is False
    assert result.stop_reason == "max_rounds"
    assert len(result.rounds) == 2


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_run_poster_harness_fallback_when_vision_unavailable(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    blueprint = _make_blueprint()
    initial = _render_initial(tmp_path, blueprint)
    calls: list[tuple[Path, Path]] = []

    def _fallback(old: Path, new: Path):
        calls.append((old, new))
        new.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")

    with patch("src.agents.poster_harness.review_rendered_poster", return_value=None):
        result = run_poster_harness(
            doc, analysis, blueprint, initial, tmp_path,
            config=_config(), fallback_optimizer=_fallback,
        )

    assert result.fallback is True
    assert result.stop_reason == "vision_unavailable"
    assert result.rounds == []
    assert len(calls) == 1
    assert Path(result.final_html).exists()


@patch("src.agents.poster_harness.LLMClient.is_configured", return_value=False)
def test_run_poster_harness_on_round_callback(mock_cfg, tmp_path):
    doc, analysis = _make_doc(), _make_analysis()
    blueprint = _make_blueprint()
    initial = _render_initial(tmp_path, blueprint)
    seen: list[tuple[int, int, int, bool, str]] = []

    def _on_round(r, total, score, needs, summary):
        seen.append((r, total, score, needs, summary))

    with patch("src.agents.poster_harness.review_rendered_poster", side_effect=_mock_reviews([6, 8])):
        run_poster_harness(doc, analysis, blueprint, initial, tmp_path, config=_config(), on_round=_on_round)

    assert [s[0] for s in seen] == [1, 2]
    assert [s[2] for s in seen] == [6, 8]
