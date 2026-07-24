from __future__ import annotations

from unittest.mock import patch

from src.agents.poster_v2 import build_layout_tree, generate_paperquiz_questions
from src.agents.poster_v2 import _normalize_severity, _normalize_quality_score
from src.agents.poster_v2 import _parse_layout_constraints
from src.schemas.analysis import PaperAnalysis, Contribution, ExperimentSummary
from src.schemas.paper import Author, Figure, PaperDocument, Section


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


def test_build_layout_tree_fallback_creates_nodes():
    doc = _make_doc()
    analysis = _make_analysis()
    with patch("src.agents.poster_v2.LLMClient.is_configured", return_value=False):
        tree = build_layout_tree(doc, analysis, use_gpt5=True)
    assert tree.paper_id == doc.paper_id
    assert tree.nodes
    assert any(node.node_id == "sec-001" for node in tree.nodes)


def test_generate_paperquiz_questions_cover_core_content():
    questions = generate_paperquiz_questions(_make_doc(), _make_analysis())
    assert questions
    assert any("problem" in q.question.lower() for q in questions)
    assert any("result" in q.question.lower() for q in questions)


def test_normalize_severity_maps_model_aliases():
    assert _normalize_severity("high") == "error"
    assert _normalize_severity("medium") == "warning"
    assert _normalize_severity("low") == "info"
    assert _normalize_severity("warning") == "warning"


def test_normalize_quality_score_maps_0_to_10():
    assert _normalize_quality_score(64) == 6
    assert _normalize_quality_score(9) == 9
    assert _normalize_quality_score(0) == 0


def test_parse_layout_constraints_accepts_list():
    constraints = _parse_layout_constraints(["Full poster canvas", "Keep headers across columns"], priority=2)
    assert constraints.priority == 2
    assert constraints.min_ratio == 0.08
