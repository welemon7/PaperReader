from __future__ import annotations

from unittest.mock import patch

from src.agents.poster_v2 import build_layout_tree
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
    tree = build_layout_tree(doc, analysis)
    assert tree.paper_id == doc.paper_id
    assert tree.nodes
    assert any(node.node_id == "sec-001" for node in tree.nodes)


