from __future__ import annotations

from pathlib import Path

from src.agents.content_importance import analyze_content_importance
from src.agents.layout_solver import solve_layout
from src.agents.poster_planner import generate_blueprint
from src.agents.poster_story_planner import plan_poster_story
from src.renderers.html_renderer import HtmlPosterRenderer
from src.schemas.analysis import Contribution, ExperimentSummary, ImportanceItem, PaperAnalysis
from src.schemas.paper import Author, PaperDocument, Section
from src.schemas.poster_v2 import LayoutConstraints, LayoutNode


def _doc() -> PaperDocument:
    return PaperDocument(
        paper_id="opt-1", arxiv_id="opt-1", title="Optimized Poster",
        authors=[Author(name="Alice")],
        sections=[Section(section_id="s1", title="Method", level=1, text="method", raw_latex="method")],
        raw_markdown="# paper",
    )


def _analysis() -> PaperAnalysis:
    return PaperAnalysis(
        paper_id="opt-1", arxiv_id="opt-1", problem_statement="An expensive baseline limits deployment.",
        contributions=[Contribution(text="We introduce an efficient architecture.", category="method")],
        method_overview="Our pipeline improves benchmark performance.",
        experiments=ExperimentSummary(main_results="The method improves accuracy on benchmark results.", metrics=["Accuracy"]),
        conclusion="The method is effective.",
    )


def test_importance_is_evidence_backed_and_bounded():
    result = analyze_content_importance(_analysis(), _doc())
    assert 0.0 <= result.main_message.importance <= 1.0
    assert result.main_message.evidence
    assert result.main_message.score_breakdown["centrality"] > 0
    assert result.main_message.confidence > 0


def test_story_plan_normalizes_area_and_exposes_visual_decisions():
    analysis = _analysis()
    analysis.content_importance = analyze_content_importance(analysis, _doc())
    plan = plan_poster_story(_doc(), analysis)
    assert len(plan.beats) >= 5
    assert round(sum(beat.target_area_ratio for beat in plan.beats), 4) == 1.0
    assert all(beat.preferred_composition for beat in plan.beats)
    assert sum(beat.visual_priority == "P0" for beat in plan.beats) <= 3


def test_layout_solver_places_nodes_without_overlap():
    nodes = [
        LayoutNode(
            node_id=f"n-{index}", node_type="text", reading_order=index,
            importance=1.0 - index * 0.1, visual_weight=1.0 - index * 0.1,
            constraints=LayoutConstraints(priority=4 if index == 0 else 2),
            target_area_ratio=0.25 if index == 0 else 0.1,
        )
        for index in range(6)
    ]
    results = solve_layout(nodes)
    assert len(results) == len(nodes)
    assert results[0].col_span >= results[-1].col_span
    for left_index, left in enumerate(results):
        for right in results[left_index + 1:]:
            assert not (
                left.col < right.col + right.col_span
                and right.col < left.col + left.col_span
                and left.row < right.row + right.row_span
                and right.row < left.row + left.row_span
            )


def test_legacy_importance_json_still_loads():
    item = ImportanceItem.model_validate({"text": "legacy", "importance": 0.5, "role": "supporting"})
    assert item.evidence == []
    assert item.score_breakdown == {}


def test_inline_figure_map_leaves_unreadable_assets_unchanged(tmp_path: Path):
    result = HtmlPosterRenderer._inline_figure_map(
        {"s1": [{"src": "figures/missing.png", "figure_id": "f1"}]}, tmp_path
    )
    assert result["s1"][0]["src"] == "figures/missing.png"


def test_production_blueprint_uses_semantic_geometry():
    analysis = _analysis()
    analysis.content_importance = analyze_content_importance(analysis, _doc())
    blueprint = generate_blueprint(_doc(), analysis)
    sections = {section.section_id: section for section in blueprint.sections}
    highlights = sections["sec-highlights"]
    assert highlights.grid_col >= 1
    assert highlights.grid_col_span >= 3
    assert highlights.grid_col + highlights.grid_col_span - 1 <= 12
    assert all(section.grid_col_span >= 1 for section in blueprint.sections)
