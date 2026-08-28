from src.agents.visual_planner import VisualAssetPlanner, plan_visual_assets
from src.schemas.analysis import Contribution, ExperimentSummary, KeyFigure, KeyFormula, PaperAnalysis
from src.schemas.paper import Figure, PaperDocument, Table


def _analysis() -> PaperAnalysis:
    return PaperAnalysis(
        paper_id="visual-test", arxiv_id="v-1",
        problem_statement="The backbone is expensive.",
        method_overview="We prune redundant components.",
        contributions=[Contribution(text="Adaptive pruning", category="method")],
        key_figures=[KeyFigure(figure_id="fig-a", caption="Framework overview", role="architecture")],
        key_formulas=[KeyFormula(formula_id="formula-a", latex="x = y", semantic_desc="Importance score")],
        experiments=ExperimentSummary(main_results="40% fewer FLOPs."),
    )


def _doc() -> PaperDocument:
    return PaperDocument(
        paper_id="visual-test", arxiv_id="v-1", title="Visual Planner",
        figures=[
            Figure(figure_id="fig-b", caption="Framework overview and architecture", section_id="s1"),
            Figure(figure_id="fig-c", caption="Accuracy benchmark comparison", section_id="s2"),
        ],
        tables=[Table(table_id="table-a", caption="Benchmark results", headers=["Method", "Accuracy"], rows=[["Ours", "99"]])],
    )


def test_figure_score_uses_all_four_factors():
    plan = plan_visual_assets(_doc(), _analysis())
    decision = next(item for item in plan.figure_decisions if item.asset_id == "fig-a")
    assert decision.score == round(
        decision.relevance * decision.readability * decision.visual_value * decision.evidence_strength, 4
    )
    assert decision.score > 0


def test_redundant_figures_receive_simplification_action():
    plan = VisualAssetPlanner(redundancy_threshold=0.5).plan(_doc(), _analysis())
    decisions = {item.asset_id: item for item in plan.figure_decisions}
    assert decisions["fig-b"].action in {"remove", "annotate"}
    assert decisions["fig-b"].redundancy_group


def test_plan_limits_assets_and_selects_evidence():
    plan = plan_visual_assets(_doc(), _analysis())
    assert len(plan.selected_figure_ids) <= 4
    assert len(plan.selected_chart_ids) <= 2
    assert len(plan.selected_formula_ids) <= 4
    assert "table-a" in plan.selected_chart_ids
