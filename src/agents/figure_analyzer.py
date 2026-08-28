from __future__ import annotations

from src.agents.visual_planner import plan_visual_assets
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.visual import VisualAssetPlan


class FigureAnalyzer:
    """Rank figures, tables, and formulas using shared deterministic signals."""

    def analyze(self, doc: PaperDocument, analysis: PaperAnalysis) -> VisualAssetPlan:
        return plan_visual_assets(doc, analysis)


def analyze_figures(doc: PaperDocument, analysis: PaperAnalysis) -> VisualAssetPlan:
    return FigureAnalyzer().analyze(doc, analysis)
