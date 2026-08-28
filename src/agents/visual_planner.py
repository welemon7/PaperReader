from __future__ import annotations

import re
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.visual import VisualAssetDecision, VisualAssetPlan

__all__ = ["VisualAssetDecision", "VisualAssetPlan", "VisualAssetPlanner", "plan_visual_assets"]


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()
        if len(token) > 2
    }


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _figure_role_text(fig) -> str:
    return f"{getattr(fig, 'caption', '')} {getattr(fig, 'role', '')} {getattr(fig, 'section_id', '')}"


class VisualAssetPlanner:
    """Rank and simplify visual assets before the layout planner sees them."""

    def __init__(self, redundancy_threshold: float = 0.72) -> None:
        self.redundancy_threshold = redundancy_threshold

    def plan(self, doc: PaperDocument, analysis: PaperAnalysis) -> VisualAssetPlan:
        figures = self._unique_figures(list(analysis.key_figures) + list(doc.figures))
        figure_decisions = [self._score_figure(fig, analysis) for fig in figures]
        self._mark_redundancy(figure_decisions, figures)
        figure_decisions.sort(key=lambda item: (-item.score, item.asset_id))

        # Keep a compact visual vocabulary. The existing placer still decides
        # the exact hero/core slots, while this list decides what may enter it.
        selected_figures = [
            item.asset_id for item in figure_decisions
            if item.action == "keep"
        ][:4]

        charts = [self._score_table(table, analysis) for table in doc.tables]
        charts.sort(key=lambda item: (-item.score, item.asset_id))
        selected_charts = [item.asset_id for item in charts[:2] if item.action == "keep"]

        formulas = [self._score_formula(formula, analysis) for formula in analysis.key_formulas]
        formulas.sort(key=lambda item: (-item.score, item.asset_id))
        selected_formulas = [item.asset_id for item in formulas[:4] if item.action == "keep"]

        return VisualAssetPlan(
            figure_decisions=figure_decisions,
            chart_decisions=charts,
            formula_decisions=formulas,
            selected_figure_ids=selected_figures,
            selected_chart_ids=selected_charts,
            selected_formula_ids=selected_formulas,
            redundancy_threshold=self.redundancy_threshold,
        )

    @staticmethod
    def _unique_figures(figures: list) -> list:
        seen: set[str] = set()
        result = []
        for fig in figures:
            asset_id = getattr(fig, "figure_id", "")
            if asset_id and asset_id in seen:
                continue
            if asset_id:
                seen.add(asset_id)
            result.append(fig)
        return result

    def _score_figure(self, fig, analysis: PaperAnalysis) -> VisualAssetDecision:
        text = _figure_role_text(fig).lower()
        is_method = any(word in text for word in ("framework", "overview", "architecture", "pipeline", "method"))
        is_result = any(word in text for word in ("result", "comparison", "benchmark", "accuracy", "ablation", "performance", "metric"))
        relevance = 0.9 if is_method or is_result else 0.48
        readability = 0.85 if len(getattr(fig, "caption", "") or "") <= 180 else 0.6
        visual_value = 0.9 if is_method else (0.82 if is_result else 0.5)
        evidence = 0.9 if is_result else (0.68 if is_method else 0.4)
        score = relevance * readability * visual_value * evidence
        role = "method" if is_method else ("evidence" if is_result else "supporting")
        return VisualAssetDecision(
            asset_id=getattr(fig, "figure_id", ""), kind="figure", score=round(score, 4),
            relevance=relevance, readability=readability, visual_value=visual_value,
            evidence_strength=evidence, target_section_id="sec-method-overview" if is_method else "sec-main-method",
            message=(getattr(fig, "caption", "") or "").strip(),
            recommended_width=0.72 if is_method or is_result else 0.42,
            reason=f"{role} visual with score={score:.3f}",
        )

    def _score_table(self, table, analysis: PaperAnalysis) -> VisualAssetDecision:
        rows = len(getattr(table, "rows", []) or [])
        cols = len(getattr(table, "headers", []) or [])
        relevance = 0.9 if any(word in (getattr(table, "caption", "") or "").lower() for word in ("result", "benchmark", "comparison", "performance")) else 0.58
        readability = 0.9 if rows <= 8 and cols <= 8 else 0.55
        visual_value = 0.75 if rows and cols else 0.2
        evidence = 0.95 if rows and cols else 0.25
        score = relevance * readability * visual_value * evidence
        return VisualAssetDecision(
            asset_id=getattr(table, "table_id", ""), kind="chart" if "chart" in (getattr(table, "caption", "") or "").lower() else "table",
            score=round(score, 4), relevance=relevance, readability=readability,
            visual_value=visual_value, evidence_strength=evidence,
            target_section_id="sec-main-method", reason="compact numeric evidence candidate",
            message=(getattr(table, "caption", "") or "").strip(),
            recommended_width=0.70 if relevance >= 0.9 else 0.45,
        )

    def _score_formula(self, formula, analysis: PaperAnalysis) -> VisualAssetDecision:
        latex = getattr(formula, "latex", "") or ""
        desc = getattr(formula, "semantic_desc", "") or ""
        relevance = 0.9 if desc else 0.62
        readability = 0.86 if len(latex) < 100 and latex.count("\\") < 12 else 0.48
        visual_value = 0.82 if desc else 0.55
        evidence = 0.78
        score = relevance * readability * visual_value * evidence
        return VisualAssetDecision(
            asset_id=getattr(formula, "formula_id", ""), kind="formula", score=round(score, 4),
            relevance=relevance, readability=readability, visual_value=visual_value,
            evidence_strength=evidence, target_section_id="sec-key-idea",
            message=desc.strip(), recommended_width=0.64 if readability >= 0.8 else 0.82,
            reason="short semantic formula candidate",
        )

    def _mark_redundancy(self, decisions: list[VisualAssetDecision], figures: list) -> None:
        for index, current in enumerate(decisions):
            current_text = next((_figure_role_text(fig) for fig in figures if getattr(fig, "figure_id", "") == current.asset_id), "")
            for previous in decisions[:index]:
                previous_text = next((_figure_role_text(fig) for fig in figures if getattr(fig, "figure_id", "") == previous.asset_id), "")
                if _similarity(current_text, previous_text) >= self.redundancy_threshold:
                    current.action = "annotate" if current.score >= previous.score else "remove"
                    current.redundancy_group = previous.asset_id
                    current.reason = f"redundant with {previous.asset_id}; convert to annotation" if current.action == "annotate" else f"redundant with {previous.asset_id}"
                    break


def plan_visual_assets(doc: PaperDocument, analysis: PaperAnalysis) -> VisualAssetPlan:
    return VisualAssetPlanner().plan(doc, analysis)
