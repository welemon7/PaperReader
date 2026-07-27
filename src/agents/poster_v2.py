from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import FigurePlacement, FormulaDisplay, PosterBlueprint, PosterSection
from src.schemas.poster_v2 import LayoutConstraints, LayoutNode, LayoutTree
from src.renderers.html_renderer import HtmlPosterRenderer
from src.storage.sqlite import PaperDatabase
from src.utils.output_paths import resolve_paper_output_dir
from src.agents.poster_planner import (
    _augment_key_formulas,
    _default_colors,
    _drop_top_summary_sections,
    _format_authors,
    _tighten_layout,
    generate_blueprint,
    normalize_analysis_for_poster,
)

logger = logging.getLogger(__name__)


def _layout_node_from_section(sec: PosterSection, reading_order: int) -> LayoutNode:
    return LayoutNode(
        node_id=sec.section_id,
        node_type="text" if sec.type != "title" else "title",
        title=sec.title,
        content_md=sec.content_md,
        child_ids=[],
        figure_ids=[],
        reading_order=reading_order,
        space_ratio=min(1.0, max(0.08, (len(sec.content_md or "") / 1400.0) + 0.1)),
        section_col_span=sec.col_span,
        section_row_span=sec.row_span,
        figure_width_ratio=0.9,
        constraints=LayoutConstraints(
            min_ratio=0.08,
            max_ratio=0.95,
            priority=2 if sec.type in {"main_method", "experiments"} else 1,
        ),
        notes=sec.type,
    )


def _layout_nodes_from_paper_section(sec, reading_order: int = 0) -> list[LayoutNode]:
    nodes = [LayoutNode(
        node_id=sec.section_id,
        node_type="title" if getattr(sec, "level", 1) == 1 else "text",
        title=getattr(sec, "title", ""),
        content_md=(getattr(sec, "text", "") or getattr(sec, "raw_latex", "") or ""),
        figure_ids=[getattr(fig, "figure_id", "") for fig in getattr(sec, "figures", []) if getattr(fig, "figure_id", "")],
        child_ids=[getattr(sub, "section_id", "") for sub in getattr(sec, "subsections", []) if getattr(sub, "section_id", "")],
        reading_order=reading_order,
        space_ratio=min(1.0, max(0.08, (len((getattr(sec, "text", "") or "")) / 1400.0) + 0.08)),
        section_col_span=1,
        section_row_span=1,
        figure_width_ratio=0.9,
        constraints=LayoutConstraints(min_ratio=0.08, max_ratio=0.95, priority=max(1, 4 - getattr(sec, "level", 1))),
        notes=f"paper-section:{getattr(sec, 'level', 1)}",
    )]
    next_order = reading_order + 1
    for sub in getattr(sec, "subsections", []) or []:
        sub_nodes = _layout_nodes_from_paper_section(sub, next_order)
        nodes.extend(sub_nodes)
        next_order += len(sub_nodes)
    return nodes


def build_layout_tree(doc: PaperDocument, analysis: PaperAnalysis) -> LayoutTree:
    """Build layout tree using deterministic fallback only."""
    analysis = normalize_analysis_for_poster(analysis.model_copy(deep=True))
    _augment_key_formulas(doc, analysis)

    required_items = [doc.title]
    if analysis.problem_statement:
        required_items.append(analysis.problem_statement)
    required_items.extend(c.text for c in analysis.contributions[:4] if c.text)
    if analysis.method_overview:
        required_items.append(analysis.method_overview)
    if analysis.experiments and analysis.experiments.main_results:
        required_items.append(analysis.experiments.main_results)

    return _fallback_layout_tree(doc, analysis, required_items)


def _fallback_layout_tree(doc: PaperDocument, analysis: PaperAnalysis, required_items: Optional[list[str]] = None) -> LayoutTree:
    """Deterministic layout tree from paper sections and static blueprint."""
    nodes = []
    order = 0
    for sec in doc.sections:
        paper_nodes = _layout_nodes_from_paper_section(sec, order)
        nodes.extend(paper_nodes)
        order += len(paper_nodes)

    sections = generate_blueprint(doc, analysis).sections
    sections = _drop_top_summary_sections(sections)
    for sec in sections:
        nodes.append(_layout_node_from_section(sec, order))
        order += 1
    return LayoutTree(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        title=doc.title,
        required_items=required_items or [doc.title],
        nodes=nodes,
        root_id="root",
        reading_path=[node.node_id for node in nodes],
        layout_notes=["Deterministic layout tree."],
    )


def layout_tree_to_blueprint(tree: LayoutTree, doc: PaperDocument, analysis: PaperAnalysis) -> PosterBlueprint:
    base = generate_blueprint(doc, analysis)
    node_map = {node.node_id: node for node in tree.nodes}

    sections: list[PosterSection] = []
    for sec in base.sections:
        node = node_map.get(sec.section_id)
        if node:
            sec.content_md = node.content_md or sec.content_md
            sec.title = node.title or sec.title
            sec.row = min(3, max(0, node.reading_order))
            sec.col_span = node.section_col_span if node.section_col_span else (2 if node.space_ratio >= 0.35 else sec.col_span)
            sec.row_span = node.section_row_span if node.section_row_span else sec.row_span
            if sec.type == "title":
                sec.col_span = 3
        sections.append(sec)

    if not sections:
        sections = base.sections

    figure_placements = list(base.figure_placements)
    formula_displays = list(base.formula_displays)

    for node in tree.nodes:
        if node.node_type == "figure":
            for figure_id in node.figure_ids:
                if figure_id not in {fp.figure_id for fp in figure_placements}:
                    figure_placements.append(FigurePlacement(
                        figure_id=figure_id,
                        section_id=node.node_id,
                        width_ratio=max(0.35, min(1.0, node.figure_width_ratio)),
                        caption=node.title,
                    ))
        if node.node_type == "formula" and node.content_md:
            formula_displays.append(FormulaDisplay(
                formula_id=node.node_id,
                section_id=node.node_id,
                latex=node.content_md,
                semantic_desc=node.notes,
            ))

    _tighten_layout(sections, figure_placements)
    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str=_format_authors(doc.authors),
        code_url=analysis.code_url,
        width_px=base.width_px,
        height_px=base.height_px,
        width_mm=base.width_mm,
        height_mm=base.height_mm,
        sections=sections,
        figure_placements=figure_placements[:4],
        formula_displays=formula_displays[:5],
        color_scheme=base.color_scheme or _default_colors(),
    )


def render_layout_tree(doc: PaperDocument, analysis: PaperAnalysis, tree: LayoutTree, output_dir: Path) -> tuple[PosterBlueprint, Path]:
    blueprint = layout_tree_to_blueprint(tree, doc, analysis)
    renderer = HtmlPosterRenderer()
    html_path = output_dir / "poster.html"
    renderer.render_to_file(blueprint, doc, html_path)
    return blueprint, html_path


def run_poster_v2(arxiv_id: str, output_dir: Path | str = Path("output")) -> dict[str, Any]:
    """Run v2 poster pipeline without LLM.

    Returns:
        dict: Contains layout_tree, blueprint, and html_path only.
    """
    out = resolve_paper_output_dir(output_dir, arxiv_id)
    db = PaperDatabase()
    doc = db.get_paper_by_arxiv(arxiv_id)
    analysis = db.get_analysis_by_arxiv(arxiv_id)
    db.close()
    if not doc or not analysis:
        raise RuntimeError(f"Paper or analysis not found for {arxiv_id}")

    tree = build_layout_tree(doc, analysis)
    blueprint, html_path = render_layout_tree(doc, analysis, tree, out)

    (out / "layout_tree.json").write_text(tree.model_dump_json(indent=2), encoding="utf-8")
    (out / "blueprint_v2.json").write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")

    return {
        "layout_tree": tree,
        "blueprint": blueprint,
        "html_path": html_path,
    }