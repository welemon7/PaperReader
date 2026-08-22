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
    role, composition = _semantic_attributes_for_section(sec)
    density = "low" if len(sec.content_md or "") < 180 else ("high" if len(sec.content_md or "") > 600 else "medium")
    return LayoutNode(
        node_id=sec.section_id,
        node_type="text" if sec.type != "title" else "title",
        title=sec.title,
        content_md=sec.content_md,
        child_ids=[],
        children=[],
        figure_ids=[],
        reading_order=reading_order,
        space_ratio=min(1.0, max(0.08, (len(sec.content_md or "") / 1400.0) + 0.1)),
        section_col_span=sec.col_span,
        section_row_span=sec.row_span,
        figure_width_ratio=0.9,
        constraints=LayoutConstraints(
            min_ratio=0.08,
            max_ratio=0.95,
            priority={"P0": 4, "P1": 3, "P2": 2, "P3": 1}.get(sec.visual_priority, 1),
        ),
        notes=f"{sec.type}; visual_priority={sec.visual_priority}; importance={sec.importance:.2f}",
        semantic_role=role,
        importance=sec.importance,
        visual_weight=sec.importance,
        content_density=density,
        composition_type=composition,
        min_area_ratio=max(0.08, sec.importance * 0.25),
        source_section_id=sec.section_id,
    )


def _semantic_attributes_for_section(sec: PosterSection) -> tuple[str, str]:
    mapping = {
        "title": ("header", "header_band"),
        "motivation": ("problem", "problem_statement"),
        "method_overview": ("hero_method", "process_diagram"),
        "key_idea": ("core_principle", "principle_callout"),
        "main_method": ("hero_metric", "metric_callout"),
        "experiments": ("benchmark_chart", "benchmark_chart"),
        "contributions": ("contributions", "footer_strip"),
        "highlights": ("supporting", "panel"),
        "project_link": ("project", "link"),
    }
    return mapping.get(sec.type, ("section", "panel"))


def _semantic_container(node_id: str, role: str, children: list[str], composition: str) -> LayoutNode:
    return LayoutNode(
        node_id=node_id,
        node_type="root" if role == "poster" else "container",
        semantic_role=role,
        composition_type=composition,
        children=children,
        child_ids=children,
        importance=1.0 if role in {"poster", "hero"} else 0.8,
        visual_weight=1.0 if role in {"poster", "hero"} else 0.7,
        content_density="low",
        min_area_ratio=0.12 if role == "hero" else 0.08,
        notes="semantic container",
    )


def _build_semantic_containers(section_nodes: list[LayoutNode]) -> list[LayoutNode]:
    by_role: dict[str, list[str]] = {"header": [], "hero": [], "primary_content": [], "evidence": [], "footer": []}
    for node in section_nodes:
        target = {
            "header": "header",
            "hero_metric": "hero",
            "hero_method": "hero",
            "core_principle": "primary_content",
            "problem": "primary_content",
            "method": "primary_content",
            "evidence": "evidence",
            "benchmark_chart": "evidence",
            "contributions": "footer",
            "project": "footer",
        }.get(node.semantic_role, "footer")
        by_role[target].append(node.node_id)
    containers = [
        _semantic_container("header", "header", by_role["header"], "header_band"),
        _semantic_container("hero", "hero", by_role["hero"], "panel"),
        _semantic_container("primary_content", "primary_content", by_role["primary_content"], "panel"),
        _semantic_container("evidence", "evidence", by_role["evidence"], "evidence_grid"),
        _semantic_container("footer", "footer", by_role["footer"], "footer_strip"),
    ]
    root_children = [node.node_id for node in containers]
    return [_semantic_container("poster", "poster", root_children, "canvas"), *containers]


def _solve_semantic_grid(sec: PosterSection, node: LayoutNode) -> None:
    """Solve semantic roles into the legacy three-column poster grid."""
    placement = {
        "header": (0, 1, 3),
        "problem": (1, 1, 1),
        "hero_method": (1, 2, 1),
        "core_principle": (1, 3, 1),
        "hero_metric": (2, 1, 3),
        "benchmark_chart": (2, 1, 3),
        "evidence": (2, 1, 3),
        "contributions": (3, 1, 1),
        "supporting": (3, 2, 1),
        "project": (3, 3, 1),
    }
    row, column, span = placement.get(node.semantic_role, (sec.row, sec.column, sec.col_span))
    sec.row, sec.column, sec.col_span = row, column, span
    sec.row_span = max(1, node.section_row_span)


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
        semantic_role="supporting",
        composition_type="panel",
        content_density="high" if len(getattr(sec, "text", "") or "") > 600 else "medium",
        source_section_id=getattr(sec, "section_id", ""),
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
    section_nodes: list[LayoutNode] = []
    for sec in sections:
        section_node = _layout_node_from_section(sec, order)
        section_nodes.append(section_node)
        nodes.append(section_node)
        order += 1
    semantic_nodes = _build_semantic_containers(section_nodes)
    for node in semantic_nodes:
        node.reading_order = len(nodes)
        nodes.append(node)
    return LayoutTree(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        title=doc.title,
        required_items=required_items or [doc.title],
        nodes=nodes,
        root_id="poster",
        reading_path=[
            "poster", "header", "hero", "primary_content", "evidence", "footer",
            *[node.node_id for node in section_nodes],
        ],
        layout_notes=["Semantic-first tree; legacy grid fields are solved for blueprint compatibility."],
        layout_mode="semantic_first",
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
            sec.importance = node.importance
            sec.visual_priority = {4: "P0", 3: "P1", 2: "P2", 1: "P3"}.get(
                node.constraints.priority, sec.visual_priority
            )
            _solve_semantic_grid(sec, node)
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
        story_plan=base.story_plan,
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
