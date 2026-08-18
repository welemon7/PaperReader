"""Build a PosterScene from the blueprint, paper and analysis.

Also prepares figure assets into the target output directory and measures each
figure's native aspect ratio so the layout solver can size frames uniformly
(no more misaligned columns from mixed image sizes).
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from pathlib import Path
from typing import Optional

from src.layout.scene import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    ElementKind,
    PosterScene,
    SceneConstraints,
    SceneElement,
    ScenePanel,
)
from src.layout.themes import resolve_colors
from src.renderers.html_renderer import HtmlPosterRenderer
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import PosterBlueprint

logger = logging.getLogger(__name__)

# panel_type -> zone
ZONE_BY_TYPE = {
    "motivation": "left",
    "method_overview": "center",
    "key_idea": "right_top",
    "main_method": "right_bottom",
    "experiments": "right_bottom",
    "contributions": "bottom_left",
    "highlights": "bottom_center",
    "project_link": "bottom_right",
}

# Figures per panel type. The key-idea panel stays bullet-only: a tiny
# thumbnail there reads as noise; figures carry the method + results story.
FIGURE_SLOTS = {"method_overview": 1, "main_method": 2, "key_idea": 0}


def build_scene(
    blueprint: PosterBlueprint,
    doc: PaperDocument,
    analysis: PaperAnalysis,
    output_dir: Path,
    theme: str = "academic",
) -> PosterScene:
    """Deterministically assemble the scene graph from the blueprint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prepare figure assets and build section -> figures map.
    try:
        renderer = HtmlPosterRenderer()
        renderer._prepare_figure_assets(blueprint, doc, output_dir)
        fig_map = HtmlPosterRenderer._build_figure_map(blueprint, doc, output_dir)
    except Exception as exc:
        logger.warning("Figure asset preparation failed: %s", exc)
        fig_map = {}

    panels: dict[str, ScenePanel] = {}
    used_figures: dict[str, int] = {}

    for sec in blueprint.sections:
        if sec.type == "title":
            continue
        zone = ZONE_BY_TYPE.get(sec.type)
        if not zone:
            continue
        panel = panels.get(zone)
        if panel is None:
            panel = ScenePanel(
                panel_id=sec.section_id,
                panel_type=sec.type,
                title=sec.title or _default_title(sec.type),
                zone=zone,
                constraints=_constraints_for(sec.type),
            )
            panels[zone] = panel
            if zone == "right_bottom":
                panel.panel_id = "sec-main-method"
                panel.title = "Core Results"

        # Text content (split the [[CORE_TABLE]] marker).  Embedded formula /
        # callout boxes are extracted into standalone elements so text height
        # estimation is pure text and formulas render exactly once.
        md = sec.content_md or ""
        intro_md, table_html = _split_core_table(md)
        intro_md, embedded_blocks = _extract_html_blocks(intro_md)
        if intro_md and intro_md.strip() and intro_md.strip() != "(not provided)":
            panel.elements.append(SceneElement(
                element_id=f"{panel.panel_id}.text-{len(panel.elements)}",
                kind="text", content_md=intro_md,
            ))
        for kind, block_html in embedded_blocks:
            el_kind = "formula" if kind == "formula-box" else kind
            panel.elements.append(SceneElement(
                element_id=f"{panel.panel_id}.{el_kind}-{len(panel.elements)}",
                kind=el_kind, content_html=block_html,
            ))

        # Figures for this panel (uniform frames, aspect measured). Figures come
        # BEFORE the table so the visual evidence keeps the prime space and the
        # compact table settles at the bottom.
        figures_added = False
        for entry in fig_map.get(sec.section_id, []):
            slot = used_figures.get(sec.section_id, 0)
            if slot >= FIGURE_SLOTS.get(sec.type, 1):
                break
            src = entry.get("src") or ""
            if not src:
                continue
            aspect = _measure_aspect(output_dir, src)
            panel.elements.append(SceneElement(
                element_id=f"{panel.panel_id}.fig-{slot}",
                kind="figure",
                figure_id=str(entry.get("figure_id") or ""),
                figure_src=src,
                figure_aspect=aspect,
                notes=str(entry.get("caption") or "")[:200],
            ))
            used_figures[sec.section_id] = slot + 1
            figures_added = True
        if table_html:
            panel.elements.append(SceneElement(
                element_id=f"{panel.panel_id}.table",
                kind="table", content_html=_compact_table(table_html),
            ))

    # Formulas from the blueprint (labels mirror the planner's conventions).
    _attach_formulas(panels, blueprint)

    # Project panel: code link only (no QR code).
    project = panels.get("bottom_right")
    if project is not None:
        project.elements = [e for e in project.elements if e.kind != "text"]
        code_url = (blueprint.code_url or "").strip()
        project.elements.append(SceneElement(
            element_id="sec-project.link", kind="link",
            content_md=code_url or "Code will be released.",
        ))

    return PosterScene(
        paper_id=doc.paper_id,
        poster_title=blueprint.poster_title,
        authors_str=blueprint.authors_str,
        code_url=blueprint.code_url,
        canvas_width=blueprint.width_px or CANVAS_WIDTH,
        canvas_height=blueprint.height_px or CANVAS_HEIGHT,
        panels=list(panels.values()),
        color_scheme=resolve_colors(theme, blueprint.color_scheme or {}),
        theme=theme,
    )


def _compact_table(table_html: str) -> str:
    """Make the results table predictable and compact.

    - keep at most 3 rows (drop rows whose label duplicates the intro text,
      e.g. Metrics),
    - truncate long cell values,
    so the rendered height stays close to the solver's estimate and the audit
    never sees a clipped table.
    """
    rows = re.findall(r"<tr><th>(.*?)</th><td>(.*?)</td></tr>", table_html, flags=re.DOTALL)
    if not rows:
        return table_html
    keep: list[tuple[str, str]] = []
    for label, value in rows:
        label = re.sub(r"<[^>]+>", "", label).strip()
        value = re.sub(r"<br\s*/?>", " / ", value)
        value = re.sub(r"<[^>]+>", "", value).strip()
        if not label or not value or value.lower() == "not specified":
            continue
        if label.lower() == "metrics":
            continue  # duplicates the intro sentence
        keep.append((label, value))
    keep = keep[:3]
    if not keep:
        return table_html
    cells = []
    for label, value in keep:
        # truncate long cells to ~46 chars
        words = value.split()
        if len(value) > 46:
            value = " ".join(words[:9]) + "…"
        cells.append(
            f"<tr><th>{html_lib.escape(label)}</th><td>{html_lib.escape(value)}</td></tr>"
        )
    return (
        '<div class="item-details-wrap">'
        '<div class="item-details-title">Item Details</div>'
        '<table class="item-details-table"><tbody>'
        + "".join(cells)
        + "</tbody></table></div>"
    )


def _split_core_table(md: str) -> tuple[str, str]:
    if "[[CORE_TABLE]]" not in md:
        return md, ""
    before, after = md.split("[[CORE_TABLE]]", 1)
    return before, after.strip()


_DIV_BLOCK_RE = re.compile(r'<div class="(formula-box|callout)"', re.IGNORECASE)


def _extract_html_blocks(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Pull formula-box / callout divs out of markdown-ish text.

    Returns (cleaned_text, [(kind, full_html_block), ...]) with balanced
    nesting, so the text element stays plain text and the extracted blocks
    become standalone scene elements (single render, predictable heights).
    """
    cleaned: list[str] = []
    blocks: list[tuple[str, str]] = []
    pos = 0
    for m in _DIV_BLOCK_RE.finditer(md or ""):
        start = m.start()
        cleaned.append(md[pos:start])
        depth = 1
        i = m.end()
        while i < len(md) and depth > 0:
            if md.startswith("<div", i) and (i + 4 >= len(md) or md[i + 4] in " \t\r\n>"):
                depth += 1
                i += 4
            elif md.startswith("</div>", i):
                depth -= 1
                i += 6
            else:
                i += 1
        blocks.append((m.group(1).lower(), md[start:i]))
        pos = i
    cleaned.append(md[pos:])
    return "".join(cleaned).strip(), blocks


def _attach_formulas(panels: dict[str, ScenePanel], blueprint: PosterBlueprint) -> None:
    from src.agents.poster_planner import _clean_formula_latex

    def _panel_latex(panel: ScenePanel) -> set[str]:
        found: set[str] = set()
        for e in panel.elements:
            if e.kind == "formula":
                m = re.search(r"\$\$(.+?)\$\$", e.content_html or "", flags=re.DOTALL)
                if m:
                    found.add(_clean_formula_latex(m.group(1)))
        return found

    method = panels.get("center")
    motivation = panels.get("left")
    seen_latex: set[str] = set()
    for idx, f in enumerate(blueprint.formula_displays):
        target = method if idx == 0 else (motivation if idx == 1 and motivation else method)
        if target is None:
            continue
        latex = _clean_formula_latex(f.latex or "")
        if not latex or latex in seen_latex or latex in _panel_latex(target):
            continue
        seen_latex.add(latex)
        desc = html_lib.escape((f.semantic_desc or "").strip())
        label = "Main Formula" if idx == 0 else "Key Formula"
        target.elements.append(SceneElement(
            element_id=f"{target.panel_id}.formula-{idx}",
            kind="formula",
            content_html=(
                '<div class="formula-box">'
                f'<div class="formula-label">{label}</div>'
                f'<div>$$ {latex} $$</div>'
                f'<div class="formula-desc">{desc}</div>'
                "</div>"
            ),
        ))


def _constraints_for(panel_type: str) -> SceneConstraints:
    if panel_type in ("main_method", "method_overview"):
        return SceneConstraints(min_ratio=0.38, max_ratio=0.9, priority=2, min_font_scale=0.82)
    if panel_type in ("motivation", "key_idea"):
        return SceneConstraints(min_ratio=0.2, max_ratio=0.8, priority=1, min_font_scale=0.82)
    if panel_type in ("contributions", "highlights"):
        # 底部栏较矮：允许字体缩到 0.82 以容纳 4 条 bullet + badges
        return SceneConstraints(min_ratio=0.1, max_ratio=1.0, priority=0, min_font_scale=0.82)
    return SceneConstraints(min_ratio=0.1, max_ratio=1.0, priority=0, min_font_scale=0.85)


def _default_title(panel_type: str) -> str:
    return {
        "motivation": "Motivation",
        "method_overview": "Method Overview",
        "key_idea": "Key Idea",
        "main_method": "Core Results",
        "experiments": "Experiments",
        "contributions": "Contributions",
        "highlights": "Highlights",
        "project_link": "Project",
    }.get(panel_type, panel_type.title())


def _measure_aspect(output_dir: Path, src: str) -> float:
    """Native width/height of a figure asset (0 when unreadable)."""
    candidate = (output_dir / src).resolve()
    if not candidate.exists():
        return 0.0
    try:
        from PIL import Image

        with Image.open(candidate) as img:
            w, h = img.size
            if h <= 0:
                return 0.0
            return round(w / float(h), 3)
    except Exception:
        return 0.0


def _default_colors() -> dict:
    return {
        "primary": "#16324f",
        "accent": "#5a7d9a",
        "background": "#fbfcfe",
        "text": "#182433",
        "section_header_bg": "#e8eef4",
        "section_header_text": "#16324f",
        "border": "#cfd8e3",
        "highlight": "#8fb3d9",
    }
