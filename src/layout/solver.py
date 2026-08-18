"""Deterministic layout solver for the 48x36 inch poster scene.

Takes a ``PosterScene`` (panels with text/figure/formula elements and
constraints) and produces absolute boxes on the 1920x1440 canvas.  No model
is involved: column widths, figure frames and font scales are computed from
content estimates and image aspect ratios, then verified by the browser audit
in the harness loop.

Template (48x36 in, 40 dpi):
- top title band
- main area: left = motivation, centre = big method area, right = key ideas
  (top) + results (bottom)
- bottom strip: contributions | highlights | QR / project
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

from pydantic import BaseModel, Field

from src.layout.scene import PosterScene, SceneElement, ScenePanel

logger = logging.getLogger(__name__)

# Canvas geometry ------------------------------------------------------------
MARGIN = 24
GUTTER = 16
TITLE_BAND_H = 150
BOTTOM_BAND_H = 320

BODY_FONT_PX = 17.0
LINE_HEIGHT = 1.42
PANEL_TITLE_H = 52
PANEL_PAD = 16
MIN_BODY_FONT_PX = 13.0
# Conservative multiplier on estimated text height: Chromium renders taller
# than a naive chars-per-line estimate (margins, li spacing, font metrics).
HEIGHT_FUDGE = 1.18


class PanelBox(BaseModel):
    panel_id: str
    zone: str = "left"
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


class ElementBox(BaseModel):
    element_id: str
    kind: str = "text"
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    figure_src: str = ""
    figure_id: str = ""
    content_html: str = ""
    content_md: str = ""
    font_scale: float = 1.0
    align: str = "left"
    hide: bool = False


class SceneLayout(BaseModel):
    canvas_width: int = 1920
    canvas_height: int = 1440
    panels: list[PanelBox] = Field(default_factory=list)
    elements: list[ElementBox] = Field(default_factory=list)

    def panel(self, panel_id: str) -> Optional[PanelBox]:
        for p in self.panels:
            if p.panel_id == panel_id:
                return p
        return None

    def elements_in(self, panel_id: str) -> list[ElementBox]:
        return [e for e in self.elements if e.element_id.startswith(panel_id + ".")]


# ---- text estimation -------------------------------------------------------


def _visible_text_len(text: str) -> int:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"(^|\n)\s*[-*]\s+", r"\1", text)
    return len(text)


def _lines_for(text: str, box_width: int, font_px: float) -> int:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\\\$\\\$", " ", text)
    text = re.sub(r"\$\$.+?\$\$", " FORMULA ", text, flags=re.DOTALL)
    chars_per_line = max(8, int(box_width / (font_px * 0.52)))
    total = 0
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            total += 1
            continue
        line_chars = 0
        lines = 1
        for word in words:
            need = len(word) + (1 if line_chars else 0)
            if line_chars + need > chars_per_line:
                lines += 1
                line_chars = len(word)
            else:
                line_chars += need
        total += lines
    return max(1, total)


def _text_height(text: str, box_width: int, font_px: float) -> float:
    """Estimate rendered text height, aware of embedded HTML blocks.

    The poster text elements may contain ``.formula-box`` / ``.callout`` /
    ``.item-details-wrap`` HTML (e.g. a Key Formula card inside a panel's
    markdown).  Those render as block boxes far taller than plain lines, so
    they are estimated separately instead of being treated as text lines.
    """
    lines_h = _lines_for(text, box_width, font_px) * font_px * LINE_HEIGHT
    extra = 0.0
    extra += 150.0 * (text or "").count('class="formula-box"')
    extra += 100.0 * (text or "").count('class="callout"')
    extra += 160.0 * (text or "").count('class="item-details-wrap"')
    bullets = (text or "").count("\n- ") + (text or "").count("\n* ")
    extra += bullets * 6.0
    return (lines_h + extra) * HEIGHT_FUDGE


def _fit_font_scale(text: str, box_width: int, box_height: float, min_scale: float) -> float:
    """Largest font scale in [min_scale, 1.0] whose estimated text fits."""
    if not text or box_height <= 0:
        return 1.0
    scale = 1.0
    for _ in range(6):
        est = _text_height(text, box_width, BODY_FONT_PX * scale)
        if est <= box_height:
            return scale
        scale -= 0.03
        if scale < min_scale:
            return min_scale
    return min_scale


# ---- figure boxes ----------------------------------------------------------

_DEFAULT_ASPECT = {"method_overview": 1.6, "main_method": 1.3, "key_idea": 1.2}
# Default figure box height as a fraction of the *remaining* panel body height.
_FIGURE_HINT = {"method_overview": 0.68, "main_method": 0.5, "key_idea": 0.5, "motivation": 0.5}


def _figure_box(
    panel: ScenePanel,
    body_x: int,
    body_y: int,
    body_w: int,
    remaining_h: float,
    fig: SceneElement,
    two_up: bool = False,
) -> ElementBox:
    aspect = fig.figure_aspect if fig.figure_aspect > 0 else _DEFAULT_ASPECT.get(panel.panel_type, 1.4)
    hint = fig.box_hint if fig.box_hint > 0 else _FIGURE_HINT.get(panel.panel_type, 0.5)
    if two_up:
        w = max(120, (body_w - 10) // 2)
        h = min(max(40.0, remaining_h * 0.72), w / aspect)
        return ElementBox(
            element_id=fig.element_id, kind="figure",
            x=body_x, y=body_y, w=w, h=max(40, round(h)),
            figure_src=fig.figure_src, figure_id=fig.figure_id, align="left",
        )
    w = body_w
    h = min(max(40.0, remaining_h * hint), w / aspect)
    return ElementBox(
        element_id=fig.element_id, kind="figure",
        x=body_x, y=body_y, w=w, h=max(40, round(h)),
        figure_src=fig.figure_src, figure_id=fig.figure_id, align="center",
    )


# ---- solver ----------------------------------------------------------------


def solve_layout(scene: PosterScene) -> SceneLayout:
    """Compute absolute panel/element boxes for the scene."""
    W = scene.canvas_width
    H = scene.canvas_height
    layout = SceneLayout(canvas_width=W, canvas_height=H)

    panels = {p.panel_id: p for p in scene.panels}
    # Title is rendered by the header, not a panel box.
    for pid in list(panels):
        if panels[pid].panel_type == "title":
            panels.pop(pid, None)

    # Main area geometry.
    y_main = MARGIN + TITLE_BAND_H + GUTTER
    y_bottom = H - MARGIN - BOTTOM_BAND_H
    main_h = y_bottom - GUTTER - y_main
    content_w = W - 2 * MARGIN - 2 * GUTTER
    left_w = round(content_w * 0.26)
    center_w = round(content_w * 0.42)
    right_w = content_w - left_w - center_w

    # ---- columns -----------------------------------------------------------
    left_x, center_x, right_x = MARGIN, MARGIN + left_w + GUTTER, MARGIN + left_w + GUTTER + center_w + GUTTER

    # Right column split: key_idea on top (bullets only), results below.
    key_h = round(main_h * 0.34)
    results_h = main_h - key_h

    zone_boxes = {
        "left": PanelBox(panel_id="", zone="left", x=left_x, y=y_main, w=left_w, h=main_h),
        "center": PanelBox(panel_id="", zone="center", x=center_x, y=y_main, w=center_w, h=main_h),
        "right_top": PanelBox(panel_id="", zone="right_top", x=right_x, y=y_main, w=right_w, h=key_h),
        "right_bottom": PanelBox(panel_id="", zone="right_bottom", x=right_x, y=y_main + key_h + GUTTER, w=right_w, h=results_h),
    }
    bottom_w = (content_w - 2 * GUTTER) // 3
    bottom_zones = {
        "bottom_left": PanelBox(panel_id="", zone="bottom_left", x=MARGIN, y=y_bottom, w=bottom_w, h=BOTTOM_BAND_H),
        "bottom_center": PanelBox(panel_id="", zone="bottom_center", x=MARGIN + bottom_w + GUTTER, y=y_bottom, w=bottom_w, h=BOTTOM_BAND_H),
        "bottom_right": PanelBox(panel_id="", zone="bottom_right", x=MARGIN + 2 * (bottom_w + GUTTER), y=y_bottom,
                                 w=content_w - 2 * bottom_w - 2 * GUTTER, h=BOTTOM_BAND_H),
    }

    for pid, panel in panels.items():
        zone_box = zone_boxes.get(panel.zone) or bottom_zones.get(panel.zone)
        if zone_box is None:
            logger.warning("No zone box for panel %s (zone=%s)", pid, panel.zone)
            continue
        box = zone_box.model_copy(update={"panel_id": pid, "zone": panel.zone})
        layout.panels.append(box)
        _layout_panel_elements(panel, box, layout)

    # Bottom-strip panels always fit their content (bullets), but keep a
    # second fitting pass per panel already done above.
    return layout


def _layout_panel_elements(panel: ScenePanel, box: PanelBox, layout: SceneLayout) -> None:
    """Place the panel's elements in scene order with deterministic fitting.

    Element order in the scene is respected (e.g. text -> figures -> table in
    the results panel).  A table element is bottom-anchored: it gets a fixed
    reserved strip at the bottom of the panel so that estimate drift in text /
    figure heights can never push it out of the panel; the elements above it
    fit within the remaining space.
    """
    body_x = box.x + PANEL_PAD
    body_y = box.y + PANEL_TITLE_H
    body_w = max(80, box.w - 2 * PANEL_PAD)
    body_h = max(40, box.h - PANEL_TITLE_H - PANEL_PAD)
    panel_bottom = body_y + body_h

    # Bottom-anchored table reserve.
    table_el = next((e for e in panel.elements if e.kind == "table"), None)
    table_h = 0.0
    if table_el is not None:
        table_h = _estimate_table_height(table_el.content_html, body_h)
        table_h = max(40.0, min(body_h, table_h))
    content_h = max(40.0, body_h - table_h)

    figures = [e for e in panel.elements if e.kind == "figure"]
    fig_idx = 0
    pending_figures = len(figures)
    fig_reserve = min(200.0, content_h * 0.42) if pending_figures else 0.0

    cursor = body_y
    remaining = content_h

    for el in panel.elements:
        if el.kind == "table":
            # Bottom-anchored: fixed strip at the panel bottom.
            layout.elements.append(ElementBox(
                element_id=el.element_id, kind="table",
                x=body_x, y=round(panel_bottom - table_h), w=body_w, h=round(table_h),
                content_html=el.content_html, content_md=el.content_md,
            ))
            continue
        if el.kind == "figure":
            if fig_idx >= len(figures):
                continue
            remaining_after = len(figures) - fig_idx
            two_up = (
                remaining_after >= 2
                and panel.panel_type in ("main_method", "experiments")
            )
            if two_up:
                f0 = _figure_box(panel, body_x, cursor, body_w, remaining, figures[fig_idx], two_up=True)
                f1 = _figure_box(panel, body_x, cursor, body_w, remaining, figures[fig_idx + 1], two_up=True)
                f0.x = body_x
                f1.x = body_x + f0.w + 10  # side-by-side regardless of equal width
                row_h = max(f0.h, f1.h)    # equal heights align the row
                if cursor + row_h > body_y + content_h:
                    row_h = max(40, body_y + content_h - cursor)
                f0.h, f1.h = row_h, row_h
                layout.elements.extend([f0, f1])
                cursor += row_h + 10
                remaining = max(0.0, remaining - row_h - 10)
                fig_idx += 2
            else:
                fbox = _figure_box(panel, body_x, cursor, body_w, remaining, figures[fig_idx])
                if cursor + fbox.h > body_y + content_h:
                    fbox.h = max(40, body_y + content_h - cursor)
                layout.elements.append(fbox)
                cursor += fbox.h + 10
                remaining = max(0.0, remaining - fbox.h - 10)
                fig_idx += 1
            pending_figures = max(0, len(figures) - fig_idx)
            continue

        # text-like elements (text / formula / table / callout only)
        if el.kind in ("text", "formula", "table", "callout"):
            if el.kind == "text":
                # Reserve space for figures that still come after this text.
                budget = max(40.0, remaining - fig_reserve) if pending_figures else remaining
                scale = _fit_font_scale(el.content_md, body_w, budget, panel.constraints.min_font_scale)
                font_px = BODY_FONT_PX * scale
                est_h = _text_height(el.content_md, body_w, font_px)
            else:
                font_px = BODY_FONT_PX * el.font_scale
                est_h = _estimate_fixed_height(el, body_w, remaining)
            h = min(remaining, max(18, round(est_h)))
            layout.elements.append(ElementBox(
                element_id=el.element_id, kind=el.kind,
                x=body_x, y=cursor, w=body_w, h=h,
                content_html=el.content_html, content_md=el.content_md,
                font_scale=round(font_px / BODY_FONT_PX, 3),
            ))
            cursor += h + 10
            remaining = max(0.0, remaining - h - 10)

    # 3) remaining elements (qr / link) get a fixed small box.
    for el in panel.elements:
        if el.kind in ("figure", "text", "formula", "table", "callout"):
            continue
        h = 120 if el.kind == "qr" else 44
        layout.elements.append(ElementBox(
            element_id=el.element_id, kind=el.kind,
            x=body_x, y=cursor, w=min(body_w, 200 if el.kind == "qr" else body_w), h=h,
            content_html=el.content_html, content_md=el.content_md,
            align="center" if el.kind == "qr" else "left",
        ))
        cursor += h + 10


def _estimate_fixed_height(el: SceneElement, body_w: int, remaining_h: float) -> float:
    if el.kind == "formula":
        html = el.content_html or ""
        h = 92.0  # label + one equation line + padding
        if 'class="formula-desc"' in html:
            h += 58.0
        # equations with alignment or tall constructs need more room
        if "\\begin{aligned}" in html or "\\frac" in html or "\\sum" in html:
            h += 46.0
        return min(remaining_h, h)
    if el.kind == "table":
        return _estimate_table_height(el.content_html, remaining_h)
    if el.kind == "callout":
        return min(remaining_h, 92.0)
    return min(remaining_h, 60.0)


def _estimate_table_height(html: str, remaining_h: float) -> float:
    """Estimate a rendered table's height from its row/cell content.

    Browsers render tables taller than a flat estimate; count rows and assume
    every row may wrap to two lines so the solver leaves the figures room.
    """
    rows = re.findall(r"<tr[^>]*>", html or "")
    per_row = 46.0 if rows else 30.0
    title_h = 26.0 if "<td" in (html or "") else 0.0
    est = title_h + len(rows) * per_row + 14.0
    return min(remaining_h, est)
