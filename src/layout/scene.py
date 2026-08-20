"""Scene graph model: panels -> elements -> constraints, plus controlled patches.

Canonical 48x36 inch landscape poster at 40 dpi => 1920x1440 CSS px.
Zones follow the agreed template: title band on top, motivation left, big
method area centre, key concepts + results right, contributions / highlights /
QR code along the bottom.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

PanelType = Literal[
    "title",
    "motivation",
    "method_overview",
    "key_idea",
    "main_method",
    "experiments",
    "contributions",
    "highlights",
    "project_link",
]

Zone = Literal[
    "title", "left", "center", "right_top", "right_bottom",
    "bottom_left", "bottom_center", "bottom_right",
]

ElementKind = Literal["text", "figure", "formula", "table", "callout", "qr", "link", "supplement"]

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1440
CANVAS_WIDTH_IN = 48
CANVAS_HEIGHT_IN = 36


class SceneConstraints(BaseModel):
    """Space/priority constraints for a panel (used by the layout solver)."""

    min_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    max_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: int = Field(default=0, ge=0)
    min_font_scale: float = Field(default=0.82, ge=0.5, le=1.0)


class SceneElement(BaseModel):
    element_id: str
    kind: ElementKind | Literal["supplement"] = "text"
    content_md: str = ""
    content_html: str = ""
    figure_id: str = ""
    figure_src: str = ""
    # Native width/height ratio of the figure asset (0 = unknown).
    figure_aspect: float = Field(default=0.0, ge=0.0, le=20.0)
    # Preferred fraction of the panel height for a figure box (0 = auto).
    box_hint: float = Field(default=0.0, ge=0.0, le=1.0)
    # Font scale applied by the solver/patches (1.0 = default).
    font_scale: float = Field(default=1.0, ge=0.6, le=1.2)
    notes: str = ""


class ScenePanel(BaseModel):
    panel_id: str
    panel_type: PanelType = "text"
    title: str = ""
    zone: Zone = "left"
    elements: list[SceneElement] = Field(default_factory=list)
    constraints: SceneConstraints = Field(default_factory=SceneConstraints)
    notes: str = ""


class PosterScene(BaseModel):
    paper_id: str = ""
    poster_title: str = ""
    authors_str: str = ""
    code_url: str = ""
    canvas_width: int = CANVAS_WIDTH
    canvas_height: int = CANVAS_HEIGHT
    panels: list[ScenePanel] = Field(default_factory=list)
    color_scheme: dict = Field(default_factory=dict)
    theme: str = "academic"

    def panel(self, panel_id: str) -> Optional[ScenePanel]:
        for p in self.panels:
            if p.panel_id == panel_id:
                return p
        return None

    def panel_by_type(self, panel_type: str) -> Optional[ScenePanel]:
        for p in self.panels:
            if p.panel_type == panel_type:
                return p
        return None


# ---------------------------------------------------------------------------
# Controlled patches
# ---------------------------------------------------------------------------


class ScenePatch(BaseModel):
    patch_id: str = ""
    kind: Literal[
        "condense_text",
        "resize_figure",
        "reflow_panel",
        "replace_figure",
        "remove_element",
        "adjust_font",
        "supplement_panel",
    ] = "reflow_panel"
    target: str = ""  # panel_id or element_id
    params: dict = Field(default_factory=dict)
    reason: str = ""

    def describe(self) -> str:
        return f"{self.kind} {self.target} {self.reason or ''}".strip()


# ---- patch helpers ----------------------------------------------------------


def make_condense_patch(panel_id: str, max_words: int, reason: str = "") -> ScenePatch:
    return ScenePatch(
        kind="condense_text", target=panel_id,
        params={"max_words": max_words}, reason=reason,
    )


def make_resize_figure_patch(panel_id: str, box_hint: float, reason: str = "") -> ScenePatch:
    return ScenePatch(
        kind="resize_figure", target=panel_id,
        params={"box_hint": max(0.15, min(0.85, box_hint))}, reason=reason,
    )


def make_reflow_panel_patch(panel_id: str, grow: bool, reason: str = "") -> ScenePatch:
    return ScenePatch(
        kind="reflow_panel", target=panel_id,
        params={"grow": bool(grow)}, reason=reason,
    )


def make_replace_figure_patch(panel_id: str, figure_id: str, reason: str = "") -> ScenePatch:
    return ScenePatch(
        kind="replace_figure", target=panel_id,
        params={"figure_id": figure_id}, reason=reason,
    )


def make_remove_element_patch(panel_id: str, element_kind: str = "figure", reason: str = "") -> ScenePatch:
    return ScenePatch(
        kind="remove_element", target=panel_id,
        params={"element_kind": element_kind}, reason=reason,
    )


def make_font_patch(panel_id: str, font_scale: float, reason: str = "") -> ScenePatch:
    return ScenePatch(
        kind="adjust_font", target=panel_id,
        params={"font_scale": max(0.7, min(1.15, font_scale))}, reason=reason,
    )
