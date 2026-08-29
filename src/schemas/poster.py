from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

SectionType = Literal['title', 'motivation', 'method_overview', 'key_idea',
    'main_method', 'experiments', 'contributions', 'highlights', 'project_link', 'references']
StoryBeatType = Literal['hook', 'why', 'idea', 'how', 'evidence']


class PosterStoryBeat(BaseModel):
    beat_id: str
    type: StoryBeatType
    question: str = ''
    text: str = ''
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source_ids: list[str] = Field(default_factory=list)
    target_section_id: str = ''


class PosterStoryPlan(BaseModel):
    """Compressed research narrative consumed by the poster planner."""

    thesis: str = ''
    beats: list[PosterStoryBeat] = Field(default_factory=list)
    reading_path: list[str] = Field(default_factory=list)

class PosterSection(BaseModel):
    section_id: str
    type: SectionType
    title: str = ''
    content_md: str = ''
    content_html: str = ''
    supplement_html: str = ''
    highlights_items: list[str] = Field(default_factory=list)
    highlights_svg_ref: str = ''
    highlights_region_width: int = 0
    highlights_region_height: int = 0
    column: int = Field(default=1, ge=1, le=3)
    col_span: int = Field(default=1, ge=1, le=3)
    row: int = Field(default=0, ge=0, le=3)
    row_span: int = Field(default=1, ge=1, le=3)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    visual_priority: Literal["P0", "P1", "P2", "P3"] = "P2"

class FigurePlacement(BaseModel):
    figure_id: str
    section_id: str
    width_ratio: float = Field(default=0.9, ge=0.1, le=1.0)
    caption: str = ''

class FormulaDisplay(BaseModel):
    formula_id: str
    section_id: str
    latex: str = ''
    semantic_desc: str = ''

class PosterBlueprint(BaseModel):
    paper_id: str
    poster_title: str = ''
    tagline: str = ''
    authors_str: str = ''
    code_url: str = ''
    # The primary deliverable is a landscape academic poster.  48 x 27 inches
    # matches the aspect ratio used by the visual reference set and gives the
    # renderer a stable, print-friendly canvas to audit.
    width_px: int = 1920
    height_px: int = 1080
    width_mm: int = 1219
    height_mm: int = 686
    sections: list[PosterSection] = Field(default_factory=list)
    figure_placements: list[FigurePlacement] = Field(default_factory=list)
    formula_displays: list[FormulaDisplay] = Field(default_factory=list)
    color_scheme: dict = Field(default_factory=dict)
    story_plan: PosterStoryPlan = Field(default_factory=PosterStoryPlan)
