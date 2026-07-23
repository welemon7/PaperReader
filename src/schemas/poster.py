from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

SectionType = Literal['title', 'motivation', 'method_overview', 'key_idea',
    'main_method', 'experiments', 'contributions', 'highlights', 'project_link', 'references']

class PosterSection(BaseModel):
    section_id: str
    type: SectionType
    title: str = ''
    content_md: str = ''
    content_html: str = ''
    column: int = Field(default=1, ge=1, le=3)
    col_span: int = Field(default=1, ge=1, le=3)
    row: int = Field(default=0, ge=0, le=3)
    row_span: int = Field(default=1, ge=1, le=3)

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
    authors_str: str = ''
    code_url: str = ''
    width_px: int = 1200
    height_px: int = 1697
    width_mm: int = 841
    height_mm: int = 1189
    sections: list[PosterSection] = Field(default_factory=list)
    figure_placements: list[FigurePlacement] = Field(default_factory=list)
    formula_displays: list[FormulaDisplay] = Field(default_factory=list)
    color_scheme: dict = Field(default_factory=dict)
