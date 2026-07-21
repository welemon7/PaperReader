from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------- leaf nodes ----------

class Formula(BaseModel):
    formula_id: str = Field(description="Unique ID, e.g. f-001")
    latex: str = Field(description="Raw LaTeX source of the formula")
    semantic_desc: str = Field(default="", description="Natural-language meaning, filled by LLM later")
    section_id: str = Field(description="Section this formula belongs to")
    label: Optional[str] = Field(default=None, description="\\label{...} if present")


class Figure(BaseModel):
    figure_id: str = Field(description="Unique ID, e.g. fig-001")
    label: Optional[str] = Field(default=None, description="\\label{fig:...} if present")
    caption: str = Field(default="")
    local_path: Optional[str] = Field(default=None, description="Path to extracted image on disk")
    minio_path: Optional[str] = Field(default=None, description="Path inside MinIO bucket")
    width: Optional[str] = Field(default=None, description="\\includegraphics[width=...]")
    section_id: str = Field(description="Section this figure belongs to")


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None


class Reference(BaseModel):
    ref_id: str = Field(description="Unique ID, e.g. ref-001")
    bibkey: str = Field(description="\\bibitem{key} or @inproceedings{key}")
    title: str
    authors: str
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None


# ---------- tree node ----------

class Section(BaseModel):
    section_id: str = Field(description="Unique ID, e.g. sec-001")
    title: str
    level: int = Field(description="1=section, 2=subsection, 3=subsubsection")
    text: str = Field(description="Plain text content (non-LaTeX)")
    raw_latex: str = Field(description="Raw LaTeX for this section body")
    formulas: list[Formula] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    subsections: list[Section] = Field(default_factory=list)


# ---------- root ----------

class PaperDocument(BaseModel):
    paper_id: str = Field(description="Internal UUID or arXiv ID")
    arxiv_id: str
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str = Field(default="")
    sections: list[Section] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list, description="Flattened index of all formulas")
    figures: list[Figure] = Field(default_factory=list, description="Flattened index of all figures")
    references: list[Reference] = Field(default_factory=list)
    raw_markdown: str = Field(default="", description="Full paper as Markdown")
    parsed_at: datetime = Field(default_factory=datetime.now)
    source_dir: str = Field(default="", description="Temporary directory with extracted LaTeX")
