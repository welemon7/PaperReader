from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


NodeType = Literal[
    "root",
    "container",
    "title",
    "text",
    "figure",
    "formula",
    "callout",
    "spacer",
]


class LayoutConstraints(BaseModel):
    min_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    max_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: int = Field(default=0, ge=0)


class LayoutNode(BaseModel):
    node_id: str
    node_type: NodeType
    title: str = ""
    content_md: str = ""
    figure_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)
    reading_order: int = Field(default=0, ge=0)
    space_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    section_col_span: int = Field(default=1, ge=1, le=3)
    section_row_span: int = Field(default=1, ge=1, le=3)
    figure_width_ratio: float = Field(default=0.9, ge=0.1, le=1.0)
    constraints: LayoutConstraints = Field(default_factory=LayoutConstraints)
    notes: str = ""


class LayoutTree(BaseModel):
    paper_id: str
    arxiv_id: str
    title: str = ""
    required_items: list[str] = Field(default_factory=list)
    nodes: list[LayoutNode] = Field(default_factory=list)
    root_id: str = "root"
    reading_path: list[str] = Field(default_factory=list)
    layout_notes: list[str] = Field(default_factory=list)


class PosterComment(BaseModel):
    issue: str
    severity: Literal["error", "warning", "info"] = "warning"
    target: str = ""
    suggestion: str = ""
    action: Literal["resize", "reflow", "rewrite", "condense", "replace_figure", "remove", "supplement", "keep"] = "rewrite"


class PosterReview(BaseModel):
    model_config = ConfigDict(extra="allow")

    quality_score: int = Field(default=0, ge=0, le=10)
    needs_improvement: bool = True
    issues: list[PosterComment] = Field(default_factory=list)
    summary: str = ""
    layout_feedback: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    # Non-negotiable render failures found without asking a model (for example
    # broken figures or clipped text).  A high VLM score must never override
    # these failures.
    hard_failures: list[str] = Field(default_factory=list)
    deterministic_checks: dict[str, object] = Field(default_factory=dict)


class EvaluationQuestion(BaseModel):
    question_id: str
    question: str
    answer: str
    evidence: list[str] = Field(default_factory=list)
    category: str = "core"


class PosterQAEval(BaseModel):
    paper_id: str
    arxiv_id: str
    questions: list[EvaluationQuestion] = Field(default_factory=list)
    poster_answers: list[str] = Field(default_factory=list)
    correct_count: int = 0
    total_count: int = 0
    accuracy: float = 0.0
    coverage: float = 0.0
    recall: float = 0.0
    visual_score: int = 0
    qa_score: int = 0
    summary: str = ""
