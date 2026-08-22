from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Contribution(BaseModel):
    text: str = Field(description="One sentence describing the contribution")
    category: Literal["method", "theory", "system", "dataset", "application", "other"] = "other"


class KeyFormula(BaseModel):
    formula_id: str
    latex: str
    semantic_desc: str = Field(description="What this formula means in plain language")


class KeyFigure(BaseModel):
    figure_id: str
    caption: str
    role: str = Field(description="What this figure illustrates, e.g. 'overview', 'architecture', 'result'")


class SelectedTable(BaseModel):
    table_id: str
    role: str = "result"
    row_indices: list[int] = Field(default_factory=list, description="Zero-based row indices from the table data rows")


class FinalTable(BaseModel):
    table_id: str
    caption: str = ""
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    row_groups: list[str] = Field(default_factory=list)
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    row_indices: list[int] = Field(default_factory=list)
    column_indices: list[int] = Field(default_factory=list)
    column_groups: list[list[int]] = Field(default_factory=list)
    notes: str = ""

    @field_validator("datasets", "metrics", "row_groups", mode="before")
    @classmethod
    def ensure_str_list(cls, v):
        if isinstance(v, str):
            items = [item.strip() for item in re.split(r"[,;]\s*", v) if item.strip()]
            return items or ([v.strip()] if v.strip() else [])
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return [str(v).strip()] if v else []


class ExperimentSummary(BaseModel):
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    main_results: str = Field(default="", description="Best results in key numbers")
    takeaways: list[str] = Field(default_factory=list)

    @field_validator('datasets', 'metrics', 'takeaways', mode='before')
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            if ',' in v or ';' in v:
                items = re.split(r'[,;]\s*', v)
                items = [item.strip() for item in items if item.strip()]
                return items if items else [v.strip()]
            else:
                return [v.strip()]
        elif isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        else:
            return [str(v)] if v else []


VisualPriority = Literal["P0", "P1", "P2", "P3"]


class ImportanceItem(BaseModel):
    """A paper claim ranked for poster-scale visual emphasis."""

    text: str = ""
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    role: str = "supporting"


class MethodComponent(ImportanceItem):
    name: str = ""


class ContentImportance(BaseModel):
    """Importance map shared by planning, layout, and visual QA."""

    main_message: ImportanceItem = Field(default_factory=ImportanceItem)
    core_innovation: ImportanceItem = Field(default_factory=ImportanceItem)
    method_components: list[MethodComponent] = Field(default_factory=list)
    supporting_information: list[ImportanceItem] = Field(default_factory=list)

    def priority_for_type(self, section_type: str) -> VisualPriority:
        mapping: dict[str, VisualPriority] = {
            "main_method": "P0",
            "experiments": "P0",
            "key_idea": "P1",
            "method_overview": "P1",
            "motivation": "P2",
            "contributions": "P3",
            "highlights": "P3",
            "project_link": "P3",
            "references": "P3",
            "title": "P0",
        }
        return mapping.get(section_type, "P2")


class PaperAnalysis(BaseModel):
    """Structured analysis of a research paper, used as input for poster planning."""

    paper_id: str
    arxiv_id: str

    # Core
    title_zh: str = Field(default="", description="Chinese translation of the title")
    problem_statement: str = Field(description="The core problem this paper solves")
    contributions: list[Contribution] = Field(default_factory=list)

    # Method
    method_overview: str = Field(description="High-level method description, 2-4 sentences")
    key_formulas: list[KeyFormula] = Field(default_factory=list)
    key_figures: list[KeyFigure] = Field(default_factory=list)
    selected_tables: list[SelectedTable] = Field(default_factory=list)
    final_tables: list[FinalTable] = Field(default_factory=list)

    # Experiments
    experiments: ExperimentSummary | None = None

    # Conclusion
    conclusion: str = Field(default="")

    # Code URL
    code_url: str = Field(default="", description="Project code repository link extracted from the paper")

    # Full text
    full_analysis_md: str = Field(default="", description="Full LLM analysis in Markdown")

    # Poster hierarchy. Kept on the analysis so it survives SQLite round-trips.
    content_importance: ContentImportance = Field(default_factory=ContentImportance)
