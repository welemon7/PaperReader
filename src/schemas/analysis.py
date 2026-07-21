from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


from typing import Union
from pydantic import BaseModel, Field, field_validator
import re


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

    # Experiments
    experiments: ExperimentSummary | None = None

    # Conclusion
    conclusion: str = Field(default="")

    # Code URL
    code_url: str = Field(default="", description="Project code repository link extracted from the paper")

    # Full text
    full_analysis_md: str = Field(default="", description="Full LLM analysis in Markdown")
