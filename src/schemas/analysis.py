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
        """统一将字符串转换为列表"""
        if isinstance(v, str):
            # 如果字符串包含中文顿号、逗号或英文逗号，按这些分隔符分割
            # 否则将整个字符串作为列表的一个元素
            if '、' in v or '，' in v or ',' in v or '；' in v or ';' in v:
                # 按常见分隔符分割
                items = re.split(r'[、，,；;]\s*', v)
                items = [item.strip() for item in items if item.strip()]
                return items if items else [v.strip()]
            else:
                # 单个值，直接放入列表
                return [v.strip()]
        elif isinstance(v, list):
            # 如果已经是列表，确保每个元素都是字符串
            return [str(item).strip() for item in v if str(item).strip()]
        else:
            # 其他情况（None, int等），转为字符串并放入列表
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

    # Full text
    full_analysis_md: str = Field(default="", description="Full LLM analysis in Markdown")
