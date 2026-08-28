from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AssetKind = Literal["figure", "chart", "table", "formula"]
AssetAction = Literal["keep", "merge", "remove", "annotate"]


class VisualAssetDecision(BaseModel):
    asset_id: str
    kind: AssetKind
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    readability: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_value: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    action: AssetAction = "keep"
    target_section_id: str = ""
    redundancy_group: str = ""
    reason: str = ""
    message: str = ""
    recommended_width: float = Field(default=0.5, ge=0.0, le=1.0)


class VisualAssetPlan(BaseModel):
    """Small, auditable set of visual assets selected for the poster."""

    figure_decisions: list[VisualAssetDecision] = Field(default_factory=list)
    chart_decisions: list[VisualAssetDecision] = Field(default_factory=list)
    formula_decisions: list[VisualAssetDecision] = Field(default_factory=list)
    selected_figure_ids: list[str] = Field(default_factory=list)
    selected_chart_ids: list[str] = Field(default_factory=list)
    selected_formula_ids: list[str] = Field(default_factory=list)
    redundancy_threshold: float = 0.72
