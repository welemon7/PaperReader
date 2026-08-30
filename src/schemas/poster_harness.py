from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ConfigDict

from src.schemas.poster_v2 import PosterComment


class HarnessConfig(BaseModel):
    """Configuration retained for the Preliminary Supplement API."""

    model_config = ConfigDict(extra="allow")

    threshold: int = Field(default=9, ge=1, le=10)
    max_rounds: int = Field(default=1, ge=1, le=20)
    zoom_crops: bool = Field(default=True)
    # Retained for API compatibility; Preliminary Supplement does not send
    # section crops to a vision model.
    max_crops: int = Field(default=0, ge=0, le=7)
    enable_qa_eval: bool = Field(default=False)
    qa_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    vision_model: Optional[str] = Field(default=None)


class HarnessRound(BaseModel):
    """One review/re-render round inside the harness loop."""

    round_no: int = Field(ge=1)
    quality_score: int = Field(ge=0, le=10)
    total_score: float = 0.0
    verdict: dict[str, object] = Field(default_factory=dict)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    hard_failures: list[str] = Field(default_factory=list)
    deterministic_checks: dict[str, object] = Field(default_factory=dict)
    needs_improvement: bool = True
    issues: list[PosterComment] = Field(default_factory=list)
    summary: str = ""
    applied_actions: list[str] = Field(default_factory=list)
    png_path: str = ""
    html_path: str = ""
    review_path: str = ""
    grid_png: str = ""
    diff_png: str = ""
    section_crops: dict[str, str] = Field(default_factory=dict)
    figure_crops: dict[str, str] = Field(default_factory=dict)
    captured_at: str = ""


class HarnessResult(BaseModel):
    """Final outcome of the harness loop."""

    passed: bool = False
    stop_reason: str = "unknown"
    rounds: list[HarnessRound] = Field(default_factory=list)
    best_round_no: int = 0
    best_score: int = 0
    final_html: str = ""
    final_png: str = ""
    report_path: str = ""
    fallback: bool = False
    fallback_reason: str = ""
    qa_eval_path: str = ""
    total_rounds: int = 0

    @property
    def stop_label(self) -> str:
        from src.schemas.review import STOP_REASONS

        return STOP_REASONS.get(self.stop_reason, self.stop_reason)

    @property
    def best_total(self) -> float:
        totals = [getattr(r, "total_score", 0.0) for r in self.rounds if getattr(r, "total_score", 0.0)]
        if totals:
            return max(totals)
        return float(self.best_score)
