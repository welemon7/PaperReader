from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.poster_v2 import PosterComment


class HarnessConfig(BaseModel):
    """Configuration for the visual-review harness loop."""

    threshold: int = Field(default=9, ge=1, le=10, description="Quality score (0-10) at which the loop passes.")
    max_rounds: int = Field(default=5, ge=1, le=20, description="Maximum review/re-render rounds.")
    zoom_crops: bool = Field(default=True, description="Also capture per-section zoom crops for the VLM.")
    max_crops: int = Field(default=3, ge=0, le=7, description="Max number of section crops sent to the VLM.")
    enable_qa_eval: bool = Field(default=True, description="Run PaperQuiz-style content QA after the loop.")
    qa_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum image-grounded PaperQuiz accuracy required to pass.")
    vision_model: Optional[str] = Field(default=None, description="Optional vision model override (None = unified config).")


class HarnessRound(BaseModel):
    """One review/re-render round inside the harness loop."""

    round_no: int = Field(ge=1)
    quality_score: int = Field(ge=0, le=10)
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
