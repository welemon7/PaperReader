from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

Severity = Literal["error", "warning", "info"]
Category = Literal[
    "missing_contribution", "incorrect_method",
    "missing_experiment", "formatting", "clarity", "completeness",
]

class ValidationIssue(BaseModel):
    severity: Severity
    category: Category
    description: str
    location: str = ""
    suggestion: str = ""

class PosterValidation(BaseModel):
    paper_id: str
    arxiv_id: str
    scores: dict = Field(default_factory=lambda: {
        "coverage": 0, "accuracy": 0, "clarity": 0, "completeness": 0,
    })
    issues: list[ValidationIssue] = Field(default_factory=list)
    summary: str = ""
    validated_at: datetime = Field(default_factory=datetime.now)