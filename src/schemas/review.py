"""100-point review contract for the poster quality gate.

The VLM no longer produces a single opaque 0-10 opinion.  A weighted,
pre-registered contract turns dimension scores into a transparent total and a
pass/fail verdict with per-gate breakdown:

- total score >= 85
- zero hard errors (deterministic layer only, never overridable by a model)
- every dimension >= 60% of its weight
- image-grounded PaperQuiz accuracy >= 80% (checked on the final candidate)

Stopping rules are explicit: only *passed* means passed; the loop may end on
``max_rounds`` or ``stopped_not_passing`` (two consecutive rounds without
effective improvement) and the UI must label those as "未达标" with reasons.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Fixed dimension -> weight map (sums to 100).
DIMENSION_WEIGHTS: dict[str, int] = {
    "layout_hierarchy": 30,
    "readability_overflow": 25,
    "figures_storytelling": 20,
    "content_coverage_facts": 20,
    "color_accessibility": 5,
}

PASS_TOTAL_SCORE = 85.0
PASS_DIM_FRACTION = 0.6  # each dimension >= 60% of its weight
PASS_QA_ACCURACY = 0.8

STOP_REASONS = {
    "passed": "已达标：通过全部门禁",
    "max_rounds": "达到最大轮数，未达标",
    "plateau": "连续多轮无提升，未达标终止",
    "stopped_not_passing": "连续两轮无有效提升，未达标终止",
    "vision_unavailable": "视觉链路不可用（浏览器或视觉模型）",
    "render_error": "渲染失败",
    "unknown": "未知",
}


class ReviewDimensions(BaseModel):
    """0-10 score per dimension (10 = excellent)."""

    layout_hierarchy: float = Field(default=0.0, ge=0.0, le=10.0)
    readability_overflow: float = Field(default=0.0, ge=0.0, le=10.0)
    figures_storytelling: float = Field(default=0.0, ge=0.0, le=10.0)
    content_coverage_facts: float = Field(default=0.0, ge=0.0, le=10.0)
    color_accessibility: float = Field(default=0.0, ge=0.0, le=10.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "layout_hierarchy": self.layout_hierarchy,
            "readability_overflow": self.readability_overflow,
            "figures_storytelling": self.figures_storytelling,
            "content_coverage_facts": self.content_coverage_facts,
            "color_accessibility": self.color_accessibility,
        }

    def weighted_total(self) -> float:
        """Total on a 0-100 scale using the pre-registered weights."""
        total = 0.0
        for name, weight in DIMENSION_WEIGHTS.items():
            total += getattr(self, name, 0.0) / 10.0 * weight
        return round(total, 1)

    def min_dimension_fraction(self) -> float:
        """Worst dimension score as a fraction of its weight (>=1.0 is perfect)."""
        fracs = [
            getattr(self, name, 0.0) / 10.0 * weight
            for name, weight in DIMENSION_WEIGHTS.items()
        ]
        return min(fracs) / max(1.0, max(DIMENSION_WEIGHTS.values())) * 10.0 if fracs else 0.0


class DeterministicCheck(BaseModel):
    """One deterministic (model-free) audit check result."""

    name: str
    passed: bool
    severity: Literal["error", "warning"] = "warning"
    detail: str = ""
    data: dict = Field(default_factory=dict)


class DeterministicAudit(BaseModel):
    """Full result of the browser-geometry audit (no model involved)."""

    available: bool = False
    reason: str = ""
    checks: list[DeterministicCheck] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def add(self, check: DeterministicCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            if check.severity == "error":
                self.hard_failures.append(check.name)
            else:
                self.warnings.append(check.name)

    def has_hard_failures(self) -> bool:
        return bool(self.hard_failures)

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
            "checks": [c.model_dump() for c in self.checks],
        }


class ContractGate(BaseModel):
    """One gate of the pass contract with its measured value."""

    name: str
    passed: bool
    required: str = ""
    actual: str = ""


class PosterVerdict(BaseModel):
    """Computed pass/fail verdict with per-gate breakdown."""

    passed: bool = False
    total_score: float = 0.0
    hard_error_count: int = 0
    gates: list[ContractGate] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def compute_verdict(
    dimensions: ReviewDimensions,
    hard_error_count: int,
    qa_accuracy: Optional[float] = None,
    total_threshold: float = PASS_TOTAL_SCORE,
    dim_fraction: float = PASS_DIM_FRACTION,
    qa_threshold: float = PASS_QA_ACCURACY,
) -> PosterVerdict:
    """Evaluate the pre-registered pass contract.

    Args:
        dimensions: VLM dimension scores (0-10 each).
        hard_error_count: number of deterministic hard failures.
        qa_accuracy: image-grounded PaperQuiz accuracy (0-1); None = not run.
        total_threshold: minimum weighted total (default 85).
        dim_fraction: minimum fraction of each dimension's weight (default 0.6).
        qa_threshold: minimum image QA accuracy (default 0.8).
    """
    total = dimensions.weighted_total()
    gates: list[ContractGate] = [
        ContractGate(
            name="total_score",
            passed=total >= total_threshold,
            required=f">= {total_threshold:.0f}",
            actual=f"{total:.1f}",
        ),
        ContractGate(
            name="no_hard_errors",
            passed=hard_error_count == 0,
            required="0",
            actual=str(hard_error_count),
        ),
        ContractGate(
            name="dimension_minimums",
            passed=_dimensions_pass(dimensions, dim_fraction),
            required=f"each >= {dim_fraction:.0%} of weight",
            actual=_dimension_minimums_str(dimensions),
        ),
    ]
    if qa_accuracy is not None:
        gates.append(ContractGate(
            name="image_qa",
            passed=qa_accuracy >= qa_threshold,
            required=f">= {qa_threshold:.0%}",
            actual=f"{qa_accuracy:.0%}",
        ))
    else:
        gates.append(ContractGate(name="image_qa", passed=False, required="run on final candidate", actual="not run"))

    passed = all(g.passed for g in gates)
    reasons = [g.name for g in gates if not g.passed]
    return PosterVerdict(
        passed=passed,
        total_score=total,
        hard_error_count=hard_error_count,
        gates=gates,
        reasons=reasons,
    )


def _dimensions_pass(dimensions: ReviewDimensions, dim_fraction: float = PASS_DIM_FRACTION) -> bool:
    for name, weight in DIMENSION_WEIGHTS.items():
        if getattr(dimensions, name, 0.0) / 10.0 * weight < dim_fraction * weight:
            return False
    return True


def _dimension_minimums_str(dimensions: ReviewDimensions) -> str:
    parts = []
    for name, weight in DIMENSION_WEIGHTS.items():
        value = getattr(dimensions, name, 0.0)
        parts.append(f"{name}={value:.1f}")
    return ", ".join(parts)
