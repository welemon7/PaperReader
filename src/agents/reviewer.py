"""VLM review layer for the 100-point contract.

Wraps the multimodal call with a retry, extracts the five pre-registered
dimension scores (0-10 each), and returns structured issues.  Everything here
is model-facing; deterministic geometry lives in ``src.visual.audit`` and the
pass contract in ``src.schemas.review``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.llm.multimodal_client import multimodal_analyze_labeled
from src.schemas.review import DIMENSION_WEIGHTS, ReviewDimensions

logger = logging.getLogger(__name__)

_DIMENSION_ALIASES: dict[str, str] = {
    "layout_hierarchy": "layout_hierarchy",
    "layout": "layout_hierarchy",
    "layout_and_hierarchy": "layout_hierarchy",
    "hierarchy": "layout_hierarchy",
    "readability_overflow": "readability_overflow",
    "readability": "readability_overflow",
    "readability_overflow_25": "readability_overflow",
    "overflow": "readability_overflow",
    "figures_storytelling": "figures_storytelling",
    "figures": "figures_storytelling",
    "figure": "figures_storytelling",
    "figures_visual_storytelling": "figures_storytelling",
    "visual_storytelling": "figures_storytelling",
    "content_coverage_facts": "content_coverage_facts",
    "content_coverage": "content_coverage_facts",
    "content": "content_coverage_facts",
    "coverage": "content_coverage_facts",
    "factual_consistency": "content_coverage_facts",
    "color_accessibility": "color_accessibility",
    "color": "color_accessibility",
    "accessibility": "color_accessibility",
}


def build_review_system_prompt() -> str:
    """The strict 100-point VLM judge prompt (weights pre-registered)."""
    weight_lines = "\n".join(
        f"- {name}: {weight}/100"
        for name, weight in DIMENSION_WEIGHTS.items()
    )
    return f"""You are a strict scientific poster reviewer (VLM judge).

You are shown the rendered poster image at true print resolution, a grid
overlay showing section boundaries, per-section crops, and figure-region
crops. Section ids follow the pattern "sec-motivation", "sec-method-overview",
"sec-key-idea", "sec-main-method", "sec-experiments", "sec-contributions",
"sec-highlights", "sec-project".

Score the poster on these five dimensions (each 0-10, 10 = excellent):

{weight_lines}

- layout_hierarchy: global balance, alignment, column structure, whitespace,
  visual hierarchy and reading order.
- readability_overflow: text legibility, font sizes, clipping, cut-off text,
  overlapping elements, line density.
- figures_storytelling: figure relevance, size, cropping, chart readability,
  how well figures tell the method/result story. Captions are intentionally
  hidden under figures; do not penalize their absence.
- content_coverage_facts: whether problem, method, and main results are
  visibly present and factually consistent (numbers, formulas, citations).
- color_accessibility: palette harmony, contrast, section differentiation,
  color-blind-safe choices.

Return JSON ONLY with this exact structure:
{{
  "dimension_scores": {{
    "layout_hierarchy": <0-10>, "readability_overflow": <0-10>,
    "figures_storytelling": <0-10>, "content_coverage_facts": <0-10>,
    "color_accessibility": <0-10>
  }},
  "needs_improvement": <true|false>,
  "summary": "<one or two sentences>",
  "issues": [
    {{
      "section_id": "<sec-...> or empty for global>",
      "description": "<what is wrong and where>",
      "severity": "error|warning|info",
      "action": "condense|resize|reflow|replace_figure|remove|rewrite|keep",
      "suggestion": "<concrete fix>"
    }}
  ]
}}

Rules:
- severity "error" is reserved for visible breakage (clipped text, empty
  panel, broken figure, unreadable font). Use "warning" for aesthetic or
  density problems, "info" for polish suggestions.
- Every issue must carry a section_id (use the ids above) and an action from
  the allowed set. Do not invent issues; only report what you can see.
- If the poster already satisfies the contract (no errors, balanced layout,
  legible text, clear figures, correct facts), set needs_improvement=false and
  return an empty issues list. Do not pad with trivia.
- Do NOT comment on figure captions being absent under images — that is
  intended design.
"""


def vlm_review_images(
    images: list[tuple[str, str]],
    user_text: str = "",
    model: Optional[str] = None,
    max_attempts: int = 2,
) -> Optional[dict[str, Any]]:
    """Call the VLM with one automatic retry; returns the parsed JSON dict."""
    system_prompt = build_review_system_prompt()
    last: Optional[dict[str, Any]] = None
    for attempt in range(1, max_attempts + 1):
        raw = multimodal_analyze_labeled(system_prompt, images, user_text=user_text, model=model)
        if raw is None:
            logger.warning("VLM review attempt %d/%d returned nothing", attempt, max_attempts)
            continue
        if isinstance(raw, dict) and ("dimension_scores" in raw or "issues" in raw):
            return raw
        last = raw
        logger.warning("VLM review attempt %d/%d had unexpected shape", attempt, max_attempts)
    return last


def extract_review_dimensions(raw: dict[str, Any]) -> ReviewDimensions:
    """Tolerant extraction of the five dimension scores from a VLM response.

    Accepts the canonical keys as well as legacy aliases; missing dimensions
    score 0 so the contract can never pass silently on incomplete input.
    """
    scores: dict[str, float] = {}
    raw_dims = raw.get("dimension_scores") or {}
    if isinstance(raw_dims, dict):
        for key, value in raw_dims.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            canonical = _DIMENSION_ALIASES.get(str(key).strip().lower())
            if canonical and 0 <= numeric <= 10:
                scores[canonical] = numeric
    # Only trust the extracted dimensions when most of the five are present;
    # a partial answer (e.g. legacy {"layout": 6, "typography": 5}) must not
    # silently zero out the remaining dimensions.
    if len(scores) < 3:
        scores = {}
    # Legacy "quality_score" (0-10 or 0-100) as a fallback for every dimension.
    if not scores:
        fallback = raw.get("quality_score")
        try:
            numeric = float(fallback)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric > 10:
            numeric = numeric / 10.0
        numeric = max(0.0, min(10.0, numeric))
        if numeric:
            scores = {name: numeric for name in DIMENSION_WEIGHTS}
    return ReviewDimensions(**{name: scores.get(name, 0.0) for name in DIMENSION_WEIGHTS})
