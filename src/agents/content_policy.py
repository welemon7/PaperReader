"""Poster content policy: per-section word budgets.

Based on the Paper2Poster copy-density guidance (poster-defaults.md): a usable
academic poster carries ~250-500 words of visible body copy, bullet-first,
with at most ~6 major sections. These budgets are shared by the planner
(content builders) and the harness (density checks / condense rewrites).
"""

from __future__ import annotations

# Total visible body-word budget (excludes title/authors/project link).
TOTAL_WORD_BUDGET = 500
# Warn threshold used by the harness density check (soft cap).
TOTAL_WORD_SOFT_CAP = 450

# Per-section word budgets, keyed by PosterSection.type.
# Counts visible words including math tokens; main_method includes the
# item-details results table (a table is legitimate poster content, not prose).
SECTION_WORD_BUDGET: dict[str, int] = {
    "motivation": 75,
    "method_overview": 75,
    "key_idea": 70,
    "main_method": 105,  # 正文精简，结果表（~40 词）另计在内
    "experiments": 100,
    "contributions": 80,  # 3-4 bullets x <= 20 words
    "highlights": 80,  # 最多 4 条 bullet x <= 20 words
    "project_link": 15,
}

# Per-bullet budget inside bullet-driven sections (contributions/highlights).
BULLET_WORD_BUDGET = 20

# How many contribution / highlight bullets to render at most.
MAX_CONTRIBUTION_BULLETS = 4
MAX_HIGHLIGHT_BULLETS = 4


def section_budget(section_type: str) -> int:
    """Return the word budget for a section type (falls back to a default)."""
    return SECTION_WORD_BUDGET.get(section_type, 60)


def count_words(text: str | None) -> int:
    """Count visible words in markdown/HTML-ish text (tags & bullet markers stripped)."""
    if not text:
        return 0
    import re

    stripped = re.sub(r"<[^>]+>", " ", text)
    stripped = re.sub(r"(^|\n)\s*[-*]\s+", r"\1", stripped)
    return len(stripped.split())


def over_budget(section_type: str, text: str | None, tolerance: int = 0) -> bool:
    """True when a section's word count exceeds its budget (+tolerance)."""
    return count_words(text) > section_budget(section_type) + tolerance


def trim_to_budget(text: str, budget: int) -> str:
    """Deterministically trim text to a word budget (sentence/bullet aware)."""
    words = (text or "").split()
    if len(words) <= budget:
        return text or ""
    # Prefer cutting to whole sentences: take the first sentences until the
    # budget would be exceeded, then hard-cut the remainder.
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    used = 0
    for part in parts:
        n = len(part.split())
        if used + n > budget and kept:
            break
        kept.append(part)
        used += n
    if kept and used <= budget:
        return " ".join(kept)
    return " ".join(words[:budget])
