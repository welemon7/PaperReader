"""Controlled patch mapping and application on the scene graph.

Reviewer issues are translated into a restricted set of ``ScenePatch`` objects
(condense_text / resize_figure / reflow_panel / replace_figure /
remove_element / adjust_font).  Patches mutate the scene; the harness re-runs
the deterministic solver + renderer and the browser audit, and rolls back when
hard errors regress.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.content_policy import count_words, section_budget, trim_to_budget
from src.layout.scene import (
    PosterScene,
    ScenePatch,
    make_condense_patch,
    make_font_patch,
    make_remove_element_patch,
    make_replace_figure_patch,
    make_resize_figure_patch,
    make_reflow_panel_patch,
)
from src.llm.client import LLMClient, LLMError
from src.schemas.poster_v2 import PosterReview

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = (
    "You are an expert scientific poster editor. Rewrite the given section content to fix the "
    "reported issue. Keep ALL factual claims, numbers, formulas, citations and technical meaning "
    "intact. Prefer concise bullet points. Output ONLY the rewritten markdown content — no "
    "explanations, no code fences, no JSON."
)


def patches_from_review(scene: PosterScene, review: PosterReview) -> list[ScenePatch]:
    """Translate a review's issues into controlled scene patches."""
    patches: list[ScenePatch] = []
    seen: set[str] = set()

    for comment in review.issues:
        target = comment.target.strip()
        panel = scene.panel(target)
        if panel is None:
            # tolerate titles / types as targets
            panel = scene.panel_by_type(target)
        if panel is None:
            continue
        key = f"{comment.action}:{panel.panel_id}"
        if key in seen:
            continue
        seen.add(key)
        issue = comment.issue.lower()

        if comment.action == "condense":
            budget = section_budget(panel.panel_type)
            patches.append(make_condense_patch(panel.panel_id, budget, reason=comment.issue[:80]))
        elif comment.action == "rewrite":
            budget = section_budget(panel.panel_type)
            patches.append(make_condense_patch(panel.panel_id, budget, reason=comment.issue[:80]))
        elif comment.action in ("resize", "reflow"):
            if "figure" in issue or "image" in issue:
                patches.append(make_resize_figure_patch(panel.panel_id, 0.4, reason=comment.issue[:80]))
            elif "dense" in issue or "overflow" in issue or "text" in issue or "clip" in issue:
                budget = section_budget(panel.panel_type)
                patches.append(make_condense_patch(panel.panel_id, budget, reason=comment.issue[:80]))
            else:
                patches.append(make_reflow_panel_patch(panel.panel_id, grow=False, reason=comment.issue[:80]))
        elif comment.action == "replace_figure":
            patches.append(make_replace_figure_patch(panel.panel_id, "", reason=comment.issue[:80]))
        elif comment.action == "remove":
            patches.append(make_remove_element_patch(panel.panel_id, "figure", reason=comment.issue[:80]))
        elif comment.action == "keep":
            continue

    # Deterministic hints from the audit (e.g. body text too small).
    for hint in review.deterministic_checks.get("checks", []):
        if not isinstance(hint, dict):
            continue
        if hint.get("name") == "min_body_font" and not hint.get("passed"):
            for panel in scene.panels:
                if any(e.kind == "text" for e in panel.elements):
                    patches.append(make_font_patch(panel.panel_id, 1.12, reason="body font too small"))

    return patches


def apply_patches(scene: PosterScene, patches: list[ScenePatch], llm: Optional[LLMClient] = None) -> list[str]:
    """Apply patches to the scene; returns human-readable applied-action strings."""
    applied: list[str] = []
    for patch in patches:
        panel = scene.panel(patch.target) or scene.panel_by_type(patch.target)
        if panel is None:
            continue
        try:
            if patch.kind == "condense_text":
                desc = _condense_panel(panel, patch, llm)
                if desc:
                    applied.append(desc)
            elif patch.kind == "resize_figure":
                desc = _resize_figure(panel, patch)
                if desc:
                    applied.append(desc)
            elif patch.kind == "reflow_panel":
                desc = _reflow_panel(panel, patch)
                if desc:
                    applied.append(desc)
            elif patch.kind == "replace_figure":
                desc = _replace_figure(panel, patch)
                if desc:
                    applied.append(desc)
            elif patch.kind == "remove_element":
                desc = _remove_element(panel, patch)
                if desc:
                    applied.append(desc)
            elif patch.kind == "adjust_font":
                desc = _adjust_font(panel, patch)
                if desc:
                    applied.append(desc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Patch %s failed: %s", patch.describe(), exc)
    return applied


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


def _condense_panel(panel, patch: ScenePatch, llm: Optional[LLMClient]) -> Optional[str]:
    text_els = [e for e in panel.elements if e.kind == "text"]
    if not text_els:
        return None
    max_words = int(patch.params.get("max_words") or section_budget(panel.panel_type))
    for el in text_els:
        current = el.content_md or ""
        if not current.strip():
            continue
        if llm is not None and len(current.split()) > max_words:
            try:
                user = (
                    f"Section title: {panel.title}\n\nCurrent content:\n{current}\n\n"
                    f"Reported issue: {patch.reason or 'too dense'}\n"
                    f"Length limit: output at most {max_words} words.\n\n"
                    "Return only the rewritten markdown content."
                )
                new_content = _strip_fences(llm.chat(system=_REWRITE_SYSTEM_PROMPT, user=user))
                if new_content and len(new_content) > 20:
                    el.content_md = trim_to_budget(new_content, max_words)
                    return f"condense {panel.panel_id} (LLM, <= {max_words} words)"
            except LLMError as exc:
                logger.warning("Scene condense LLM failed: %s", exc)
        old_words = count_words(current)
        el.content_md = trim_to_budget(current, max_words)
        if count_words(el.content_md) < old_words:
            return f"condense {panel.panel_id} (trim to {max_words})"
    return None


def _resize_figure(panel, patch: ScenePatch) -> Optional[str]:
    figs = [e for e in panel.elements if e.kind == "figure"]
    if not figs:
        return None
    hint = float(patch.params.get("box_hint") or 0.4)
    for fig in figs:
        fig.box_hint = max(0.15, min(0.7, hint))
    return f"resize_figure {panel.panel_id} (box_hint={hint})"


def _reflow_panel(panel, patch: ScenePatch) -> Optional[str]:
    grow = bool(patch.params.get("grow"))
    if grow:
        panel.constraints.max_ratio = min(1.0, panel.constraints.max_ratio + 0.15)
        panel.constraints.priority += 1
        return f"reflow {panel.panel_id} (grow)"
    panel.constraints.max_ratio = max(0.5, panel.constraints.max_ratio - 0.1)
    return f"reflow {panel.panel_id} (shrink)"


def _replace_figure(panel, patch: ScenePatch) -> Optional[str]:
    figs = [e for e in panel.elements if e.kind == "figure"]
    src = str(patch.params.get("figure_src") or "")
    if not figs:
        return None
    if src:
        fig = figs[0]
        fig.figure_src = src
        fig.figure_aspect = float(patch.params.get("figure_aspect") or fig.figure_aspect or 1.4)
        return f"replace_figure {panel.panel_id} -> {src}"
    # No spare asset: shrink the existing figure instead of leaving an empty slot.
    figs[0].box_hint = max(0.25, figs[0].box_hint * 0.8)
    return f"replace_figure {panel.panel_id} (no spare asset; shrunk)"

def _remove_element(panel, patch: ScenePatch) -> Optional[str]:
    kind = str(patch.params.get("element_kind") or "figure")
    for idx, el in enumerate(panel.elements):
        if el.kind == kind:
            del panel.elements[idx]
            return f"remove {kind} in {panel.panel_id}"
    return None


def _adjust_font(panel, patch: ScenePatch) -> Optional[str]:
    """Restore shrunken body text toward 1.0 — never above it.

    Upscaling past 1.0 re-introduces clipping in size-limited panels, so the
    patch is a no-op for text already at or above the default size.
    """
    target = min(1.0, float(patch.params.get("font_scale") or 1.0))
    changed = False
    for el in panel.elements:
        if el.kind == "text" and el.font_scale < 0.95 and el.font_scale < target:
            el.font_scale = target
            changed = True
    return f"adjust_font {panel.panel_id} (restore to x{target})" if changed else None
