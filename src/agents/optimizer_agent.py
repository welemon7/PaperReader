from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from src.agents.poster_planner import generate_blueprint
from src.config import settings
from src.llm.client import LLMClient, LLMError
from src.llm.multimodal_client import multimodal_analyze, capture_poster
from src.renderers.html_renderer import HtmlPosterRenderer
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import PosterBlueprint, PosterSection
from src.storage.sqlite import PaperDatabase
from src.utils.figure_assets import copy_or_rasterize_asset, resolve_figure_source, sanitize_asset_name
from src.utils.output_paths import resolve_paper_output_dir

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a scientific poster optimization expert. "
    "Review a research poster blueprint and its rendered screenshot. "
    "Think like a strict scientific reviewer: inspect the poster content, layout, figure coverage, and factual alignment with the paper. "
    "Evaluate on: coverage, clarity, completeness, accuracy, visual hierarchy, and paper fidelity. "
    "Treat the screenshot as the primary source of truth for layout issues.\n\n"
    "Return JSON with:\n"
    '- "quality_score": int (1-10)\n'
    '- "issues": list of {"severity": str, "description": str}\n'
    '- "suggestions": dict of section_id -> {"content_md": "improved text"}\n'
    '- "layout": dict of section_id -> {"row": int, "column": int, "col_span": int}\n'
    '- "figure_selection": list of {"keep": bool, "figure_id": str, "section_id": str, "width_ratio": float, "caption": str}\n'
    '- "needs_improvement": bool (false if quality >= 8)\n\n'
    "Rules:\n"
    "- Keep core method/overview figures, especially framework and introduction figures.\n"
    "- Keep result/comparison figures in the main result area or core summary area.\n"
    "- Prefer at most 4 figures total.\n"
    "- Do not invent facts not supported by the paper.\n"
    "- Keep poster-facing text in English and close to the paper wording.\n"
    "- If Highlights are weak, rewrite them from contributions and experiment takeaways.\n"
    "- If the poster is already strong, return minimal changes and set needs_improvement=false.\n"
    "- Optimize global whitespace, table height, figure crop fit, and figure relevance from the screenshot alone."
)


def _regenerate_figures(blueprint, doc, output_dir):
    """Keep the figure set stable during optimization.

    The earlier version added every cached PDF as a new placement, which made the
    poster noisy and inconsistent across render passes. Optimization should only
    refine the existing figure plan.
    """
    return blueprint, doc, []


def _poster_vision_provider() -> str:
    provider = (os.getenv("POSTER_VISION_PROVIDER") or settings.poster_vision_provider or "agnes").lower()
    return provider if provider in {"agnes", "gemini", "openai"} else "agnes"


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])(?:\d+\.\d+|\d+)(?:%|x)?", text or ""))


def _extract_formula_tokens(text: str) -> set[str]:
    tokens = set()
    for match in re.findall(r"\$\$(.+?)\$\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", text or "", flags=re.DOTALL):
        expr = next((m for m in match if m), "")
        if expr:
            tokens.add(re.sub(r"\s+", " ", expr).strip())
    return tokens


def _build_core_figure_assets(doc: PaperDocument, analysis: PaperAnalysis, output_dir: Path, limit: int = 4) -> list[Path]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    assets: list[Path] = []
    seen: set[str] = set()
    figures = list(doc.figures)
    if analysis.key_figures:
        wanted = {f.figure_id for f in analysis.key_figures}
        figures = [f for f in figures if f.figure_id in wanted] + [f for f in figures if f.figure_id not in wanted]

    for fig in figures:
        if len(assets) >= limit:
            break
        src = resolve_figure_source(fig.local_path or fig.minio_path, doc.source_dir)
        if not src:
            continue
        key = fig.figure_id or sanitize_asset_name(src.stem)
        if key in seen:
            continue
        prepared = copy_or_rasterize_asset(src, fig_dir, key)
        if prepared and prepared.exists():
            assets.append(prepared)
            seen.add(key)
    return assets


def _generate_result_chart(analysis, output_dir, chart_data=None):
    """Generate a matplotlib chart from experiment results."""
    from pathlib import Path
    if chart_data and chart_data.get("labels") and chart_data.get("values"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = chart_data["labels"]
            values = [float(v) for v in chart_data["values"]]
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = ["#1a5276", "#2980b9", "#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
            bars = ax.bar(labels, values, color=colors[:len(labels)])
            ax.set_ylabel(chart_data.get("ylabel", "Score"))
            ax.set_title(chart_data.get("title", "Experimental Results"), fontsize=12)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                        str(val), ha="center", va="bottom", fontsize=9)
            chart_dir = Path(output_dir) / "figures"
            chart_dir.mkdir(parents=True, exist_ok=True)
            chart_path = chart_dir / "results_chart.png"
            plt.savefig(str(chart_path), dpi=200, bbox_inches="tight")
            plt.close()
            return str(chart_path)
        except:
            pass
    return None


_VISION_PROMPT = (
    "You are a scientific poster design expert. Review the poster image and the attached figure images. "
    "Return JSON with:\n"
    '- "quality_score": int (1-10)\n'
    '- "figure_selection": list of {"keep": bool, "figure_id": str, "section_id": str, "width_ratio": float, "caption": str}\n'
    "  - For each figure, decide if it belongs in the poster. Keep only essential ones (architecture, core results).\n"
    '  - Assign to correct section (sec-main-method, sec-experiments, etc.)\n'
    '  - Set width_ratio based on importance (0.5 for smaller, 1.0 for full width)\n'
    '- "layout_feedback": list of {"section": str, "suggestion": str}\n'
    '- "issues": list of {"severity": str, "description": str}\n'
    '- "needs_improvement": bool\n'
    "Be critical: 2-4 figures max. Remove redundant ones. Keep the visual hierarchy clean and poster-like."
)

_AGNES_REVIEW_PROMPT = (
    "You are a strict scientific poster reviewer. Compare the poster screenshot and core figures against the paper summary. "
    "Reject any suggestion that changes formulas, numeric results, or factual claims without exact evidence. "
    "Output JSON with: quality_score, issues, figure_selection, layout_feedback, and needs_improvement. "
    "For each issue include severity and description. Treat the screenshot as the primary layout source."
)


def _apply_vision_feedback(blueprint, doc, vision_result, output_dir):
    """Apply multimodal vision analysis results to blueprint."""
    from src.schemas.poster import FigurePlacement
    fig_sel = vision_result.get("figure_selection")
    if fig_sel and isinstance(fig_sel, list):
        kept_ids = {f["figure_id"] for f in fig_sel if f.get("keep")}
        if kept_ids:
            before = len(blueprint.figure_placements)
            blueprint.figure_placements = [fp for fp in blueprint.figure_placements if fp.figure_id in kept_ids]
            for sel in fig_sel:
                if sel.get("keep"):
                    fid = sel["figure_id"]
                    for fp in blueprint.figure_placements:
                        if fp.figure_id == fid:
                            fp.section_id = sel.get("section_id", fp.section_id)
                            fp.width_ratio = sel.get("width_ratio", fp.width_ratio)
                            if sel.get("caption"):
                                fp.caption = sel["caption"]
            logger.info("Vision: %d -> %d figures (kept %d)", before, len(blueprint.figure_placements), len(kept_ids))
    issues = vision_result.get("issues", [])
    for iss in issues:
        logger.info("Vision issue [%s]: %s", iss.get("severity", "?"), iss.get("description", ""))


def _apply_structured_quality_gate(
    blueprint: PosterBlueprint,
    doc: PaperDocument,
    analysis: PaperAnalysis,
    review: dict[str, Any],
) -> dict[str, Any]:
    safe_review = dict(review or {})
    suggestions = safe_review.get("suggestions", {}) if isinstance(safe_review, dict) else {}
    if isinstance(suggestions, dict):
        numeric_sources = [doc.raw_markdown, analysis.problem_statement, analysis.method_overview, analysis.conclusion]
        numeric_sources.extend([c.text for c in analysis.contributions])
        if analysis.experiments:
            numeric_sources.append(analysis.experiments.main_results)
            numeric_sources.extend(analysis.experiments.takeaways)
        source_numbers = set().union(*(_extract_numbers(x) for x in numeric_sources if x)) if numeric_sources else set()
        source_formulas = set().union(*(_extract_formula_tokens(x) for x in numeric_sources if x)) if numeric_sources else set()

        section_lookup = {s.section_id: s for s in blueprint.sections}
        for sec_id, payload in list(suggestions.items()):
            if sec_id not in section_lookup or not isinstance(payload, dict):
                continue
            text = payload.get("content_md") or ""
            if text and source_numbers and not _extract_numbers(text).issubset(source_numbers):
                logger.info("Dropped suggestion for %s due to unmatched numeric content", sec_id)
                payload.pop("content_md", None)
            if text and source_formulas and not _extract_formula_tokens(text).issubset(source_formulas):
                logger.info("Dropped suggestion for %s due to unmatched formula content", sec_id)
                payload.pop("content_md", None)
            if not payload:
                suggestions.pop(sec_id, None)

    safe_review["suggestions"] = suggestions
    return safe_review


def _create_vision_client() -> LLMClient:
    provider = _poster_vision_provider()
    if provider == "agnes":
        return LLMClient(
            api_key=settings.agnes_api_key or settings.gemini_api_key,
            base_url=settings.agnes_base_url or settings.gemini_base_url,
            model=settings.agnes_model or settings.gemini_model,
        )
    return LLMClient(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        model=settings.gemini_model,
    )


def _build_optimize_prompt(
    blueprint: PosterBlueprint,
    doc: PaperDocument,
    analysis: PaperAnalysis,
    screenshot_note: str = "",
) -> str:
    parts: list[str] = []
    parts.append(f"## Paper\nTitle: {doc.title}")
    if doc.abstract:
        parts.append(f"Abstract: {doc.abstract[:800]}")
    if analysis.contributions:
        parts.append("\n## Contributions")
        for c in analysis.contributions:
            parts.append(f"- {c.text}")
    if analysis.code_url:
        parts.append(f"\n## Code / Project URL\n{analysis.code_url}")
    if analysis.experiments:
        parts.append("\n## Experiment Summary")
        if analysis.experiments.datasets:
            parts.append(f"Datasets: {', '.join(analysis.experiments.datasets)}")
        if analysis.experiments.metrics:
            parts.append(f"Metrics: {', '.join(analysis.experiments.metrics)}")
        if analysis.experiments.main_results:
            parts.append(f"Main results: {analysis.experiments.main_results}")
        if analysis.experiments.takeaways:
            parts.append("Takeaways:")
            for takeaway in analysis.experiments.takeaways:
                parts.append(f"- {takeaway}")
    if analysis.key_figures:
        parts.append("\n## Key Figures")
        for fig in analysis.key_figures:
            parts.append(f"- {fig.figure_id}: {fig.caption} [{fig.role}]")
    parts.append("\n## Current Poster Sections\n")
    for sec in blueprint.sections:
        if sec.type == "title":
            continue
        parts.append(f"### {sec.title} (Row {sec.row}, Col {sec.column})")
        text = sec.content_md[:400] if sec.content_md else "(empty)"
        parts.append(text + "\n")
    if screenshot_note:
        parts.append("\n## Screenshot Review\n")
        parts.append(screenshot_note)
    parts.append("\n## Global QA Checklist\n")
    parts.append("- Check whether the poster has too much whitespace or too little whitespace.")
    parts.append("- Check whether table heights are balanced and readable, not too tall or too compressed.")
    parts.append("- Check whether figure crops are appropriate and whether each figure is the right size.")
    parts.append("- Check whether each figure is the right semantic match for its section.")
    parts.append("- Check whether the layout is visually compact, balanced, and aesthetically pleasing.")
    parts.append("- Prefer factual corrections, better figure placement, and tighter section sizing over rewriting content.")
    parts.append("\n## Task\n")
    parts.append("Review each section and suggest improved content_md. "
                 "You may also adjust layout and figure selection for better balance. "
                 "Keep the same section_id set. Return JSON with suggestions keyed by section_id.")
    return "\n".join(parts)


def _apply_suggestions(
    blueprint: PosterBlueprint,
    suggestions: dict[str, Any],
) -> PosterBlueprint:
    section_map = {s.section_id: s for s in blueprint.sections}
    for sec_id, sug in suggestions.items():
        content_md = None
        if isinstance(sug, dict):
            content_md = sug.get("content_md")
        elif isinstance(sug, str):
            content_md = sug
        if content_md and sec_id in section_map:
            old = section_map[sec_id].content_md
            if content_md != old:
                section_map[sec_id].content_md = content_md
                logger.info("  Updated section [%s]: %d chars -> %d chars",
                            sec_id, len(old), len(content_md))
    return blueprint


def _build_harness_prompt(blueprint: PosterBlueprint, analysis: PaperAnalysis, doc: PaperDocument) -> str:
    highlights = []
    for c in analysis.contributions[:3]:
        highlights.append(c.text)
    if analysis.experiments and analysis.experiments.takeaways:
        highlights.extend(analysis.experiments.takeaways[:3])
    figure_lines = [f"- {fp.figure_id}: {fp.section_id} / {fp.width_ratio:.2f}" for fp in blueprint.figure_placements]
    section_lines = [f"- {s.section_id}: {s.title} @ r{s.row} c{s.column} span{s.col_span}" for s in blueprint.sections]
    return (
        f"Title: {doc.title}\n"
        f"Core contributions: {chr(10).join('- ' + x for x in highlights) if highlights else 'none'}\n"
        f"Sections:\n{chr(10).join(section_lines)}\n"
        f"Figures:\n{chr(10).join(figure_lines) if figure_lines else 'none'}\n"
    )


def optimize_poster(
    arxiv_id: str,
    output_dir: str = "output",
    max_iterations: int = 1,
) -> dict[str, Any]:
    logger.info("=== Poster Optimization with Gemini ===")

    db = PaperDatabase()
    doc = db.get_paper_by_arxiv(arxiv_id)
    analysis = db.get_analysis_by_arxiv(arxiv_id)
    db.close()
    if not doc or not analysis:
        raise RuntimeError(f"Paper or analysis not found for {arxiv_id}")

    vision_client = _create_vision_client()
    if not vision_client.api_key:
        logger.warning("Poster vision API key not configured; optimization will rely on local heuristics")

    out = resolve_paper_output_dir(output_dir, arxiv_id)
    bp_path = out / "blueprint.json"
    review_path = out / "poster_review.json"

    if bp_path.exists():
        blueprint = PosterBlueprint.model_validate_json(bp_path.read_text(encoding="utf-8"))
    else:
        blueprint = generate_blueprint(doc, analysis)

    renderer = HtmlPosterRenderer()
    html_path = out / "poster.html"
    screenshot_path = out / "poster.png"
    history: list[dict] = []
    latest_response: dict[str, Any] = {}
    latest_review: dict[str, Any] = {}
    best_quality = -1
    best_blueprint = copy.deepcopy(blueprint)

    try:
        for iteration in range(max_iterations):
            logger.info("--- Iteration %d/%d ---", iteration + 1, max_iterations)

            renderer.render_to_file(blueprint, doc, html_path)
            capture_poster(html_path, screenshot_path)

            core_assets = _build_core_figure_assets(doc, analysis, out)

            screenshot_note = []
            if screenshot_path.exists():
                screenshot_note.append(f"Screenshot path: {screenshot_path.as_posix()}")
            if core_assets:
                screenshot_note.append("Core figures: " + ", ".join(p.as_posix() for p in core_assets))

            provider = _poster_vision_provider()
            screenshot_review = multimodal_analyze(
                system_prompt=_AGNES_REVIEW_PROMPT if provider == "agnes" else _VISION_PROMPT,
                image_paths=([str(screenshot_path)] if screenshot_path.exists() else []) + [str(p) for p in core_assets],
                user_text=_build_harness_prompt(blueprint, analysis, doc),
                provider=provider,
            )
            if screenshot_review is not None:
                latest_review = _apply_structured_quality_gate(blueprint, doc, analysis, screenshot_review)
                screenshot_note.append(json.dumps(latest_review, ensure_ascii=False))
            prompt = _build_optimize_prompt(
                blueprint,
                doc,
                analysis,
                screenshot_note="\n".join(screenshot_note),
            )
            response = vision_client.chat_json(system=_SYSTEM_PROMPT, user=prompt)
            latest_response = response

            quality = response.get("quality_score", 0)
            issues = response.get("issues", [])
            suggestions = response.get("suggestions", {})
            needs = response.get("needs_improvement", True)

            history.append({
                "iteration": iteration + 1,
                "quality": quality,
                "issues": len(issues),
            })
            logger.info("Quality: %d/10, Issues: %d, Needs improvement: %s",
                        quality, len(issues), needs)

            # Process layout suggestions
            layout = response.get("layout", {})
            if layout:
                for sec in blueprint.sections:
                    if sec.section_id in layout:
                        ls = layout[sec.section_id]
                        if "row" in ls: sec.row = ls["row"]
                        if "column" in ls: sec.column = ls["column"]
                        if "col_span" in ls: sec.col_span = ls["col_span"]
                        logger.info("  Layout updated: %s -> row=%s col=%s span=%s",
                                    sec.section_id, sec.row, sec.column, sec.col_span)

            figure_selection = response.get("figure_selection", [])
            if isinstance(figure_selection, list) and figure_selection:
                from src.schemas.poster import FigurePlacement
                selection_map = {f.get("figure_id"): f for f in figure_selection if f.get("figure_id")}
                kept_ids = {fid for fid, sel in selection_map.items() if sel.get("keep")}
                if kept_ids:
                    before = len(blueprint.figure_placements)
                    blueprint.figure_placements = [fp for fp in blueprint.figure_placements if fp.figure_id in kept_ids]
                    for fp in blueprint.figure_placements:
                        sel = selection_map.get(fp.figure_id, {})
                        fp.section_id = sel.get("section_id", fp.section_id)
                        fp.width_ratio = sel.get("width_ratio", fp.width_ratio)
                        if sel.get("caption"):
                            fp.caption = sel["caption"]
                    logger.info("  Figure selection updated: %d -> %d", before, len(blueprint.figure_placements))

            # Generate chart from experiment data
            chart_data = response.get("chart_data")
            if chart_data:
                chart_path = _generate_result_chart(analysis, out, chart_data)
                if chart_path:
                    from src.schemas.paper import Figure as PaperFigure
                    from src.schemas.poster import FigurePlacement
                    # Add to doc.figures
                    chart_fig_id = "fig-results-chart"
                    if chart_fig_id not in {f.figure_id for f in doc.figures}:
                        doc.figures.append(PaperFigure(
                            figure_id=chart_fig_id, local_path=chart_path,
                            caption="Experimental Results", section_id="sec-experiments",
                        ))
                    # Add to blueprint
                    if chart_fig_id not in {fp.figure_id for fp in blueprint.figure_placements}:
                        blueprint.figure_placements.append(FigurePlacement(
                            figure_id=chart_fig_id, section_id="sec-experiments",
                            width_ratio=0.95, caption="Experimental Results",
                        ))
                    logger.info("  Result chart generated: %s", chart_path)

            if suggestions:
                blueprint = _apply_suggestions(blueprint, suggestions)
            if quality > best_quality:
                best_quality = quality
                best_blueprint = copy.deepcopy(blueprint)
            bp_path.write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")

            renderer.render_to_file(blueprint, doc, html_path)
            if not needs or quality >= 8:
                break

    except Exception as e:
        logger.exception("Optimization error: %s", e)

    blueprint = best_blueprint
    renderer.render_to_file(blueprint, doc, html_path)
    capture_poster(html_path, screenshot_path)
    review_path.write_text(json.dumps(latest_review or latest_response, ensure_ascii=False, indent=2), encoding="utf-8")

    final_q = history[-1]["quality"] if history else 0
    logger.info("=== Optimization complete: quality %d/10, %d iterations ===",
                final_q, len(history))
    return {
        "blueprint": blueprint,
        "history": history,
        "final_quality": final_q,
        "iterations": len(history),
        "response": latest_response,
        "review": latest_review,
    }
