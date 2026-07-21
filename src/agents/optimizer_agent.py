from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a scientific poster optimization expert. "
    "Review a research poster blueprint and suggest specific improvements. "
    "Evaluate on: coverage, clarity, completeness, accuracy.\n\n"
    "Return JSON with:\n"
    '- "quality_score": int (1-10)\n'
    '- "issues": list of {"severity": str, "description": str}\n'
    '- "suggestions": dict of section_id -> {"content_md": "improved text"}\n'
    '- "layout": dict of section_id -> {"row": int, "column": int, "col_span": int}\n'
    '- "figure_selection": list of {"keep": bool, "figure_id": str, "section_id": str, "width_ratio": float, "caption": str}\n'
    '- "needs_improvement": bool (false if quality >= 8)\n\n'
    "For each section, suggest specific content improvements "
    "that are more concise, accurate, and impactful."
)


def _regenerate_figures(blueprint, doc, output_dir):
    """Scan arXiv cache for PDF figures, convert to PNG, add to blueprint."""
    from pathlib import Path
    if not doc.source_dir:
        return blueprint, []
    try:
        import fitz
    except ImportError:
        return blueprint, []
    src = Path(doc.source_dir)
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    new_placements = []
    existing_ids = {fp.figure_id for fp in blueprint.figure_placements}
    from src.schemas.poster import FigurePlacement
    for pdf in sorted(src.rglob("*.pdf")):
        stem = pdf.stem
        if len(stem) > 30 or stem.startswith("page_"):
            continue
        png_path = fig_dir / (stem + ".png")
        if not png_path.exists():
            try:
                with fitz.open(str(pdf)) as pdf_doc:
                    page = pdf_doc[0]
                    pix = page.get_pixmap(dpi=200)
                    pix.save(str(png_path))
            except:
                continue
        fig_id = "fig-cache-" + stem
        if fig_id not in existing_ids:
            new_placements.append(FigurePlacement(
                figure_id=fig_id, section_id="sec-main-method",
                width_ratio=0.9, caption="Figure: " + stem,
            ))
    from src.schemas.paper import Figure as PaperFigure
    from src.schemas.poster import FigurePlacement
    new_paper_figs = []
    existing_fig_ids = {f.figure_id for f in doc.figures}
    for fig_id in [p.figure_id for p in new_placements]:
        if fig_id not in existing_fig_ids:
            png = fig_dir / (fig_id.replace("fig-cache-", "") + ".png")
            if png.exists():
                new_paper_figs.append(PaperFigure(
                    figure_id=fig_id, local_path=str(png),
                    caption="", section_id="sec-main-method",
                ))
    if new_paper_figs:
        doc.figures.extend(new_paper_figs)
    if new_placements:
        blueprint.figure_placements.extend(new_placements)
    return blueprint, doc, new_placements


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
    "Be critical: 2-4 figures max. Remove redundant ones."
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


def _create_gemini_client() -> LLMClient:
    return LLMClient(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        model=settings.gemini_model,
    )


def _build_optimize_prompt(
    blueprint: PosterBlueprint,
    doc: PaperDocument,
    analysis: PaperAnalysis,
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

    gemini = _create_gemini_client()
    if not gemini.api_key or gemini.api_key == settings.gemini_api_key and not gemini.api_key:
        logger.warning("Gemini API key not configured, using DeepSeek fallback")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bp_path = out / "blueprint.json"

    if bp_path.exists():
        blueprint = PosterBlueprint.model_validate_json(bp_path.read_text(encoding="utf-8"))
    else:
        blueprint = generate_blueprint(doc, analysis)

    renderer = HtmlPosterRenderer()
    html_path = out / "poster.html"
    history: list[dict] = []
    latest_response: dict[str, Any] = {}

    try:
        for iteration in range(max_iterations):
            logger.info("--- Iteration %d/%d ---", iteration + 1, max_iterations)

            prompt = _build_optimize_prompt(blueprint, doc, analysis)
            response = gemini.chat_json(system=_SYSTEM_PROMPT, user=prompt)
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
                chart_path = _generate_result_chart(analysis, output_dir, chart_data)
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

            # Regenerate PDF figures from cache
            blueprint, doc, new_figs = _regenerate_figures(blueprint, doc, output_dir)
            if new_figs:
                logger.info("  Added %d figures from cache", len(new_figs))

            if suggestions:
                blueprint = _apply_suggestions(blueprint, suggestions)
            bp_path.write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")

            renderer.render_to_file(blueprint, doc, html_path)
            break

    except Exception as e:
        logger.exception("Optimization error: %s", e)

    renderer.render_to_file(blueprint, doc, html_path)

    final_q = history[-1]["quality"] if history else 0
    logger.info("=== Optimization complete: quality %d/10, %d iterations ===",
                final_q, len(history))
    return {
        "blueprint": blueprint,
        "history": history,
        "final_quality": final_q,
        "iterations": len(history),
        "response": latest_response,
    }
