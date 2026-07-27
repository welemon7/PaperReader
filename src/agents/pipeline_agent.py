from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.agents.parse_agent import run_parse_paper
from src.agents.understand_agent import run_understand_paper
from src.agents.poster_planner import generate_blueprint, normalize_analysis_for_poster
from src.agents.poster_v2 import build_layout_tree, layout_tree_to_blueprint
from src.agents.validation_agent import validate_poster
from src.renderers.html_renderer import HtmlPosterRenderer
from src.storage.sqlite import PaperDatabase
from src.schemas.paper import PaperDocument
from src.schemas.analysis import PaperAnalysis
from src.schemas.poster import PosterBlueprint
from src.schemas.validation import PosterValidation
from src.config import settings
from src.utils.output_paths import resolve_paper_output_dir
logger = logging.getLogger(__name__)



def validate_doc(doc: PaperDocument) -> list[str]:
    issues = []
    if not doc.title:
        issues.append("Missing title")
    if not doc.sections:
        issues.append("Missing sections")
    if not doc.raw_markdown or len(doc.raw_markdown) < 100:
        issues.append("Content too short")
    return issues


def validate_analysis(analysis: Optional[PaperAnalysis]) -> list[str]:
    issues = []
    if not analysis:
        return ["No analysis produced"]
    if not analysis.problem_statement:
        issues.append("Missing problem statement")
    if not analysis.contributions:
        issues.append("Missing contributions")
    if not analysis.method_overview:
        issues.append("Missing method overview")
    return issues


def run_pipeline(
    arxiv_id: str,
    output_dir: Path = Path("output"),
    force: bool = True,
    max_retries: int = 2,
) -> dict:
    """Run all 5 phases sequentially with validation and retry loops."""
    output_dir = resolve_paper_output_dir(output_dir, arxiv_id)
    results: dict = {}

    # ---- Phase 1: Parse ----
    logger.info("=== Phase 1: Parse ===")
    doc: Optional[PaperDocument] = None
    for attempt in range(1, max_retries + 1):
        try:
            doc = run_parse_paper(arxiv_id, force=(force and attempt == 1))
            break
        except Exception as e:
            logger.warning("Parse attempt %d failed: %s", attempt, e)
            if attempt == max_retries:
                raise RuntimeError(f"Parse failed after {max_retries} attempts: {e}")
    issues = validate_doc(doc)
    if issues:
        logger.warning("Parse validation issues: %s", issues)
    doc_path = output_dir / "paper.json"
    doc_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    results["paper"] = doc
    logger.info("Phase 1 complete: %d sections, %d chars", len(doc.sections), len(doc.raw_markdown))

    # ---- Phase 2: Understand ----
    logger.info("=== Phase 2: Understand ===")
    analysis: Optional[PaperAnalysis] = None
    for attempt in range(1, max_retries + 1):
        try:
            analysis = run_understand_paper(arxiv_id)
            issues = validate_analysis(analysis)
            if not issues:
                break
            logger.warning("Analysis validation issues (attempt %d): %s", attempt, issues)
        except Exception as e:
            logger.warning("Understand attempt %d failed: %s", attempt, e)
            if attempt == max_retries:
                raise RuntimeError(f"Understand failed after {max_retries} attempts: {e}")
    results["analysis"] = analysis
    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Phase 2 complete: %d contributions, %d formulas",
        len(analysis.contributions), len(analysis.key_formulas),
    )
    analysis = normalize_analysis_for_poster(analysis)
    analysis_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")

    # ---- Phase 3: Plan ----
    logger.info("=== Phase 3: Plan ===")
    blueprint = generate_blueprint(doc, analysis)
    results["blueprint"] = blueprint
    blueprint_path = output_dir / "blueprint.json"
    blueprint_path.write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Phase 3 complete: %d sections, %d figures", len(blueprint.sections), len(blueprint.figure_placements))

    # ---- Phase 3b: v2 Layout Tree ----
    try:
        layout_tree = build_layout_tree(doc, analysis)
        results["layout_tree"] = layout_tree
        layout_tree_path = output_dir / "layout_tree.json"
        layout_tree_path.write_text(layout_tree.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Phase 3b complete: %d tree nodes", len(layout_tree.nodes))
    except Exception as e:
        logger.warning("Phase 3b skipped: %s", e)
        results["layout_tree"] = None

    # ---- Phase 4: Render ----
    logger.info("=== Phase 4: Render ===")
    renderer = HtmlPosterRenderer()
    html_path = output_dir / "poster.html"
    renderer.render_to_file(blueprint, doc, html_path)
    results["html_path"] = html_path
    logger.info("Phase 4 complete: %s (%d bytes)", html_path, html_path.stat().st_size)


    # ---- Phase 5: Validate ----
    logger.info("=== Phase 5: Validate (waiting writing)===")
    results["validation"] = None

    logger.info("=== Pipeline complete ====")
    return results
