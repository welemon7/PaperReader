from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.agents.parse_agent import run_parse_paper
from src.agents.understand_agent import run_understand_paper
from src.agents.poster_planner import generate_blueprint, normalize_analysis_for_poster
from src.agents.validation_agent import validate_poster
from src.renderers.html_renderer import HtmlPosterRenderer
from src.storage.sqlite import PaperDatabase
from src.schemas.paper import PaperDocument
from src.schemas.analysis import PaperAnalysis
from src.schemas.poster import PosterBlueprint
from src.schemas.validation import PosterValidation
from src.config import settings
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
    with_optimize: bool = False,
) -> dict:
    """Run all 5 phases sequentially with validation and retry loops."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
    use_gemini = bool(settings.gemini_api_key) if hasattr(settings, "gemini_api_key") else False
    blueprint = generate_blueprint(doc, analysis, use_gemini=use_gemini)
    results["blueprint"] = blueprint
    blueprint_path = output_dir / "blueprint.json"
    blueprint_path.write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Phase 3 complete: %d sections, %d figures", len(blueprint.sections), len(blueprint.figure_placements))

    # ---- Phase 4: Render ----
    logger.info("=== Phase 4: Render ===")
    renderer = HtmlPosterRenderer()
    html_path = output_dir / "poster.html"
    renderer.render_to_file(blueprint, doc, html_path)
    results["html_path"] = html_path
    logger.info("Phase 4 complete: %s (%d bytes)", html_path, html_path.stat().st_size)

    # ---- Phase 5: Validate ----
    logger.info("=== Phase 5: Validate ===")
    try:
        validation = validate_poster(arxiv_id, str(blueprint_path))
        results["validation"] = validation
        validation_path = output_dir / "validation.json"
        validation_path.write_text(validation.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Phase 5 complete: scores=%s, issues=%d", validation.scores, len(validation.issues))
    except Exception as e:
        logger.warning("Phase 5 (Validate) skipped: %s", e)
        results["validation"] = None

    if with_optimize:
        logger.info("=== Phase 6: Optimize with Gemini ====")
        try:
            from src.agents.optimizer_agent import optimize_poster
            opt = optimize_poster(arxiv_id, output_dir=str(output_dir), max_iterations=max_retries)
            results["optimization"] = opt
            logger.info("Phase 6: quality %d/10, %d iterations", opt["final_quality"], opt["iterations"])
        except Exception as e:
            logger.warning("Phase 6 skipped: %s", e)
            results["optimization"] = None

    logger.info("=== Pipeline complete ====")
    return results
