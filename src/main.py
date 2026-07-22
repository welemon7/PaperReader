from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.agents.parse_agent import run_parse_paper
from src.agents.understand_agent import run_understand_paper
from src.agents.poster_planner import generate_blueprint
from src.utils.output_paths import resolve_paper_output_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cli_entry() -> None:
    """CLI entry point (console_scripts hook)."""
    args = _parse_args()
    _run(args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper Reader LLM — Phase 1: Parse arXiv paper into structured knowledge",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # parse subcommand
    p = sub.add_parser("parse", help="Parse an arXiv paper")
    p.add_argument("arxiv_id", help="arXiv ID, e.g. '2303.08774'")
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file for the PaperDocument",
    )
    p.add_argument(
        "--markdown", "-m",
        type=Path,
        default=None,
        help="Output Markdown file for the full paper",
    )
    p.add_argument("--force", action="store_true", help="Force re-parse, overwrite existing data")

    # list subcommand
    pf = sub.add_parser("parse-pdf", help="Parse a local PDF file into structured knowledge")
    pf.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    pf.add_argument("--output", "-o", type=Path, default=None, help="Output JSON file")
    pf.add_argument("--markdown", "-m", type=Path, default=None, help="Output Markdown file")

    pipeline = sub.add_parser("pipeline", help="Run all phases: parse -> understand -> plan -> render -> validate")
    pipeline.add_argument("arxiv_id", help="arXiv ID")
    pipeline.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    pipeline.add_argument("--force", action="store_true", help="Force re-parse")
    pipeline.add_argument("--optimize", action="store_true", help="Enable Gemini optimization")
    pipeline.add_argument("--no-validate", action="store_true", help="Skip Phase 5 validation")

    sub.add_parser("list", help="List parsed papers in the database")
    pl = sub.add_parser("plan", help="Generate poster blueprint from parsed and understood paper")
    pl.add_argument("arxiv_id", help="arXiv ID of the paper")
    pl.add_argument("--output", "-o", type=Path, default=None, help="Output JSON file for the PosterBlueprint")
    v = sub.add_parser("validate", help="Validate poster blueprint against the original paper")
    v.add_argument("arxiv_id", help="arXiv ID of the parsed paper")
    v.add_argument("blueprint", type=Path, help="Path to the PosterBlueprint JSON file")

    op = sub.add_parser("optimize", help="Optimize poster content using Gemini LLM")
    op.add_argument("arxiv_id", help="arXiv ID")
    op.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    op.add_argument("--iterations", type=int, default=1, help="Max optimization iterations")
    v.add_argument("--output", "-o", type=Path, default=None, help="Output validation JSON file")
    p = sub.add_parser("render", help="Render poster HTML from a blueprint JSON file")
    p.add_argument("blueprint", type=Path, help="Path to the PosterBlueprint JSON file")
    p.add_argument("--output", "-o", type=Path, default=None, help="Output HTML file path")
    p.add_argument("--png", type=Path, default=None, help="Output PNG file path (requires playwright)")

    # understand subcommand
    u = sub.add_parser("understand", help="Analyze a parsed paper with LLM to extract insights")
    u.add_argument("arxiv_id", help="arXiv ID of the previously parsed paper")
    u.add_argument("--output", "-o", type=Path, default=None, help="Output JSON file for the PaperAnalysis")

    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    if args.command == "parse":
        logger.info("Starting Phase 1 pipeline for arXiv paper: %s", args.arxiv_id)
        try:
            doc = run_parse_paper(args.arxiv_id, force=args.force)
        except Exception:
            logger.exception("Phase 1 pipeline failed")
            sys.exit(1)

        # Output JSON
        json_str = doc.model_dump_json(indent=2, exclude={"source_dir"})

        if args.output:
            args.output.write_text(json_str, encoding="utf-8")
            logger.info("PaperDocument saved to %s", args.output)
        else:
            print(json_str)

        # Output Markdown
        if args.markdown:
            args.markdown.write_text(doc.raw_markdown, encoding="utf-8")
            logger.info("Markdown saved to %s", args.markdown)

        logger.info(
            "Phase 1 complete: %d sections, %d formulas, %d figures, %d references",
            len(doc.sections),
            len(doc.formulas),
            len(doc.figures),
            len(doc.references),
        )

    
    elif args.command == "pipeline":
        from src.agents.pipeline_agent import run_pipeline
        logger.info("Starting full pipeline for %s", args.arxiv_id)
        try:
            output_dir = resolve_paper_output_dir(args.output_dir, args.arxiv_id)
            results = run_pipeline(
                args.arxiv_id,
                output_dir=output_dir,
                force=args.force,
                with_optimize=args.optimize,
            )
            bp = results.get("blueprint")
            if bp:
                logger.info("Pipeline complete: %d sections, %d figures", len(bp.sections), len(bp.figure_placements))
        except Exception:
            logger.exception("Pipeline failed")
            sys.exit(1)

    elif args.command == "plan":
        logger.info("Starting Phase 3 planning for arXiv paper: %s", args.arxiv_id)
        try:
            from src.storage.sqlite import PaperDatabase
            db = PaperDatabase()
            doc = db.get_paper_by_arxiv(args.arxiv_id)
            analysis = db.get_analysis_by_arxiv(args.arxiv_id)
            db.close()
            if not analysis:
                logger.error("No analysis found for %s. Run understand first.", args.arxiv_id)
                sys.exit(1)
            if not doc:
                logger.error("No paper found for %s. Run parse first.", args.arxiv_id)
                sys.exit(1)
            bp = generate_blueprint(doc, analysis)
        except Exception:
            logger.exception("Phase 3 planning failed")
            sys.exit(1)
        json_str = bp.model_dump_json(indent=2)
        if args.output:
            args.output.write_text(json_str, encoding="utf-8")
            logger.info("PosterBlueprint saved to %s", args.output)
        else:
            print(json_str)
        logger.info("Phase 3 complete: %d sections, %d figures", len(bp.sections), len(bp.figure_placements))

    elif args.command == "list":
        from src.storage.sqlite import PaperDatabase

        db = PaperDatabase()
        papers = db.list_papers()
        if not papers:
            print("No papers in database.")
        else:
            print(f"{'Paper ID':<20} {'arXiv ID':<20} {'Parsed At':<30}")
            print("-" * 70)
            for p in papers:
                print(
                    f"{p['paper_id']:<20} {p['arxiv_id']:<20} {p['created_at']:<30}"
                )


    elif args.command == "parse-pdf":
        from src.parsers.pdf_extractor import extract_pdf_text, extract_pdf_figures, infer_title, extract_formulas_from_text, extract_sections_from_markdown
        from src.schemas.paper import PaperDocument, Section, Formula, Figure
        pdf_path = Path(args.pdf_path)
        logger.info("Parsing PDF: %s", pdf_path)
        try:
            text = extract_pdf_text(pdf_path)
            figures = extract_pdf_figures(pdf_path)
            formulas = extract_formulas_from_text(text)
            sections = extract_sections_from_markdown(text)
            title = infer_title(text, pdf_path.stem)
            sec_list = []
            for i, (sn, sb) in enumerate(sections.items()):
                sec_list.append(Section(section_id=f"sec-{i+1:03d}", title=sn, level=1, text=sb[:5000], raw_latex=sb[:5000]))
            formula_models = [Formula(formula_id=f["formula_id"], latex=f["latex"], section_id=sec_list[0].section_id if sec_list else "sec-000") for f in formulas]
            doc = PaperDocument(
                paper_id=pdf_path.stem, arxiv_id=pdf_path.stem,
                title=title, authors=[], abstract=text[:2000],
                sections=sec_list, formulas=formula_models,
                raw_markdown=text[:15000],
            )
        except Exception:
            logger.exception("PDF parsing failed")
            sys.exit(1)

        from src.storage.sqlite import PaperDatabase
        db = PaperDatabase()
        db.save_paper(doc)
        db.close()

        json_str = doc.model_dump_json(indent=2, exclude={"source_dir"})
        if args.output:
            args.output.write_text(json_str, encoding="utf-8")
            logger.info("PaperDocument saved to %s", args.output)
        else:
            print(json_str)
        if args.markdown:
            args.markdown.write_text(text, encoding="utf-8")
            logger.info("Markdown saved to %s", args.markdown)
        logger.info("PDF parsed: %d chars, %d sections, %d formulas, %d figures", len(text), len(sections), len(formulas), len(figures))

    elif args.command == "understand":
        logger.info("Starting Phase 2 analysis for arXiv paper: %s", args.arxiv_id)
        try:
            analysis = run_understand_paper(args.arxiv_id)
        except Exception:
            logger.exception("Phase 2 pipeline failed")
            sys.exit(1)
        json_str = analysis.model_dump_json(indent=2)
        if args.output:
            args.output.write_text(json_str, encoding="utf-8")
            logger.info("PaperAnalysis saved to %s", args.output)
        else:
            print(json_str)
        logger.info(
            "Phase 2 complete: %d contributions, %d formulas, %d figures",
            len(analysis.contributions),
            len(analysis.key_formulas),
            len(analysis.key_figures),
        )

if __name__ == "__main__":
    cli_entry()
