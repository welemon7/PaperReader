from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    END = "__end__"
    StateGraph = None

from src.parsers.extractor import ComponentExtractor
from src.parsers.latex_parser import LatexParser, ParseResult
from src.parsers.markdown import MarkdownConverter
from src.schemas.paper import (
    Author,
    Figure,
    Formula,
    PaperDocument,
    Reference,
    Section,
)
from src.storage.minio import ImageStorage
from src.storage.sqlite import PaperDatabase
from src.utils.arxiv import ArxivDownloader

logger = logging.getLogger(__name__)


# ---------- State ----------


class ParseState(TypedDict):
    arxiv_id: str
    source_dir: Optional[str]
    main_tex: Optional[str]
    parse_result: Optional[ParseResult]
    components: Optional[dict]
    paper_document: Optional[PaperDocument]
    error: Optional[str]


# ---------- Node functions ----------


def download_node(state: ParseState) -> dict:
    """Download LaTeX source from arXiv."""
    arxiv_id = state.get("arxiv_id", "")
    if not arxiv_id:
        return {"error": "No arXiv ID provided"}

    try:
        downloader = ArxivDownloader()
        source_dir, main_tex, source_type = downloader.download(arxiv_id)
        logger.info("Downloaded source to %s, main = %s", source_dir, main_tex)
        return {
            "source_dir": str(source_dir),
            "main_tex": str(main_tex),
        }
    except Exception as e:
        logger.exception("Download failed for %s", arxiv_id)
        return {"error": f"Download failed: {e}"}


def parse_node(state: ParseState) -> dict:
    """Parse LaTeX structure using LatexParser."""
    main_tex_path = state.get("main_tex")
    source_dir_path = state.get("source_dir")
    if not main_tex_path:
        return {"error": "No main.tex path available"}

    try:
        parser = LatexParser(Path(source_dir_path))
        parse_result = parser.parse(Path(main_tex_path))
        logger.info(
            "Parsed %d top-level sections, %d total",
            len(parse_result.section_bodies),
            len(parse_result.all_section_bodies),
        )
        return {"parse_result": parse_result}
    except Exception as e:
        logger.exception("Parsing failed for %s", main_tex_path)
        return {"error": f"Parsing failed: {e}"}


def extract_node(state: ParseState) -> dict:
    """Extract formulas, figures, and references."""
    parse_result = state.get("parse_result")
    if not parse_result:
        return {"error": "No parse result available"}

    try:
        extractor = ComponentExtractor()
        components = extractor.extract_all(parse_result)
        logger.info(
            "Extracted %d formulas, %d figures, %d references",
            len(components["formulas"]),
            len(components["figures"]),
            len(components["references"]),
        )
        return {"components": components}
    except Exception as e:
        logger.exception("Extraction failed")
        return {"error": f"Extraction failed: {e}"}


def build_node(state: ParseState) -> dict:
    """Assemble PaperDocument from parse result + components."""
    parse_result = state.get("parse_result")
    components = state.get("components")
    arxiv_id = state.get("arxiv_id", "")

    if not parse_result or not components:
        return {"error": "Missing parse result or components"}

    try:
        # Build section tree
        sections = _build_section_tree(parse_result)

        # Convert to Markdown
        converter = MarkdownConverter()
        raw_markdown = converter.convert(parse_result)

        # Build flattened model instances
        formulas = []
        for comp in components.get("formulas", []):
            formulas.append(Formula(**comp))

        figures = []
        for comp in components.get("figures", []):
            figures.append(Figure(**comp))

        references = []
        for comp in components.get("references", []):
            references.append(Reference(**comp))

        authors = [
            Author(**a) for a in parse_result.authors
        ]

        doc = PaperDocument(
            paper_id=arxiv_id,
            arxiv_id=arxiv_id,
            title=parse_result.title,
            authors=authors,
            abstract=parse_result.abstract,
            sections=sections,
            formulas=formulas,
            figures=figures,
            references=references,
            raw_markdown=raw_markdown,
            source_dir=state.get("source_dir", ""),
        )

        logger.info("Built PaperDocument with %d sections", len(sections))
        return {"paper_document": doc}
    except Exception as e:
        logger.exception("Build failed")
        return {"error": f"Build failed: {e}"}


def store_node(state: ParseState) -> dict:
    """Store PaperDocument to SQLite + MinIO."""
    doc = state.get("paper_document")
    if not doc:
        return {"error": "No paper document to store"}

    try:
        # SQLite
        db = PaperDatabase()
        db.save_paper(doc)
        db.close()

        # MinIO (only if available)
        source_dir = state.get("source_dir", "")
        if source_dir:
            image_map = {}
            for fig in doc.figures:
                if fig.local_path:
                    local_path = Path(fig.local_path)
                    full_path = str(local_path if local_path.is_absolute() else Path(source_dir) / local_path)
                    image_map[fig.figure_id] = full_path

            if image_map:
                try:
                    img_store = ImageStorage()
                    minio_paths = img_store.upload_images_batch(
                        image_map, doc.arxiv_id
                    )
                    # Update figure minio_path in document
                    for fig in doc.figures:
                        if fig.figure_id in minio_paths:
                            fig.minio_path = minio_paths[fig.figure_id]
                    img_store.close()
                except Exception as e:
                    logger.warning("MinIO upload skipped: %s", e)

            logger.info("Stored paper %s successfully", doc.paper_id)

        return {"paper_document": doc}
    except Exception as e:
        logger.exception("Store failed")
        return {"error": f"Store failed: {e}"}


# ---------- Router ----------


def router(state: ParseState) -> str:
    """Route to the next node or END on error."""
    if state.get("error"):
        return "end"
    # Check which nodes have run
    if state.get("paper_document"):
        # After store, we're done
        # But if we need to distinguish more steps, use state flags
        pass
    return "continue"


# ---------- Build graph ----------


def build_parse_graph() -> StateGraph:
    """Build the Phase 1 parsing LangGraph."""
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed")
    workflow = StateGraph(ParseState)

    workflow.add_node("download", download_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("build", build_node)
    workflow.add_node("store", store_node)

    workflow.set_entry_point("download")

    workflow.add_edge("download", "parse")
    workflow.add_edge("parse", "extract")
    workflow.add_edge("extract", "build")
    workflow.add_edge("build", "store")
    workflow.add_edge("store", END)

    return workflow.compile()


# ---------- Convenience runner ----------


_compiled_graph = None


def run_parse_paper(arxiv_id: str, force: bool = False) -> PaperDocument:
    """Run the full parsing pipeline for an arXiv paper.

    This is the main entry point for Phase 1.
    """
    global _compiled_graph
    if _compiled_graph is None:
        try:
            _compiled_graph = build_parse_graph()
        except RuntimeError:
            _compiled_graph = None

    if _compiled_graph is None:
        state: ParseState = {
            "arxiv_id": arxiv_id,
            "source_dir": None,
            "main_tex": None,
            "parse_result": None,
            "components": None,
            "paper_document": None,
            "error": None,
        }
        for node in (download_node, parse_node, extract_node, build_node, store_node):
            state.update(node(state))
            if state.get("error"):
                raise RuntimeError(state["error"])
        return state["paper_document"]

    initial_state: ParseState = {
        "arxiv_id": arxiv_id,
        "source_dir": None,
        "main_tex": None,
        "parse_result": None,
        "components": None,
        "paper_document": None,
        "error": None,
    }

    if force:
        from src.storage.sqlite import PaperDatabase
        from src.storage.sqlite import SectionRecord, FigureRecord, FormulaRecord, ReferenceRecord, PaperRecord
        logger.info("Force mode: deleting existing data for %s", arxiv_id)
        try:
            db = PaperDatabase()
            with db.Session() as session:
                for tbl in (ReferenceRecord, FigureRecord, FormulaRecord, SectionRecord, PaperRecord):
                    session.query(tbl).filter(tbl.paper_id == arxiv_id).delete()
                session.commit()
            db.close()
        except Exception:
            pass
    result = _compiled_graph.invoke(initial_state)

    if result.get("error"):
        raise RuntimeError(f"Parsing failed: {result['error']}")

    doc = result.get("paper_document")
    if not doc:
        raise RuntimeError("No paper document produced")

    return doc


# ---------- Internal helpers ----------


def _build_section_tree(parse_result: ParseResult) -> list[Section]:
    """Convert ParseResult section_bodies into nested Section model instances."""
    sections = []
    for sb in parse_result.section_bodies:
        section = _section_body_to_model(sb, parse_result)
        sections.append(section)
    return sections


def _section_body_to_model(sb, parse_result: ParseResult) -> Section:
    """Convert a single SectionBody (raw) to a Section model.

    Also attaches matching formulas and figures from the flat indices.
    """
    # Find formulas that belong to this section
    formulas = []
    for comp_f in parse_result.formulas:
        if comp_f.section_id == sb.section_id:
            formulas.append(Formula(**comp_f))

    figures = []
    for comp_fig in parse_result.figures:
        if comp_fig.section_id == sb.section_id:
            figures.append(Figure(**comp_fig))

    subsections = []
    if hasattr(sb, "subsections") and sb.subsections:
        for sub in sb.subsections:
            subsections.append(_section_body_to_model(sub, parse_result))

    return Section(
        section_id=sb.section_id,
        title=sb.title,
        level=sb.level,
        text="",  # Will be populated from cleaned LaTeX in future
       raw_latex=sb.raw_latex,
       formulas=formulas,
       figures=figures,
       subsections=subsections,
   )
