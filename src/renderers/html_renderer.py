from __future__ import annotations

import html as html_lib
import logging
import os
import re
from pathlib import Path
from typing import Optional

import markdown as md_lib
from jinja2 import Environment, FileSystemLoader

from src.schemas.paper import PaperDocument
from src.schemas.poster import PosterBlueprint, FigurePlacement
from src.utils.figure_assets import copy_or_rasterize_asset, resolve_figure_source, sanitize_asset_name

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class HtmlPosterRenderer:
    def __init__(self, template_dir: str = _TEMPLATE_DIR) -> None:
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.env.filters["summarize_text"] = self._summarize_text
        self.template = self.env.get_template("poster.html.j2")

    @staticmethod
    def _markdown_with_latex(text: str) -> str:
        """Convert markdown to HTML while preserving LaTeX math blocks."""
        placeholders: dict[str, str] = {}
        counter = [0]

        def _protect_math(m):
            counter[0] += 1
            key = f"%%MATH{chr(64+counter[0])}%%"
            placeholders[key] = m.group(0)
            return key

        # Protect display math $$...$$
        text = re.sub(r"\$\$(.+?)\$\$", _protect_math, text, flags=re.DOTALL)
        # Protect inline math \(...\)
        text = re.sub(r"\\\((.+?)\\\)", _protect_math, text, flags=re.DOTALL)

        html = md_lib.markdown(text, extensions=["extra"])

        for key, val in placeholders.items():
            html = html.replace(key, val)
        return html

    @staticmethod
    def _summarize_text(text: str, max_sentences: int = 2) -> str:
        """Return a short readable summary without adding ellipses.

        The poster hero cards should show complete copy, so we keep the first
        one or two sentences instead of doing a hard character truncate.
        """
        if not text:
            return ""

        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = html_lib.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""

        sentences = re.split(r"(?<=[.!?。！？])\s+", cleaned)
        if len(sentences) <= max_sentences:
            return cleaned
        return " ".join(sentences[:max_sentences]).strip()

    @staticmethod
    def _clean_formula_latex(text: str) -> str:
        from src.agents.poster_planner import _clean_formula_latex

        return _clean_formula_latex(text)

    @staticmethod
    def _clean_html_text(text: str) -> str:
        from src.agents.poster_planner import _normalize_latex_command_names

        text = _normalize_latex_command_names(text or "")
        text = re.sub(r"~\\(?:cite|ref|eqref|autoref|label)\s*\{[^{}]*\}", "", text)
        text = re.sub(r"\\(?:cite|ref|eqref|autoref|label)\s*\{[^{}]*\}", "", text)
        text = re.sub(r"\\protect\s*", "", text)

        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).rstrip()
            if cleaned.strip():
                lines.append(cleaned)
            elif line.strip() == "":
                lines.append("")
        return "\n".join(lines).strip()

    def render(
        self,
        blueprint: PosterBlueprint,
        doc: PaperDocument,
        output_dir: Path | None = None,
    ) -> str:
        if output_dir is None:
            from src.utils.output_paths import sanitize_output_name

            output_dir = Path("output") / sanitize_output_name(doc.arxiv_id or doc.paper_id)
        else:
            output_dir = Path(output_dir)
        try:
            self._prepare_figure_assets(blueprint, doc, output_dir)
        except Exception:
            pass
        rows = self._organize_rows(blueprint)
        figure_map = self._build_figure_map(blueprint, doc, output_dir)
        cleaned_formulas = []
        for formula in blueprint.formula_displays:
            cleaned_latex = self._clean_formula_latex(formula.latex)
            if not cleaned_latex:
                continue
            formula.latex = cleaned_latex
            formula.semantic_desc = re.sub(r"\s+", " ", (formula.semantic_desc or "")).strip()
            cleaned_formulas.append(formula)
        blueprint.formula_displays = cleaned_formulas

        for row_data in rows:
            for sec in row_data["sections"]:
                sec.content_md = self._clean_html_text(sec.content_md)
                sec.content_html = self._markdown_with_latex(
                    sec.content_md
                )

        return self.template.render(
            poster_title=blueprint.poster_title,
            authors_str=blueprint.authors_str,
            poster_width=blueprint.width_px,
            poster_height=blueprint.height_px,
            color_scheme=blueprint.color_scheme,
            rows=rows,
            figure_map=figure_map,
        )

    def render_to_file(
        self,
        blueprint: PosterBlueprint,
        doc: PaperDocument,
        output_path: Path,
    ) -> Path:
        html = self.render(blueprint, doc, output_path.parent)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Poster HTML saved to %s", output_path)
        return output_path

    def _prepare_figure_assets(self, blueprint: PosterBlueprint, doc: PaperDocument, output_dir: Path) -> None:
        """Normalize all poster figures to browser-friendly local assets.

        The HTML poster is rendered in a browser, so PDF figures are rasterized
        to PNG first and copied into the poster-local figures directory.
        """
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fp, fig in self._iter_placement_figures(blueprint, doc):
            if not fig:
                continue
            candidate = self._resolve_figure_path(fig.local_path or fig.minio_path or "", doc.source_dir)
            if not candidate:
                continue
            prepared = self._ensure_browser_asset(candidate, fig_dir, fig.figure_id)
            if prepared:
                fig.local_path = str(prepared)

    def _ensure_browser_asset(self, src: Path, out_dir: Path, figure_id: str) -> Path | None:
        src = src.resolve()
        if not src.exists():
            return None

        target_name = sanitize_asset_name(figure_id or src.stem, src.stem)
        prepared = copy_or_rasterize_asset(src, out_dir, target_name)
        if prepared:
            return prepared
        if src.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
            return None
        return src

    @staticmethod
    def _resolve_figure_path(local_path: str, source_dir: str) -> Path | None:
        return resolve_figure_source(local_path, source_dir)

    @staticmethod
    def _normalize_figure_key(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    @classmethod
    def _figure_aliases(cls, fig) -> set[str]:
        aliases = {
            cls._normalize_figure_key(getattr(fig, "figure_id", "")),
            cls._normalize_figure_key(getattr(fig, "label", "")),
            cls._normalize_figure_key(getattr(fig, "caption", "")),
        }
        local_path = getattr(fig, "local_path", None)
        if local_path:
            aliases.add(cls._normalize_figure_key(Path(local_path).stem))
            aliases.add(cls._normalize_figure_key(Path(local_path).name))
        minio_path = getattr(fig, "minio_path", None)
        if minio_path:
            aliases.add(cls._normalize_figure_key(Path(minio_path).stem))
            aliases.add(cls._normalize_figure_key(Path(minio_path).name))
        return {alias for alias in aliases if alias}

    @classmethod
    def _build_figure_indexes(cls, doc: PaperDocument) -> tuple[list, dict[str, object]]:
        figures = list(doc.figures)
        alias_lookup: dict[str, object] = {}
        for fig in figures:
            for alias in cls._figure_aliases(fig):
                alias_lookup.setdefault(alias, fig)
        return figures, alias_lookup

    @classmethod
    def _match_placement_figure(
        cls,
        fp: FigurePlacement,
        figures: list,
        alias_lookup: dict[str, object],
        used_ids: set[str],
    ):
        placement_keys = [
            cls._normalize_figure_key(fp.figure_id),
            cls._normalize_figure_key(fp.caption),
        ]
        for key in placement_keys:
            if not key:
                continue
            fig = alias_lookup.get(key)
            if fig and getattr(fig, "figure_id", None) not in used_ids:
                return fig

        for fig in figures:
            if getattr(fig, "figure_id", None) in used_ids:
                continue
            if cls._normalize_figure_key(getattr(fig, "figure_id", "")) == cls._normalize_figure_key(fp.figure_id):
                return fig

        for fig in figures:
            if getattr(fig, "figure_id", None) in used_ids:
                continue
            if cls._normalize_figure_key(getattr(fig, "caption", "")) == cls._normalize_figure_key(fp.caption):
                return fig

        for fig in figures:
            if getattr(fig, "figure_id", None) not in used_ids:
                return fig
        return None

    @classmethod
    def _iter_placement_figures(
        cls,
        blueprint: PosterBlueprint,
        doc: PaperDocument,
    ) -> list[tuple[FigurePlacement, object | None]]:
        figures, alias_lookup = cls._build_figure_indexes(doc)
        used_ids: set[str] = set()
        matched: list[tuple[FigurePlacement, object | None]] = []

        for fp in blueprint.figure_placements:
            fig = cls._match_placement_figure(fp, figures, alias_lookup, used_ids)
            if fig and getattr(fig, "figure_id", None):
                used_ids.add(fig.figure_id)
            matched.append((fp, fig))
        return matched

    def _prepare_asset_uri(self, src: Path, out_dir: Path, target_name: str) -> str | None:
        prepared = copy_or_rasterize_asset(src, out_dir, target_name)
        if not prepared:
            return None
        return self._browser_asset_uri(prepared)

    @staticmethod
    def _organize_rows(blueprint: PosterBlueprint) -> list[dict]:
        grouped: dict[int, list] = {}
        for sec in blueprint.sections:
            if sec.type == "title":
                continue
            grouped.setdefault(sec.row, []).append(sec)
        rows = []
        for row_id in sorted(grouped.keys()):
            sections = sorted(grouped[row_id], key=lambda s: s.column)
            rows.append({"row_id": row_id, "sections": sections})
        return rows

    @staticmethod
    def _build_figure_map(
        blueprint: PosterBlueprint,
        doc: PaperDocument,
        output_dir: Path,
    ) -> dict[str, list[dict]]:
        fig_map: dict[str, list[dict]] = {}
        for fp, fig in HtmlPosterRenderer._iter_placement_figures(blueprint, doc):
            section_type = ""
            for sec in blueprint.sections:
                if sec.section_id == fp.section_id:
                    section_type = sec.type
                    break
            entry = {
                "figure_id": fp.figure_id,
                "caption": fp.caption or (fig.caption if fig else ""),
                "src": None,
                "width_ratio": fp.width_ratio,
                "section_type": section_type,
            }
            if fig:
                src = fig.local_path or fig.minio_path
                if src:
                    resolved = HtmlPosterRenderer._resolve_figure_path(src, doc.source_dir)
                    if resolved and resolved.suffix.lower() == ".pdf":
                        resolved = copy_or_rasterize_asset(resolved, output_dir / "figures", fig.figure_id) or resolved
                    if resolved:
                        entry["src"] = HtmlPosterRenderer._browser_asset_uri(resolved, output_dir)
                    else:
                        entry["src"] = src if src.startswith("file:") else Path(src).resolve().as_uri() if Path(src).exists() else src
            fig_map.setdefault(fp.section_id, []).append(entry)
        return fig_map

    @staticmethod
    def _browser_asset_uri(path: Path, output_dir: Path | None = None) -> str:
        resolved = path.resolve()
        if output_dir is None:
            output_root = resolved.parent.parent.resolve() if resolved.parent.name == "figures" else resolved.parent.resolve()
        else:
            output_root = Path(output_dir).resolve()
        try:
            return resolved.relative_to(output_root).as_posix()
        except ValueError:
            return resolved.as_uri()

    @staticmethod
    def capture_png(html_path: Path, png_path: Path, width: int = 1200, height: int = 1697) -> bool:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.warning(
                "playwright not installed. Install with: pip install playwright && playwright install chromium"
            )
            return False
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                )
                page.goto(html_path.as_uri())
                page.wait_for_timeout(3000)
                page.screenshot(path=str(png_path), full_page=True)
                browser.close()
            logger.info("Poster PNG saved to %s", png_path)
            return True
        except Exception as e:
            logger.exception("PNG capture failed: %s", e)
            return False
