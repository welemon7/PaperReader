from __future__ import annotations

import logging
import re
import os
from pathlib import Path
from typing import Optional

import markdown as md_lib
from jinja2 import Environment, FileSystemLoader

from src.schemas.paper import PaperDocument
from src.schemas.poster import PosterBlueprint, FigurePlacement

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class HtmlPosterRenderer:
    def __init__(self, template_dir: str = _TEMPLATE_DIR) -> None:
        self.env = Environment(loader=FileSystemLoader(template_dir))
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

    def render(self, blueprint: PosterBlueprint, doc: PaperDocument) -> str:
        try:
            self._convert_pdf_figures(blueprint, doc)
        except Exception:
            pass
        rows = self._organize_rows(blueprint)
        figure_map = self._build_figure_map(blueprint, doc)

        for row_data in rows:
            for sec in row_data["sections"]:
                sec.content_html = self._markdown_with_latex(
                    sec.content_md
                )

        return self.template.render(
            poster_title=blueprint.poster_title,
            authors_str=blueprint.authors_str,
            poster_width=blueprint.width_px,
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
        html = self.render(blueprint, doc)
        html = html.replace("output/figures/", "figures/").replace("output\\figures\\", "figures/")
        output_path.write_text(html, encoding="utf-8")
        logger.info("Poster HTML saved to %s", output_path)
        return output_path

    def _convert_pdf_figures(self, blueprint: PosterBlueprint, doc: PaperDocument) -> None:
        from pathlib import Path
        try:
            import fitz
            fig_dir = Path("output/figures")
            fig_dir.mkdir(parents=True, exist_ok=True)
            lookup = {f.figure_id: f for f in doc.figures}
            for fp in blueprint.figure_placements:
                fig = lookup.get(fp.figure_id)
                if fig and fig.local_path and doc.source_dir:
                    found = list(Path(doc.source_dir).rglob(fig.local_path))
                    if found:
                        pdf = found[0]
                        png = fig_dir / (pdf.stem + ".png")
                        if not png.exists():
                            with fitz.open(str(pdf)) as pdoc:
                                page = pdoc[0]
                                pix = page.get_pixmap(dpi=200)
                                pix.save(str(png))
                        fig.local_path = str(png)
        except Exception:
            pass

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
        blueprint: PosterBlueprint, doc: PaperDocument
    ) -> dict[str, list[dict]]:
        fig_lookup = {f.figure_id: f for f in doc.figures}
        fig_map: dict[str, list[dict]] = {}
        for fp in blueprint.figure_placements:
            fig = fig_lookup.get(fp.figure_id)
            entry = {
                "figure_id": fp.figure_id,
                "caption": fp.caption or (fig.caption if fig else ""),
                "src": None,
                "width_ratio": fp.width_ratio,
            }
            if fig:
                entry["src"] = fig.minio_path or fig.local_path
            fig_map.setdefault(fp.section_id, []).append(entry)
        return fig_map

    @staticmethod
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