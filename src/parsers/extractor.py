from __future__ import annotations

import logging
import re

from .latex_parser import ParseResult
from src.utils.figure_assets import resolve_figure_source, sanitize_asset_name
from .table_extractor import extract_tables

logger = logging.getLogger(__name__)


class ComponentExtractor:
    """Extract formulas, figures, and references from parsed LaTeX."""

    def __init__(self) -> None:
        self._formula_counter = 0
        self._figure_counter = 0
        self._ref_counter = 0

    def extract_all(self, result: ParseResult) -> dict:
        """Extract all components and return as dicts keyed by type."""
        self._formula_counter = 0
        self._figure_counter = 0
        self._ref_counter = 0

        formulas = self._extract_formulas(result)
        figures = self._extract_figures(result)
        references = self._extract_references(result.merged_latex)
        tables = extract_tables(result.merged_latex, self._section_id_for_position(result))

        return {
            "formulas": formulas,
            "figures": figures,
            "references": references,
            "tables": tables,
        }

    @staticmethod
    def _section_id_for_position(result: ParseResult):
        return lambda pos: ComponentExtractor._find_section_id(result, pos)

    # ---- formulas ----

    def _extract_formulas(self, result: ParseResult) -> list[dict]:
        """Extract all displayed and inline formulas with section mapping."""
        # First pass: extract formula bodies from the merged LaTeX
        formulas: list[dict] = []

        # Pattern 1: \[ ... \] (display math)
        for m in re.finditer(r"\\\[(.*?)\\\]", result.merged_latex, re.DOTALL):
            self._formula_counter += 1
            fid = f"f-{self._formula_counter:03d}"
            sec_id = self._find_section_id(result, m.start())
            formulas.append({
                "formula_id": fid,
                "latex": m.group(1).strip(),
                "semantic_desc": "",
                "section_id": sec_id,
                "label": self._find_label(result.merged_latex, m.end()),
            })

        # Pattern 2: \begin{equation}...\end{equation} / \begin{align}...\end{align}
        for env in (r"equation", r"equation\*", r"align", r"align\*", r"gather", r"gather\*"):
            pattern = re.compile(
                rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", re.DOTALL
            )
            for m in pattern.finditer(result.merged_latex):
                self._formula_counter += 1
                fid = f"f-{self._formula_counter:03d}"
                sec_id = self._find_section_id(result, m.start())
                formulas.append({
                    "formula_id": fid,
                    "latex": m.group(1).strip(),
                    "semantic_desc": "",
                    "section_id": sec_id,
                    "label": self._find_label(result.merged_latex, m.end()),
                })

        # Pattern 3: \( ... \) (inline math)  — skip for now, too many
        # We'll include them in the markdown but not as separate Formula entries

        return formulas

    # ---- figures ----

    def _extract_figures(self, result: ParseResult) -> list[dict]:
        """Extract \\includegraphics commands with surrounding context."""
        figures: list[dict] = []

        # Find figure environments
        fig_pattern = re.compile(
            r"\\begin\{figure\}(.*?)\\end\{figure\}", re.DOTALL
        )
        for m in fig_pattern.finditer(result.merged_latex):
            fig_content = m.group(1)
            sec_id = self._find_section_id(result, m.start())

            # Extract caption
            caption = ""
            cap_m = re.search(r"\\caption\s*\{([^}]*)\}", fig_content, re.DOTALL)
            if cap_m:
                caption = cap_m.group(1).strip()

            # Extract label
            label = ""
            lab_m = re.search(r"\\label\s*\{([^}]*)\}", fig_content)
            if lab_m:
                label = lab_m.group(1).strip()

            # Extract includegraphics
            for g_m in re.finditer(
                r"\\includegraphics(?:\[([^\]]*)\])?\s*\{([^}]*)\}", fig_content
            ):
                self._figure_counter += 1
                fid = f"fig-{self._figure_counter:03d}"
                opts = g_m.group(1) or ""
                path = g_m.group(2)
                width = ""
                if opts:
                    wm = re.search(r"width\s*=\s*([^,}\]]+)", opts)
                    if wm:
                        width = wm.group(1).strip()

                figures.append({
                    "figure_id": fid,
                    "label": label or None,
                    "caption": caption,
                    "local_path": path.strip(),
                    "asset_filename": sanitize_asset_name(fid, fid),
                    "minio_path": None,
                    "width": width or None,
                    "section_id": sec_id,
                })

        # Also handle figure* (double-column) environments
        fig_star_pattern = re.compile(
            r"\\begin\{figure\*\}(.*?)\\end\{figure\*\}", re.DOTALL
        )
        for m in fig_star_pattern.finditer(result.merged_latex):
            fig_content = m.group(1)
            sec_id = self._find_section_id(result, m.start())

            caption = ""
            cap_m = re.search(r"\\caption\s*\{([^}]*)\}", fig_content, re.DOTALL)
            if cap_m:
                caption = cap_m.group(1).strip()

            label = ""
            lab_m = re.search(r"\\label\s*\{([^}]*)\}", fig_content)
            if lab_m:
                label = lab_m.group(1).strip()

            for g_m in re.finditer(
                r"\\includegraphics(?:\[([^\]]*)\])?\s*\{([^}]*)\}", fig_content
            ):
                self._figure_counter += 1
                fid = f"fig-{self._figure_counter:03d}"
                opts = g_m.group(1) or ""
                path = g_m.group(2)
                width = ""
                if opts:
                    wm = re.search(r"width\s*=\s*([^,}\]]+)", opts)
                    if wm:
                        width = wm.group(1).strip()

                figures.append({
                    "figure_id": fid,
                    "label": label or None,
                    "caption": caption,
                    "local_path": path.strip(),
                    "asset_filename": sanitize_asset_name(fid, fid),
                    "minio_path": None,
                    "width": width or None,
                    "section_id": sec_id,
                })

        # Resolve common arXiv figure path variants early so downstream stages
        # can copy/rasterize a real asset instead of carrying a bare LaTeX stem.
        if getattr(result, "source_dir", None):
            for fig in figures:
                resolved = resolve_figure_source(fig.get("local_path"), result.source_dir)
                if resolved:
                    fig["local_path"] = str(resolved)

        return figures

    # ---- references ----

    def _extract_references(self, latex: str) -> list[dict]:
        """Extract references from \\begin{thebibliography}...\\end{thebibliography}."""
        references: list[dict] = []

        # Thebibliography environment
        bib_pattern = re.compile(
            r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}", re.DOTALL
        )
        m = bib_pattern.search(latex)
        if not m:
            return references

        bib_content = m.group(1)
        # Each bibitem
        for item in re.finditer(r"\\bibitem\s*(?:\[([^\]]*)\])?\s*\{([^}]*)\}\s*(.*?)(?=\\bibitem|\Z)", bib_content, re.DOTALL):
            self._ref_counter += 1
            rid = f"ref-{self._ref_counter:03d}"
            bibkey = item.group(2).strip()
            rest = item.group(3).strip()

            # Try to parse year
            year = None
            ym = re.search(r"\b(19|20)\d{2}\b", rest)
            if ym:
                year = int(ym.group(0))

            references.append({
                "ref_id": rid,
                "bibkey": bibkey,
                "title": rest,
                "authors": "",
                "journal": None,
                "year": year,
                "doi": None,
            })

        return references

    # ---- section mapping helper ----

    @staticmethod
    def _find_section_id(result: ParseResult, char_pos: int) -> str:
        """Given a character position in merged_latex, find which section it belongs to."""
        best = ""
        best_end = -1
        for sb in result.all_section_bodies:
            # Approximate: find the section's start position in the merged text
            title_pos = result.merged_latex.find(sb.title)
            if title_pos >= 0 and title_pos > best_end and title_pos <= char_pos:
                best = sb.section_id
                best_end = title_pos

        if not best and result.section_bodies:
            # Default to first section
            return result.section_bodies[0].section_id
        return best

    @staticmethod
    def _find_label(latex: str, pos: int) -> str:
        """Look for \\label{...} near the given position."""
        window = latex[pos : pos + 200]
        m = re.search(r"\\label\s*\{([^}]*)\}", window)
        if m:
            return m.group(1).strip()
        return ""
