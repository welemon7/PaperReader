from __future__ import annotations

import html as html_lib
import logging
import os
import re
from pathlib import Path
from typing import Optional

import markdown as md_lib
from jinja2 import Environment, FileSystemLoader

from src.llm.client import LLMClient, LLMError
from src.schemas.paper import PaperDocument
from src.schemas.poster import PosterBlueprint, FigurePlacement
from src.utils.figure_assets import copy_or_rasterize_asset, resolve_figure_source, sanitize_asset_name

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_DEFAULT_HTML_OPTIMIZER_PROMPT = Path(__file__).resolve().parents[2] / "example" / "LLM-up.txt"




class HtmlPosterRenderer:
    def __init__(self, template_dir: str = _TEMPLATE_DIR, optimizer_prompt_path: Path | None = None) -> None:
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.env.filters["summarize_text"] = self._summarize_text
        self.template = self.env.get_template("poster.html.j2")
        self.optimizer_prompt_path = Path(optimizer_prompt_path) if optimizer_prompt_path else _DEFAULT_HTML_OPTIMIZER_PROMPT

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
        """Return a short readable summary without adding ellipses."""
        if not text:
            return ""

        # 清理HTML标签和实体
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = html_lib.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned:
            return ""

        sentence_endings = r"[.!?。！？]"
        sentences = re.split(f"(?<={sentence_endings})\\s+", cleaned)

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
        text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
        text = re.sub(r"\\textit\{([^{}]*)\}", r"*\1*", text)
        text = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", text)
        text = re.sub(r"\\(?:text)?tt\{([^{}]*)\}", r"`\1`", text)
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
        optimize_with_llm: bool = False,
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
        layout = self._build_layout(blueprint)
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

        for sec in layout:
            sec.content_md = self._clean_html_text(sec.content_md)
            sec.content_html = self._markdown_with_latex(sec.content_md)

        html = self.template.render(
            poster_title=blueprint.poster_title,
            authors_str=blueprint.authors_str,
            code_url=blueprint.code_url,
            poster_width=blueprint.width_px,
            poster_height=blueprint.height_px,
            color_scheme=blueprint.color_scheme,
            layout=layout,
            figure_map=figure_map,
        )

        if optimize_with_llm:
            html = self._optimize_html_with_llm(html, doc, blueprint, output_dir)

        return html

    def render_to_file(
        self,
        blueprint: PosterBlueprint,
        doc: PaperDocument,
        output_path: Path,
        optimize_with_llm: bool = False,
    ) -> Path:
        html = self.render(blueprint, doc, output_path.parent, optimize_with_llm=optimize_with_llm)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Poster HTML saved to %s", output_path)
        return output_path

    def _optimize_html_with_llm(
        self,
        html: str,
        doc: PaperDocument,
        blueprint: PosterBlueprint,
        output_dir: Path,
    ) -> str:
        if not LLMClient.is_configured():
            logger.info("LLM optimizer disabled: API key not configured")
            return html

        try:
            instruction = self.optimizer_prompt_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("LLM optimizer prompt missing, skipping optimization: %s", e)
            return html

        client = LLMClient()
        system_prompt = (
            "You are a strict HTML poster editor. "
            "Return a complete, standalone HTML document only. "
            "Preserve the paper content, figures, and math rendering. "
            "Do not add explanations, markdown, or code fences."
        )
        user_prompt = (
            f"{instruction.strip()}\n\n"
            f"## Paper\nTitle: {doc.title}\nArXiv: {doc.arxiv_id or doc.paper_id}\n"
            f"## Poster Blueprint\nTitle: {blueprint.poster_title}\n"
            f"Authors: {blueprint.authors_str}\n"
            f"Output directory: {output_dir.as_posix()}\n\n"
            "## Initial HTML\n"
            f"{html}"
        )

        try:
            optimized = client.chat(system=system_prompt, user=user_prompt)
        except LLMError as e:
            logger.warning("HTML optimization skipped: %s", e)
            return html

        optimized = optimized.strip()
        if not optimized:
            logger.warning("HTML optimization returned empty content; keeping original HTML")
            return html
        if "<html" not in optimized.lower() or "</html>" not in optimized.lower():
            logger.warning("HTML optimization did not return a full document; keeping original HTML")
            return html

        logger.info("HTML optimized with LLM (%d -> %d chars)", len(html), len(optimized))
        return optimized

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
                logger.warning("Skipping figure %s: source asset not found", getattr(fig, "figure_id", ""))
                fig.local_path = None
                continue
            prepared = self._ensure_browser_asset(
                candidate,
                fig_dir,
                getattr(fig, "asset_filename", None) or fig.figure_id,
            )
            if prepared and prepared.resolve().is_relative_to(fig_dir.resolve()):
                fig.local_path = str(prepared)
            else:
                logger.warning("Skipping figure %s: normalized asset missing under figures/", getattr(fig, "figure_id", ""))
                fig.local_path = None

    def _ensure_browser_asset(self, src: Path, out_dir: Path, figure_id: str) -> Path | None:
        src = src.resolve()
        if not src.exists():
            return None

        target_name = sanitize_asset_name(figure_id or src.stem, src.stem)
        prepared = copy_or_rasterize_asset(src, out_dir, target_name)
        if prepared:
            return prepared
        return None

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
            cls._normalize_figure_key(getattr(fig, "asset_filename", "")),
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

        # 最后兜底：caption 关键词重叠匹配（容错 label/id 不完全一致的情况）
        fp_tokens = cls._caption_tokens(fp.caption)
        if fp_tokens:
            best_fig = None
            best_score = 0
            for fig in figures:
                if getattr(fig, "figure_id", None) in used_ids:
                    continue
                score = len(fp_tokens & cls._caption_tokens(getattr(fig, "caption", "")))
                if score > best_score:
                    best_score = score
                    best_fig = fig
            if best_score >= 2 and best_fig is not None:
                return best_fig

        for fig in figures:
            if getattr(fig, "figure_id", None) not in used_ids:
                return fig
        return None

    @classmethod
    def _caption_tokens(cls, text: str | None) -> set[str]:
        if not text:
            return set()
        return {
            t for t in re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
            if len(t) > 2
        }

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
        """Compatibility wrapper that returns row-grouped sections."""
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
    def _build_layout(blueprint: PosterBlueprint) -> list:
        sections = [sec for sec in blueprint.sections if sec.type != "title"]
        return sorted(sections, key=lambda s: (s.row, s.column))

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
                    figures_dir = (output_dir / "figures").resolve()
                    if resolved:
                        if resolved.suffix.lower() == ".pdf":
                            target_name = getattr(fig, "asset_filename", None) or fig.figure_id
                            resolved = copy_or_rasterize_asset(resolved, output_dir / "figures", target_name) or resolved
                        if resolved.resolve().is_relative_to(figures_dir):
                            entry["src"] = HtmlPosterRenderer._browser_asset_uri(resolved, output_dir)
                        else:
                            logger.warning("Dropping non-normalized figure asset for %s", fp.figure_id)
                            entry["src"] = None
            # 只保留可渲染的图条目；无法解析的图不留占位、不占版面
            if entry["src"]:
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



