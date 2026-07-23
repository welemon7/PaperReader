from __future__ import annotations

import logging
import re

from src.schemas.analysis import PaperAnalysis
from src.schemas.analysis import KeyFormula
from src.schemas.paper import PaperDocument
from src.schemas.poster import (
    FigurePlacement,
    FormulaDisplay,
    PosterBlueprint,
    PosterSection,
)
from src.schemas.analysis import KeyFigure

logger = logging.getLogger(__name__)

POSTER_WIDTH_MM = 841
POSTER_HEIGHT_MM = 1189
POSTER_WIDTH_PX = 1200
POSTER_HEIGHT_PX = 1697

def _clean_poster_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    text = re.sub(r"\\protect\s*", "", text)
    text = _strip_latex_commands(text, {"cite", "ref", "eqref", "autoref", "label"})
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" -;,")
    return text


def _normalize_latex_command_names(text: str) -> str:
    text = re.sub(r"\\Tilde\b", r"\\tilde", text)
    text = re.sub(r"\\Dtilde\b", r"\\tilde", text)
    return text


def _latex_is_balanced(text: str) -> bool:
    pairs = {"{": "}", "(": ")", "[": "]"}
    opens = {"{": 0, "(": 0, "[": 0}
    closes = {"}": "{", ")": "(", "]": "["}
    stack: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 1
            while i < len(text) and text[i].isalpha():
                i += 1
            continue
        if ch in opens:
            stack.append(ch)
        elif ch in closes:
            if not stack or stack[-1] != closes[ch]:
                return False
            stack.pop()
        i += 1
    return not stack


def _clean_formula_latex(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    text = _normalize_latex_command_names(text)
    text = _strip_latex_commands(text, {"cite", "ref", "eqref", "autoref", "label"})
    text = re.sub(r"\\protect\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Drop obvious fragments that would render as broken MathJax output.
    if text.count("$$") % 2 == 1:
        return ""
    if text.count("$") % 2 == 1:
        return ""
    if text.count("{") != text.count("}"):
        return ""
    if not _latex_is_balanced(text):
        return ""
    if re.search(r"\\(?:cite|ref|label|autoref|eqref)\b", text):
        return ""
    if text.count("||") % 2 == 1:
        return ""
    if text.count("\\left|") != text.count("\\right|"):
        return ""
    if text.endswith(("+", "-", "=", ",", "(", "[", "{", "\\", "/")):
        return ""
    return text


def _strip_latex_commands(text: str, commands: set[str]) -> str:
    """Remove citation/reference-style LaTeX commands while preserving nearby text."""
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            j = i + 1
            while j < n and text[j].isalpha():
                j += 1
            cmd = text[i + 1 : j]
            if cmd in commands:
                k = j
                while k < n and text[k] in " \t*":
                    k += 1
                if k < n and text[k] == "{":
                    depth = 1
                    k += 1
                    while k < n and depth > 0:
                        if text[k] == "{":
                            depth += 1
                        elif text[k] == "}":
                            depth -= 1
                        elif text[k] in "\r\n" and depth == 1:
                            break
                        k += 1
                else:
                    while k < n and text[k] not in " \t\r\n,.;:!?)]}":
                        k += 1
                while result and result[-1].isspace():
                    result.pop()
                if result and result[-1] == "~":
                    result.pop()
                i = k
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def generate_blueprint(
    doc: PaperDocument,
    analysis: PaperAnalysis,
    use_gemini: bool = False,
) -> PosterBlueprint:
    analysis = normalize_analysis_for_poster(analysis.model_copy(deep=True))
    _augment_key_formulas(doc, analysis)
    if use_gemini:
        try:
            from src.config import settings
            if settings.gemini_api_key:
                blueprint = _gemini_layout(doc, analysis)
                blueprint.sections = _drop_top_summary_sections(blueprint.sections)
                _normalize_compact_layout(blueprint.sections)
                return blueprint
        except Exception:
            logger.warning("Gemini layout failed, falling back to static layout")
    sections = []
    sections.append(_build_title_section(doc, analysis))
    sections.extend(_build_compact_layout(doc, analysis))
    sections = _drop_top_summary_sections(sections)
    figure_placements = _place_figures(doc, analysis, sections)
    formula_displays = _place_formulas(analysis)
    _tighten_layout(sections, figure_placements)
    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str=_format_authors(doc.authors),
        code_url=analysis.code_url,
        width_px=POSTER_WIDTH_PX, height_px=POSTER_HEIGHT_PX,
        width_mm=POSTER_WIDTH_MM, height_mm=POSTER_HEIGHT_MM,
        sections=sections,
        figure_placements=figure_placements,
        formula_displays=formula_displays,
        color_scheme=_default_colors(),
    )


def _drop_top_summary_sections(sections: list[PosterSection]) -> list[PosterSection]:
    """Remove the three redundant top summary cards from the poster layout."""
    blocked_ids = {"sec-method-overview", "sec-key-idea"}
    blocked_types = {"method_overview", "key_idea"}
    return [
        sec
        for sec in sections
        if sec.section_id not in blocked_ids and sec.type not in blocked_types
    ]


def normalize_analysis_for_poster(analysis: PaperAnalysis) -> PaperAnalysis:
    """Keep poster-facing analysis in English and close to the source paper."""
    for field_name in ("problem_statement", "method_overview", "conclusion", "full_analysis_md"):
        value = getattr(analysis, field_name, "") or ""
        setattr(analysis, field_name, _english_clean(value))

    cleaned_contribs = []
    for contrib in analysis.contributions:
        contrib.text = _english_clean(contrib.text)
        cleaned_contribs.append(contrib)
    analysis.contributions = cleaned_contribs

    if analysis.experiments:
        analysis.experiments.main_results = _english_clean(analysis.experiments.main_results)
        analysis.experiments.takeaways = [_english_clean(x) for x in analysis.experiments.takeaways]
        analysis.experiments.datasets = [_english_clean(x) for x in analysis.experiments.datasets]
        analysis.experiments.metrics = [_english_clean(x) for x in analysis.experiments.metrics]

    analysis.key_figures = [
        KeyFigure(
            figure_id=fig.figure_id,
            caption=_clean_poster_text(fig.caption),
            role=fig.role,
        )
        for fig in analysis.key_figures
    ]

    return analysis


def _build_title_section(doc: PaperDocument, analysis: PaperAnalysis) -> PosterSection:
    authors_line = _format_authors(doc.authors)
    content = doc.title
    if authors_line:
        content = content + "\n\n" + authors_line
    return PosterSection(
        section_id="sec-title", type="title",
        title=doc.title,
        content_md=content,
        column=1, col_span=3, row=0,
    )


def _build_row1(analysis: PaperAnalysis) -> list[PosterSection]:
    """Compatibility wrapper for the compact layout's first row."""
    return [_build_motivation_section(analysis)]


def _build_row2(doc: PaperDocument, analysis: PaperAnalysis) -> list[PosterSection]:
    """Compatibility wrapper for the compact layout's middle row."""
    return [_build_results_section(analysis), PosterSection(
        section_id="sec-main-method", type="main_method",
        title="Method",
        content_md=_join_with_paragraphs(analysis.method_overview, ""),
        column=1, col_span=1, row=2,
    )]


def _build_motivation_section(analysis: PaperAnalysis) -> PosterSection:
    return PosterSection(
        section_id="sec-motivation", type="motivation",
        title="Motivation",
        content_md=analysis.problem_statement or "(not provided)",
        column=1, col_span=1, row=1,
    )


def _build_results_section(analysis: PaperAnalysis) -> PosterSection:
    exp = analysis.experiments
    exp_lines = []
    if exp:
        table_lines = []
        if exp.main_results or exp.datasets or exp.metrics:
            table_lines.append("| **Metric** | **Value** |")
            sep = "|" + "-" * 11 + "|" + "-" * 10 + "|"
            table_lines.append(sep)
            if exp.datasets:
                table_lines.append("| Datasets | " + ", ".join(exp.datasets) + " |")
            if exp.metrics:
                table_lines.append("| Metrics | " + ", ".join(exp.metrics) + " |")
            if exp.main_results:
                table_lines.append("| Best Result | " + exp.main_results + " |")
            table_lines.append("")
        exp_lines.extend(table_lines)
        if exp.takeaways:
            if exp_lines:
                exp_lines.append("")
            exp_lines.append("**Takeaways:**")
            for takeaway in exp.takeaways[:3]:
                exp_lines.append("- " + takeaway)
    exp_content = "\n".join(exp_lines) if exp_lines else "(experimental results not available)"

    return PosterSection(
        section_id="sec-experiments", type="experiments",
        title="Results",
        content_md=exp_content,
        column=2, col_span=1, row=1, row_span=2,
    )


def _build_contributions_section(analysis: PaperAnalysis) -> PosterSection:
    contrib_lines = []
    for c in analysis.contributions:
        prefix = "-"
        if c.category:
            prefix = "- **[" + c.category + "]**"
        contrib_lines.append(prefix + " " + c.text)
    contrib_content = "\n".join(contrib_lines) if contrib_lines else "(not provided)"

    return PosterSection(
        section_id="sec-contributions", type="contributions",
        title="Contribution",
        content_md=contrib_content,
        column=1, col_span=1, row=3,
    )


def _build_highlights_section(analysis: PaperAnalysis) -> PosterSection:
    return PosterSection(
        section_id="sec-highlights", type="highlights",
        title="Key Takeaways",
        content_md="\n".join(_build_highlights(analysis)),
        column=2, col_span=1, row=3,
    )


def _build_compact_layout(doc: PaperDocument, analysis: PaperAnalysis) -> list[PosterSection]:
    formulas_text = ""
    if analysis.key_formulas:
        lines = []
        for f in analysis.key_formulas:
            cleaned_latex = _clean_formula_latex(f.latex)
            if not cleaned_latex:
                continue
            lines.append("- $$ " + cleaned_latex + " $$")
            if f.semantic_desc:
                lines.append("  " + _clean_poster_text(f.semantic_desc))
        if lines:
            formulas_text = "\n\n**Key Formulas:**\n\n" + "\n".join(lines)

    method_detail = PosterSection(
        section_id="sec-main-method", type="main_method",
        title="Method",
        content_md=_join_with_paragraphs(analysis.method_overview, formulas_text),
        column=1, col_span=1, row=2,
    )

    return [
        _build_motivation_section(analysis),
        method_detail,
        _build_results_section(analysis),
        _build_contributions_section(analysis),
        _build_highlights_section(analysis),
    ]


def _build_row3(analysis: PaperAnalysis) -> list[PosterSection]:
    """Compatibility wrapper for the compact layout's last row."""
    return [
        _build_contributions_section(analysis),
        _build_highlights_section(analysis),
    ]


def _place_figures(doc: PaperDocument, analysis: PaperAnalysis, sections: list[PosterSection]) -> list[FigurePlacement]:
    placements = []
    method_hero_taken = False
    prioritized = sorted(
        _figure_candidates(doc, analysis),
        key=lambda f: _figure_priority(_figure_caption(f), _figure_role(f)),
    )
    seen_signatures: set[str] = set()
    for fig in prioritized:
        caption = _figure_caption(fig)
        role = _figure_role(fig)
        signature = _figure_signature(caption, role, getattr(fig, "figure_id", ""))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        if _is_result_figure(caption, role):
            target = "sec-experiments"
            width_ratio = 0.96
        elif _is_method_figure(caption, role):
            target = "sec-main-method"
            width_ratio = 0.98 if not method_hero_taken else 0.72
            method_hero_taken = True
        else:
            if len(placements) >= 3:
                continue
            target = "sec-main-method"
            width_ratio = 0.76 if not method_hero_taken else 0.64
            method_hero_taken = True
        if len(placements) >= 4:
            break
        placements.append(FigurePlacement(
            figure_id=getattr(fig, "figure_id", ""), section_id=target,
            width_ratio=width_ratio, caption=caption,
        ))
    return placements


def _place_formulas(analysis: PaperAnalysis) -> list[FormulaDisplay]:
    displays = []
    for f in analysis.key_formulas:
        cleaned_latex = _clean_formula_latex(f.latex)
        if not cleaned_latex:
            continue
        displays.append(FormulaDisplay(
            formula_id=f.formula_id, section_id="sec-main-method",
            latex=cleaned_latex, semantic_desc=_clean_poster_text(f.semantic_desc),
        ))
    return displays


def _augment_key_formulas(doc: PaperDocument, analysis: PaperAnalysis) -> None:
    """Backfill missing key formulas from the paper's extracted formula index."""
    existing = []
    seen_keys: set[str] = set()
    for f in analysis.key_formulas:
        cleaned_latex = _clean_formula_latex(f.latex)
        if not cleaned_latex:
            continue
        key = _formula_key(cleaned_latex)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        existing.append(KeyFormula(
            formula_id=f.formula_id,
            latex=cleaned_latex,
            semantic_desc=(f.semantic_desc or "").strip(),
        ))

    fallbacks: list[KeyFormula] = []
    for formula in doc.formulas:
        cleaned_latex = _clean_formula_latex(getattr(formula, "latex", ""))
        if not cleaned_latex:
            continue
        key = _formula_key(cleaned_latex)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        fallbacks.append(KeyFormula(
            formula_id=getattr(formula, "formula_id", ""),
            latex=cleaned_latex,
            semantic_desc="",
        ))
        if len(existing) + len(fallbacks) >= 5:
            break

    if len(existing) < 2 and fallbacks:
        analysis.key_formulas = (existing + fallbacks)[:5]
    else:
        analysis.key_formulas = existing[:5]


def _formula_key(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    text = re.sub(r"\\label\s*\{[^{}]*\}", "", text)
    return text


def _tighten_layout(sections: list[PosterSection], figure_placements: list[FigurePlacement]) -> None:
    """Slightly rebalance the static layout toward denser sections.

    The goal is to reduce large empty columns/rows while keeping the existing
    reading order and section semantics unchanged.
    """
    fig_count_by_section: dict[str, int] = {}
    for fp in figure_placements:
        fig_count_by_section[fp.section_id] = fig_count_by_section.get(fp.section_id, 0) + 1

    for sec in sections:
        if sec.type == "title":
            continue
        text_len = len((sec.content_md or "").strip())
        figure_bonus = fig_count_by_section.get(sec.section_id, 0)
        density = text_len // 280 + figure_bonus * 2

        if sec.type in {"method_overview", "main_method"} and density >= 3:
            sec.col_span = max(sec.col_span, 2)
        elif sec.type in {"experiments", "contributions"} and density <= 2:
            sec.col_span = min(sec.col_span, 1)

        if sec.type == "highlights" and text_len < 180:
            sec.col_span = 1

        if sec.type in {"motivation", "key_idea"} and density >= 2:
            sec.col_span = max(sec.col_span, 1)


def _normalize_compact_layout(sections: list[PosterSection]) -> None:
    """Pin the default poster layout to the compact two-column story grid."""
    for sec in sections:
        if sec.type == "title":
            sec.column = 1
            sec.col_span = 2
            sec.row = 0
            sec.row_span = 1
        elif sec.type == "motivation":
            sec.column = 1
            sec.col_span = 1
            sec.row = 1
            sec.row_span = 1
        elif sec.type == "main_method":
            sec.column = 1
            sec.col_span = 1
            sec.row = 2
            sec.row_span = 1
        elif sec.type == "experiments":
            sec.column = 2
            sec.col_span = 1
            sec.row = 1
            sec.row_span = 2
        elif sec.type == "contributions":
            sec.column = 1
            sec.col_span = 1
            sec.row = 3
            sec.row_span = 1
        elif sec.type == "highlights":
            sec.column = 2
            sec.col_span = 1
            sec.row = 3
            sec.row_span = 1
        else:
            sec.row_span = max(getattr(sec, "row_span", 1), 1)


def _figure_priority(caption: str, role: str) -> tuple[int, int]:
    text = f"{caption} {role}".lower()
    if any(k in text for k in ("framework", "overview", "architecture", "pipeline", "model overview", "main architecture")):
        return (0, 0)
    if any(k in text for k in ("introduction", "intro", "motivation")):
        return (1, 0)
    if any(k in text for k in ("result", "comparison", "qualitative", "experiment", "accuracy", "ablation", "benchmark", "performance", "table", "metric", "seg")):
        return (2, 0)
    return (3, 0)


def _figure_signature(caption: str, role: str, figure_id: str = "") -> str:
    text = f"{caption} {role}".lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if figure_id:
        return f"{figure_id.lower()}::{text[:120]}"
    return text[:120]


def _figure_candidates(doc: PaperDocument, analysis: PaperAnalysis) -> list:
    candidates = []
    seen_ids: set[str] = set()
    for fig in list(analysis.key_figures) + list(doc.figures):
        fig_id = getattr(fig, "figure_id", "") or ""
        if fig_id and fig_id in seen_ids:
            continue
        if fig_id:
            seen_ids.add(fig_id)
        candidates.append(fig)
    return candidates


def _result_visual_candidates(doc: PaperDocument, analysis: PaperAnalysis) -> list:
    return [fig for fig in _figure_candidates(doc, analysis) if _is_result_figure(_figure_caption(fig), _figure_role(fig))]


def _figure_caption(fig) -> str:
    return _clean_poster_text(getattr(fig, "caption", "") or "")


def _figure_role(fig) -> str:
    role = getattr(fig, "role", "") or ""
    if role:
        return role
    caption = _figure_caption(fig).lower()
    section_id = (getattr(fig, "section_id", "") or "").lower()
    text = f"{caption} {section_id}"
    if any(k in text for k in ("framework", "overview", "architecture", "pipeline", "method overview", "main architecture")):
        return "architecture"
    if any(k in text for k in ("result", "comparison", "qualitative", "experiment", "accuracy", "ablation", "benchmark", "performance", "table", "metric", "seg")):
        return "result"
    return ""


def _is_method_figure(caption: str, role: str) -> bool:
    text = f"{caption} {role}".lower()
    return any(k in text for k in ("framework", "overview", "architecture", "pipeline", "model overview", "main architecture"))


def _is_result_figure(caption: str, role: str) -> bool:
    text = f"{caption} {role}".lower()
    return any(k in text for k in ("result", "comparison", "qualitative", "experiment", "accuracy", "ablation", "benchmark", "performance", "table", "metric", "seg"))


def _format_authors(authors) -> str:
    names = []
    for author in authors or []:
        name = getattr(author, "name", "") or ""
        name = re.sub(r"\\textsuperscript\s*\{.*?\}", "", name, flags=re.DOTALL)
        name = re.sub(r"\\(?:thanks|footnote)\{.*?\}", "", name, flags=re.DOTALL)
        name = re.sub(r"\\(?:inst|email|correspondingauthor)\b", "", name)
        name = re.sub(r"\\[a-zA-Z]+", "", name)
        name = name.replace("{", "").replace("}", "")
        name = re.sub(r"\s+", " ", name).strip()
        if name:
            names.append(name)
    return "; ".join(names)


def _english_clean(text: str) -> str:
    text = _clean_poster_text(text)
    if not text:
        return text
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -;,")
    return text


def _build_highlights(analysis: PaperAnalysis) -> list[str]:
    lines: list[str] = []
    if analysis.contributions:
        for i, c in enumerate(analysis.contributions[:3], 1):
            lines.append(f"{i}. {c.text}")
    if analysis.experiments and analysis.experiments.takeaways:
        if lines:
            lines.append("")
        start = len(lines)
        for i, t in enumerate(analysis.experiments.takeaways[:3], start + 1):
            lines.append(f"{i}. {t}")
    if not lines:
        if analysis.problem_statement:
            lines.append(f"1. {analysis.problem_statement}")
        elif analysis.method_overview:
            lines.append(f"1. {analysis.method_overview[:180]}")
        else:
            lines.append("1. See paper for details.")
    return lines


def _join_with_paragraphs(*parts: str) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(cleaned)

_GEMINI_LAYOUT_PROMPT = (
    "You are a scientific poster layout designer. "
    "Given a paper analysis, design the optimal poster layout.\n\n"
    "Return JSON with:\n"
    '- "sections": list of {section_id, type, title, column, col_span, row} '
    "- design the grid layout\n"
    '- "figure_placements": list of {figure_id, section_id, width_ratio, caption} '
    "- max 4 figures\n"
    '- "color_scheme": dict with primary, accent, background, text, '
    "section_header_bg, section_header_text, border, highlight\n\n"
    "Layout rules:\n"
    f"- Poster size is fixed to A0 portrait: {POSTER_WIDTH_MM}mm x {POSTER_HEIGHT_MM}mm ({POSTER_WIDTH_PX}px x {POSTER_HEIGHT_PX}px)\n"
    "- Core architecture/overview figures: place in method section, "
    "width_ratio >= 0.8\n"
    "- Result/comparison figures: group in experiments section, "
    "width_ratio 0.45-0.55\n"
    "- Max 4 figures total. Remove redundant ones.\n"
    "- Design a grid that tells a visual story: motivation -> method "
    "-> results -> contributions\n"
    "- Use 2-3 rows with flexible column spans"
)


def _gemini_layout(
    doc: PaperDocument,
    analysis: PaperAnalysis,
) -> PosterBlueprint:
    """Use Gemini to design the poster layout based on paper analysis."""
    from src.config import settings
    from src.llm.client import LLMClient

    gemini = LLMClient(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        model=settings.gemini_model,
    )

    prompt_parts = []
    prompt_parts.append(f"Paper: {doc.title}")
    prompt_parts.append(f"\nProblem: {analysis.problem_statement}")
    prompt_parts.append("\nContributions:")
    for contrib in analysis.contributions:
        prompt_parts.append(f"- {contrib.text}")
    prompt_parts.append(f"\nMethod: {analysis.method_overview}")
    if analysis.code_url:
        prompt_parts.append(f"\nCode / Project URL: {analysis.code_url}")
    prompt_parts.append("\nKey Figures:")
    for fig in analysis.key_figures:
        prompt_parts.append(f"- [{fig.figure_id}] {fig.caption} ({fig.role})")
    if doc.figures:
        prompt_parts.append("\nPaper figure index:")
        for fig in doc.figures[:12]:
            label = getattr(fig, "label", None) or "(no label)"
            prompt_parts.append(f"- [{fig.figure_id}] {label}: {fig.caption}")
    if analysis.experiments:
        prompt_parts.append("\nExperiments:")
        exp = analysis.experiments
        if exp.datasets:
            prompt_parts.append(f"Datasets: {', '.join(exp.datasets)}")
        prompt_parts.append(f"Results: {exp.main_results}")

    user_prompt = "\n".join(prompt_parts)
    try:
        result = gemini.chat_json(system=_GEMINI_LAYOUT_PROMPT, user=user_prompt)
    except Exception as e:
        logger.warning("Gemini layout call failed: %s, using static fallback", e)
        return _static_layout(doc, analysis)

    sections = []
    for s in result.get("sections", []):
        sections.append(PosterSection(
            section_id=s.get("section_id", ""),
            type=s.get("type", "motivation"),
            title=s.get("title", ""),
            content_md=s.get("content_md", ""),
            column=s.get("column", 1),
            col_span=s.get("col_span", 1),
            row=s.get("row", 0),
        ))

    figure_placements = []
    for fp in result.get("figure_placements", []):
        figure_placements.append(FigurePlacement(
            figure_id=fp.get("figure_id", ""),
            section_id=fp.get("section_id", ""),
            width_ratio=fp.get("width_ratio", 0.9),
            caption=fp.get("caption", ""),
        ))

    formula_displays = _place_formulas(analysis)
    color_scheme = result.get("color_scheme", _default_colors())

    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str="; ".join(a.name for a in doc.authors),
        code_url=analysis.code_url,
        width_px=POSTER_WIDTH_PX, height_px=POSTER_HEIGHT_PX,
        width_mm=POSTER_WIDTH_MM, height_mm=POSTER_HEIGHT_MM,
        sections=sections,
        figure_placements=figure_placements,
        formula_displays=formula_displays,
        color_scheme=color_scheme,
    )


def _static_layout(
    doc: PaperDocument,
    analysis: PaperAnalysis,
) -> PosterBlueprint:
    """Static fallback layout when Gemini is unavailable."""
    sections = []
    sections.append(_build_title_section(doc, analysis))
    sections.extend(_build_compact_layout(doc, analysis))
    sections = _drop_top_summary_sections(sections)
    figure_placements = _place_figures(doc, analysis, sections)
    formula_displays = _place_formulas(analysis)
    _normalize_compact_layout(sections)
    _tighten_layout(sections, figure_placements)
    _normalize_compact_layout(sections)
    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str="; ".join(a.name for a in doc.authors),
        code_url=analysis.code_url,
        width_px=POSTER_WIDTH_PX, height_px=POSTER_HEIGHT_PX,
        width_mm=POSTER_WIDTH_MM, height_mm=POSTER_HEIGHT_MM,
        sections=sections,
        figure_placements=figure_placements,
        formula_displays=formula_displays,
        color_scheme=_default_colors(),
    )


def _default_colors() -> dict:
    return {
        "primary": "#16324f",
        "accent": "#5a7d9a",
        "background": "#fbfcfe",
        "text": "#182433",
        "section_header_bg": "#e8eef4",
        "section_header_text": "#16324f",
        "border": "#cfd8e3",
        "highlight": "#8fb3d9",
    }
