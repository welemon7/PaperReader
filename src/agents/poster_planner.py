from __future__ import annotations

import logging
import re

from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import (
    FigurePlacement,
    FormulaDisplay,
    PosterBlueprint,
    PosterSection,
)

logger = logging.getLogger(__name__)

POSTER_WIDTH_MM = 841
POSTER_HEIGHT_MM = 1189
POSTER_WIDTH_PX = 1200
POSTER_HEIGHT_PX = 1697


def generate_blueprint(
    doc: PaperDocument,
    analysis: PaperAnalysis,
    use_gemini: bool = False,
) -> PosterBlueprint:
    if use_gemini:
        try:
            from src.config import settings
            if settings.gemini_api_key:
                return _gemini_layout(doc, analysis)
        except Exception:
            logger.warning("Gemini layout failed, falling back to static layout")
    sections = []
    sections.append(_build_title_section(doc, analysis))
    sections.extend(_build_row1(analysis))
    sections.extend(_build_row2(doc, analysis))
    sections.extend(_build_row3(analysis))
    figure_placements = _place_figures(doc, analysis, sections)
    formula_displays = _place_formulas(analysis)
    _tighten_layout(sections, figure_placements)
    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str=_format_authors(doc.authors),
        width_px=POSTER_WIDTH_PX, height_px=POSTER_HEIGHT_PX,
        width_mm=POSTER_WIDTH_MM, height_mm=POSTER_HEIGHT_MM,
        sections=sections,
        figure_placements=figure_placements,
        formula_displays=formula_displays,
        color_scheme=_default_colors(),
    )


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
    motiv = PosterSection(
        section_id="sec-motivation", type="motivation",
        title="Problem",
        content_md=analysis.problem_statement or "(not provided)",
        column=1, col_span=1, row=1,
    )
    method_ov = PosterSection(
        section_id="sec-method-overview", type="method_overview",
        title="Core Idea",
        content_md=analysis.method_overview or "(not provided)",
        column=2, col_span=1, row=1,
    )
    key_idea_text = (
        analysis.contributions[0].text
        if analysis.contributions
        else analysis.method_overview[:200]
    )
    key_idea = PosterSection(
        section_id="sec-key-idea", type="key_idea",
        title="Why It Matters",
        content_md=key_idea_text,
        column=3, col_span=1, row=1,
    )
    return [motiv, method_ov, key_idea]


def _build_row2(doc: PaperDocument, analysis: PaperAnalysis) -> list[PosterSection]:
    formulas_text = ""
    if analysis.key_formulas:
        lines = []
        for f in analysis.key_formulas:
            lines.append("- $$ " + f.latex + " $$")
            if f.semantic_desc:
                lines.append("  " + f.semantic_desc)
        formulas_text = "\n\n**Key Formulas:**\n" + "\n".join(lines)

    method_detail = PosterSection(
        section_id="sec-main-method", type="main_method",
        title="Method",
        content_md=analysis.method_overview + formulas_text,
        column=1, col_span=2, row=2,
    )

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
            exp_lines.append("**Takeaways:**")
            for t in exp.takeaways:
                exp_lines.append("- " + t)
        if analysis.key_figures or doc.figures:
            exp_lines.append("")
            exp_lines.append("**Recommended visual focus:**")
            for fig in _result_visual_candidates(doc, analysis)[:3]:
                exp_lines.append(f"- {_figure_caption(fig)}")
    exp_content = "\n".join(exp_lines) if exp_lines else "(experimental results not available)"

    experiments = PosterSection(
        section_id="sec-experiments", type="experiments",
        title="Results",
        content_md=exp_content,
        column=3, col_span=1, row=2,
    )
    return [method_detail, experiments]


def _build_row3(analysis: PaperAnalysis) -> list[PosterSection]:
    contrib_lines = []
    for c in analysis.contributions:
        prefix = "-"
        if c.category:
            prefix = "- **[" + c.category + "]**"
        contrib_lines.append(prefix + " " + c.text)
    contrib_content = "\n".join(contrib_lines) if contrib_lines else "(not provided)"

    contributions = PosterSection(
        section_id="sec-contributions", type="contributions",
        title="Contributions",
        content_md=contrib_content,
        column=1, col_span=1, row=3,
    )

    hl_lines = _build_highlights(analysis)

    highlights = PosterSection(
        section_id="sec-highlights", type="highlights",
        title="Key Takeaways",
        content_md="\n".join(hl_lines),
        column=2, col_span=1, row=3,
    )

    if analysis.code_url:
        code_link_md = f"Project / code: [{analysis.code_url}]({analysis.code_url})"
    else:
        code_link_md = "Project / code: not found in the paper text."
    proj_link = PosterSection(
        section_id="sec-project-link", type="project_link",
        title="Code / Project",
        content_md=code_link_md,
        column=3, col_span=1, row=3,
    )
    return [contributions, highlights, proj_link]


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
        displays.append(FormulaDisplay(
            formula_id=f.formula_id, section_id="sec-main-method",
            latex=f.latex, semantic_desc=f.semantic_desc,
        ))
    return displays


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

        if sec.type in {"highlights", "project_link"} and text_len < 180:
            sec.col_span = 1

        if sec.type in {"motivation", "key_idea"} and density >= 2:
            sec.col_span = max(sec.col_span, 1)


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
    return getattr(fig, "caption", "") or ""


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
    text = (text or "").strip()
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

    if sections and not any(sec.section_id == "sec-project-link" for sec in sections):
        sections.append(PosterSection(
            section_id="sec-project-link",
            type="project_link",
            title="Code / Project",
            content_md=(f"Project / code: [{analysis.code_url}]({analysis.code_url})" if analysis.code_url else "Project / code: not found in the paper text."),
            column=3,
            col_span=1,
            row=3,
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
    sections.extend(_build_row1(analysis))
    sections.extend(_build_row2(doc, analysis))
    sections.extend(_build_row3(analysis))
    figure_placements = _place_figures(doc, analysis, sections)
    formula_displays = _place_formulas(analysis)
    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str="; ".join(a.name for a in doc.authors),
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
