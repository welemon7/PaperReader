from __future__ import annotations

import logging

from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import (
    FigurePlacement,
    FormulaDisplay,
    PosterBlueprint,
    PosterSection,
)

logger = logging.getLogger(__name__)


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
    sections.extend(_build_row2(analysis))
    sections.extend(_build_row3(analysis))
    figure_placements = _place_figures(analysis, sections)
    formula_displays = _place_formulas(analysis)
    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str="; ".join(a.name for a in doc.authors),
        width_px=1200, height_px=1697,
        width_mm=841, height_mm=1189,
        sections=sections,
        figure_placements=figure_placements,
        formula_displays=formula_displays,
        color_scheme=_default_colors(),
    )


def _build_title_section(doc: PaperDocument, analysis: PaperAnalysis) -> PosterSection:
    authors_line = "; ".join(a.name for a in doc.authors)
    content = doc.title
    if authors_line:
        content = content + "\n\n" + authors_line
    if analysis.title_zh and analysis.title_zh != doc.title:
        content = content + "\n\n" + analysis.title_zh
    return PosterSection(
        section_id="sec-title", type="title",
        title=doc.title,
        content_md=content,
        column=1, col_span=3, row=0,
    )


def _build_row1(analysis: PaperAnalysis) -> list[PosterSection]:
    motiv = PosterSection(
        section_id="sec-motivation", type="motivation",
        title="Motivation",
        content_md=analysis.problem_statement or "(not provided)",
        column=1, col_span=1, row=1,
    )
    method_ov = PosterSection(
        section_id="sec-method-overview", type="method_overview",
        title="Method Overview",
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
        title="Key Idea",
        content_md=key_idea_text,
        column=3, col_span=1, row=1,
    )
    return [motiv, method_ov, key_idea]


def _build_row2(analysis: PaperAnalysis) -> list[PosterSection]:
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
    exp_content = "\n".join(exp_lines) if exp_lines else "(experimental results not available)"

    experiments = PosterSection(
        section_id="sec-experiments", type="experiments",
        title="Experiments",
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

    hl_lines = []
    if analysis.experiments and analysis.experiments.takeaways:
        for i, t in enumerate(analysis.experiments.takeaways, 1):
            hl_lines.append(str(i) + ". " + t)
    else:
        hl_lines.append("See paper for details.")

    highlights = PosterSection(
        section_id="sec-highlights", type="highlights",
        title="Highlights",
        content_md="\n".join(hl_lines),
        column=2, col_span=1, row=3,
    )

    code_link_text = analysis.code_url if analysis.code_url else "Code will be available at paper project page (if applicable)."
    if analysis.code_url:
        code_link_md = f"[{analysis.code_url}]({analysis.code_url})"
    else:
        code_link_md = code_link_text
    proj_link = PosterSection(
        section_id="sec-project-link", type="project_link",
        title="Code / Project",
        content_md=code_link_md,
        column=3, col_span=1, row=3,
    )
    return [contributions, highlights, proj_link]


def _place_figures(analysis: PaperAnalysis, sections: list[PosterSection]) -> list[FigurePlacement]:
    placements = []
    for fig in analysis.key_figures:
        if fig.role in ("result", "qualitative", "comparison"):
            target = "sec-experiments"
        elif fig.role in ("overview", "architecture"):
            target = "sec-main-method"
        elif fig.role in ("pipeline", "illustration"):
            target = "sec-method-overview"
        else:
            target = "sec-main-method"
        placements.append(FigurePlacement(
            figure_id=fig.figure_id, section_id=target,
            width_ratio=0.95, caption=fig.caption,
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
    prompt_parts.append("\nKey Figures:")
    for fig in analysis.key_figures:
        prompt_parts.append(f"- [{fig.figure_id}] {fig.caption} ({fig.role})")
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
        width_px=1200, height_px=1697,
        width_mm=841, height_mm=1189,
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
    sections.extend(_build_row2(analysis))
    sections.extend(_build_row3(analysis))
    figure_placements = _place_figures(analysis, sections)
    formula_displays = _place_formulas(analysis)
    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str="; ".join(a.name for a in doc.authors),
        width_px=1200, height_px=1697,
        width_mm=841, height_mm=1189,
        sections=sections,
        figure_placements=figure_placements,
        formula_displays=formula_displays,
        color_scheme=_default_colors(),
    )


def _default_colors() -> dict:
    return {
        "primary": "#1a5276",
        "accent": "#2980b9",
        "background": "#ffffff",
        "text": "#2c3e50",
        "section_header_bg": "#1a5276",
        "section_header_text": "#ffffff",
        "border": "#d5dbdb",
        "highlight": "#f39c12",
    }