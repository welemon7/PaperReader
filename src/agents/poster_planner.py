from __future__ import annotations

import html as html_lib
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
from src.agents.content_policy import BULLET_WORD_BUDGET, trim_to_budget

logger = logging.getLogger(__name__)

# 48 x 27 inch landscape canvas.  The previous A0 portrait defaults made the
# generated layout fundamentally different from a conference-poster reading
# path (wide title band -> three columns -> prominent centre panel).
POSTER_WIDTH_MM = 1219
POSTER_HEIGHT_MM = 686
POSTER_WIDTH_PX = 1920
POSTER_HEIGHT_PX = 1080


def _short_bullet(text: str, max_words: int = BULLET_WORD_BUDGET) -> str:
    """One short, scannable bullet: first sentence, capped at max_words."""
    text = _clean_poster_text(text)
    if not text:
        return ""
    sentence = _first_sentence(text)
    return trim_to_budget(sentence, max_words)

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
) -> PosterBlueprint:
    """Generate poster blueprint using static layout only."""
    analysis = normalize_analysis_for_poster(analysis.model_copy(deep=True))
    _augment_key_formulas(doc, analysis)
    sections = []
    sections.append(_build_title_section(doc, analysis))
    sections.extend(_build_compact_layout(doc, analysis))
    figure_placements = _place_figures(doc, analysis, sections)
    formula_displays = _place_formulas(analysis)
    _tighten_layout(sections, figure_placements)
    _normalize_compact_layout(sections)
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
    """Compatibility shim kept for older callers.

    The poster now keeps the method overview and key idea blocks in the top row,
    so this helper no longer removes them.
    """
    return sections


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
    return [
        _build_motivation_section(analysis),
        _build_method_overview_section(analysis),
        _build_key_idea_section(analysis),
    ]


def _build_row2(doc: PaperDocument, analysis: PaperAnalysis) -> list[PosterSection]:
    """Compatibility wrapper for the compact layout's middle row."""
    return [_build_core_section(analysis)]


def _build_motivation_section(analysis: PaperAnalysis) -> PosterSection:
    motivation = _build_motivation_content(analysis)
    return PosterSection(
        section_id="sec-motivation", type="motivation",
        title="Motivation",
        content_md=motivation or "(not provided)",
        column=1, col_span=1, row=1,
    )


def _build_method_overview_section(analysis: PaperAnalysis) -> PosterSection:
    content = _build_method_overview_content(analysis) or "(method overview not provided)"
    return PosterSection(
        section_id="sec-method-overview", type="method_overview",
        title="Method Overview",
        content_md=content,
        column=2, col_span=1, row=1,
    )


def _build_key_idea_section(analysis: PaperAnalysis) -> PosterSection:
    key_title = _infer_key_idea_title(analysis)
    content = _build_key_idea_content(analysis) or "(key idea not provided)"
    return PosterSection(
        section_id="sec-key-idea", type="key_idea",
        title=f"Key Idea: {key_title}",
        content_md=content,
        column=3, col_span=1, row=1,
    )


def _build_core_section(doc: PaperDocument, analysis: PaperAnalysis) -> PosterSection:
    content = _build_core_results_content(doc, analysis) or "(core method not provided)"
    return PosterSection(
        section_id="sec-main-method", type="main_method",
        title="Core Results",
        content_md=content,
        column=1, col_span=3, row=2,
    )



def _build_result_block(analysis: PaperAnalysis) -> str:
    """Build a result summary block from analysis."""
    if not analysis.experiments:
        return ""

    parts = []
    exp = analysis.experiments

    # Main results
    if exp.main_results:
        parts.append(f"**Main Results:** {_clean_poster_text(exp.main_results)}")

    # Datasets
    if exp.datasets:
        parts.append(f"**Datasets:** {', '.join(_clean_poster_text(d) for d in exp.datasets)}")

    # Metrics
    if exp.metrics:
        parts.append(f"**Metrics:** {', '.join(_clean_poster_text(m) for m in exp.metrics)}")

    # Takeaways
    if exp.takeaways:
        takeaways = [_clean_poster_text(t) for t in exp.takeaways[:3]]
        parts.append("**Key Takeaways:**")
        parts.extend(f"- {t}" for t in takeaways)

    return "\n".join(parts) if parts else ""

def _build_results_section(analysis: PaperAnalysis) -> PosterSection:
    """Compatibility wrapper for the result summary used by older tests."""
    return PosterSection(
        section_id="sec-experiments", type="experiments",
        title="Result",
        content_md=_build_result_block(analysis) or "(experimental results not available)",
        column=1, col_span=1, row=2,
    )


def _build_contributions_section(analysis: PaperAnalysis) -> PosterSection:
    contrib_content = _build_contributions_content(analysis) or "(not provided)"

    return PosterSection(
        section_id="sec-contributions", type="contributions",
        title="Contributions",
        content_md=contrib_content,
        column=1, col_span=1, row=3,
    )


def _build_highlights_section(analysis: PaperAnalysis) -> PosterSection:
    content = _build_highlights_content(analysis) or "- See paper for details."
    return PosterSection(
        section_id="sec-highlights", type="highlights",
        title="Highlights",
        content_md=content,
        column=2, col_span=1, row=3,
    )


def _build_project_section(analysis: PaperAnalysis) -> PosterSection:
    content = _build_project_content(analysis)
    return PosterSection(
        section_id="sec-project", type="project_link",
        title="Project",
        content_md=content,
        column=3, col_span=1, row=3,
    )


def _summarize_motivation(analysis: PaperAnalysis, max_words: int = 80) -> str:
    """Return a compact motivation teaser without inline emphasis."""
    problem = _first_sentence(_clean_poster_text(analysis.problem_statement or ""))
    method_hint = _first_sentence(_clean_poster_text(analysis.method_overview or ""))
    result_hint = _result_hint(analysis)

    parts: list[str] = []
    if problem:
        parts.append(problem.rstrip(" ,;:.-"))
    if method_hint:
        parts.append(f"The paper addresses this by {method_hint.rstrip(' ,;:.-')}.")
    if result_hint:
        parts.append(result_hint)

    text = " ".join(parts).strip()
    if not text:
        return ""

    return text.rstrip(" ,;:.-")


def _build_motivation_content(analysis: PaperAnalysis) -> str:
    paragraphs: list[str] = []
    problem = _first_sentence(_clean_poster_text(analysis.problem_statement or ""))
    if problem:
        paragraphs.append(problem)

    advantage_source = _first_clause(_clean_poster_text(analysis.method_overview or ""))
    if not advantage_source and analysis.contributions:
        advantage_source = _first_clause(_clean_poster_text(analysis.contributions[0].text))
    if advantage_source:
        paragraphs.append(f"The core advantage is that {advantage_source.rstrip(' ,;:.-')}.")

    prose = " ".join(p for p in paragraphs if p).strip()
    prose = trim_to_budget(prose, 40)

    formula_html = _format_formula_box(analysis, label="Key Formula", formula_index=0)
    parts = [prose] if prose else []
    if formula_html:
        parts.append(formula_html)
    else:
        # 无公式时才用 callout 补充关键洞见，避免文字堆叠
        callout_text = _build_motivation_callout(analysis)
        if callout_text:
            parts.append(callout_text)
    return "\n\n".join(parts).strip()


def _build_method_overview_content(analysis: PaperAnalysis) -> str:
    intro = _first_sentence(_clean_poster_text(analysis.method_overview or ""))
    if not intro:
        intro = "We summarize the method as a compact pipeline from input to output."

    bullet_points: list[str] = []
    clause = _first_clause(_clean_poster_text(analysis.method_overview or ""))
    if clause:
        bullet_points.append(f"- {_short_bullet(clause, 18)}")
    if len(bullet_points) < 2:
        for contrib in analysis.contributions[:2]:
            item = _short_bullet(contrib.text, 18)
            if item and len(bullet_points) < 2:
                bullet_points.append(f"- {item}")
    while len(bullet_points) < 2:
        bullet_points.append("- See the architecture figure for the overall pipeline.")

    formula_html = _format_formula_box(analysis, label="Main Formula", formula_index=1)
    content_parts = [trim_to_budget(intro, 24), "\n".join(bullet_points[:2])]
    if formula_html:
        content_parts.append(formula_html)
    return "\n\n".join(content_parts).strip()


def _build_key_idea_content(analysis: PaperAnalysis) -> str:
    bullets: list[str] = []
    for contrib in analysis.contributions[:3]:
        item = _short_bullet(contrib.text, 18)
        if item:
            bullets.append(f"- {item}")
    if analysis.key_figures:
        figure_hint = _clean_poster_text(analysis.key_figures[0].caption)
        if figure_hint and len(bullets) < 3:
            bullets.append(f"- {_short_bullet(figure_hint, 18)}")
    while len(bullets) < 2:
        bullets.append("- See the architecture figure for the overall pipeline.")
    return "\n".join(bullets[:3]).strip()


def _build_core_results_content(doc: PaperDocument, analysis: PaperAnalysis) -> str:
    intro = _build_core_intro(analysis)
    table = _build_core_result_table(doc, analysis)
    parts = [intro]
    if table:
        parts.append("[[CORE_TABLE]]")
        parts.append(table)
    return "\n\n".join(parts).strip()


def _build_core_intro(analysis: PaperAnalysis) -> str:
    exp = analysis.experiments
    if not exp:
        return "Core results are unavailable."
    summary = _first_sentence(_clean_poster_text(exp.main_results or ""))
    if not summary:
        summary = _first_sentence(_clean_poster_text(analysis.method_overview or ""))
    metrics = ", ".join(exp.metrics[:2]) if exp.metrics else ""
    if summary and metrics:
        return trim_to_budget(f"{summary} The main advantages are reflected in {metrics}.", 30)
    if summary:
        return trim_to_budget(summary, 24)
    if metrics:
        return f"The main advantages are reflected in {metrics}."
    return "Core results summarize the strongest gains across the paper's benchmarks."


def _build_core_result_table(doc: PaperDocument, analysis: PaperAnalysis) -> str:
    exp = analysis.experiments
    if not exp:
        return ""

    def _format_value(items: list[str], fallback: str) -> str:
        cleaned = [trim_to_budget(_clean_poster_text(item), 15) for item in items if _clean_poster_text(item)]
        return "<br>".join(cleaned) if cleaned else html_lib.escape(fallback)

    rows = [
        ("Datasets", _format_value(exp.datasets, "Not specified")),
        ("Metrics", _format_value(exp.metrics, "Not specified")),
        ("Main Results", html_lib.escape(
            trim_to_budget(_first_sentence(_clean_poster_text(exp.main_results or "")) or "Not specified", 20)
        )),
        ("Takeaways", _format_value(exp.takeaways[:3], "Not specified") if exp.takeaways else html_lib.escape("Not specified")),
    ]

    table_rows = "".join(f"<tr><th>{html_lib.escape(label)}</th><td>{value}</td></tr>" for label, value in rows)
    return (
        '<div class="item-details-wrap">'
        '<div class="item-details-title">Item Details</div>'
        '<table class="item-details-table">'
        f"<tbody>{table_rows}</tbody>"
        '</table>'
        '</div>'
    )


def _build_contributions_content(analysis: PaperAnalysis) -> str:
    bullets: list[str] = []
    for contrib in analysis.contributions[:4]:
        item = _short_bullet(contrib.text, BULLET_WORD_BUDGET)
        if item:
            bullets.append(f"- {item}")
    while len(bullets) < 4:
        fallback = [
            "- The method improves the core target task.",
            "- The design stays lightweight and easy to deploy.",
            "- Results stay consistent across settings.",
            "- The analysis highlights a clear practical benefit.",
        ][len(bullets)]
        bullets.append(fallback)
    return "\n".join(bullets[:4]).strip()


def _build_highlights_content(analysis: PaperAnalysis) -> str:
    items = [_short_bullet(x, BULLET_WORD_BUDGET) for x in _build_highlights(analysis)]
    items = [x for x in items if x]
    while len(items) < 4:
        defaults = [
            "A clean pipeline for the target task.",
            "Strong quantitative gains on the main benchmarks.",
            "A compact design with clear interpretability.",
            "Stable behavior under common variations.",
        ]
        items.append(defaults[len(items)])
    return "\n".join(f"- {item}" for item in items[:4])


def _build_project_content(analysis: PaperAnalysis) -> str:
    code_url = (analysis.code_url or "").strip()
    if code_url:
        link_html = f'<a class="code-link" href="{code_url}" target="_blank" rel="noreferrer noopener">{code_url}</a>'
    else:
        link_html = "Code will be release."
    return (
        "<div class=\"project-box\">"
        "<div class=\"qr-placeholder\">[QR]</div>"
        "<div class=\"code-cta\">"
        "<div class=\"label\">Code & Project</div>"
        f"<div>{link_html}</div>"
        "</div>"
        "</div>"
    )


def _extract_first_tex_table(doc: PaperDocument) -> str:
    sources: list[str] = []
    if doc.raw_markdown:
        sources.append(doc.raw_markdown)
    for sec in doc.sections:
        raw = getattr(sec, "raw_latex", "") or ""
        if raw:
            sources.append(raw)
    if not sources:
        return ""

    table_block = ""
    for source in sources:
        match = re.search(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", source, re.DOTALL)
        if match:
            table_block = match.group(1)
            break
        match = re.search(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", source, re.DOTALL)
        if match:
            table_block = match.group(1)
            break
    if not table_block:
        return ""

    table_block = re.sub(r"\\(?:hline|cline\{[^}]*\})", "", table_block)
    table_block = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", table_block)
    table_block = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", table_block)
    table_block = re.sub(r"\\textbf\{([^}]*)\}", r"\1", table_block)
    table_block = re.sub(r"\\emph\{([^}]*)\}", r"\1", table_block)
    table_block = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", "", table_block)

    rows: list[list[str]] = []
    for raw_row in re.split(r"\\\\", table_block):
        row = raw_row.strip()
        if not row:
            continue
        row = re.sub(r"\\[a-zA-Z]+", "", row)
        row = row.replace("\\", "")
        cells = [re.sub(r"\s+", " ", _clean_poster_text(cell)).strip() for cell in row.split("&")]
        cells = [cell for cell in cells if cell]
        if len(cells) >= 2:
            rows.append(cells)

    if len(rows) < 2:
        return ""

    header = rows[0]
    body = rows[1:]
    col_count = len(header)
    if col_count < 2:
        return ""
    header = header[:col_count]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * col_count) + " |"]
    for row in body[:3]:
        padded = row[:col_count] + [""] * max(0, col_count - len(row))
        lines.append("| " + " | ".join(padded[:col_count]) + " |")
    return "\n".join(lines)


def _build_motivation_callout(analysis: PaperAnalysis) -> str:
    insight = _first_sentence(_clean_poster_text(analysis.conclusion or ""))
    if not insight and analysis.experiments and analysis.experiments.main_results:
        insight = _first_sentence(_clean_poster_text(analysis.experiments.main_results))
    if not insight:
        insight = _first_clause(_clean_poster_text(analysis.problem_statement or ""))
    if not insight:
        insight = "The central insight is that the paper trades extra complexity for a cleaner, stronger result."
    return f'<div class="callout">Key insight: {insight}</div>'


def _format_formula_box(analysis: PaperAnalysis, label: str, formula_index: int = 0) -> str:
    formulas = getattr(analysis, "key_formulas", []) or []
    if formula_index >= len(formulas):
        return ""
    target = formulas[formula_index]
    cleaned_latex = _clean_formula_latex(target.latex)
    if not cleaned_latex:
        return ""
    semantic = _clean_poster_text(target.semantic_desc or "")
    semantic_html = f"<div class=\"formula-desc\">{semantic}</div>" if semantic else ""
    return (
        '<div class="formula-box">'
        f'<div class="formula-label">{label}</div>'
        f'<div>$$ {cleaned_latex} $$</div>'
        f'{semantic_html}'
        '</div>'
    )


def _result_hint(analysis: PaperAnalysis) -> str:
    exp = analysis.experiments
    if not exp:
        return ""
    parts: list[str] = []
    if exp.main_results:
        parts.append(_first_sentence(_clean_poster_text(exp.main_results)))
    if exp.datasets:
        parts.append(f"Benchmarks include {', '.join(exp.datasets[:3])}")
    if parts:
        return " ".join(parts).strip().rstrip(" ,;:.-") + "."
    return ""


def _infer_key_idea_title(analysis: PaperAnalysis) -> str:
    candidates = [
        _clean_poster_text(analysis.problem_statement or ""),
        _clean_poster_text(analysis.method_overview or ""),
    ]
    candidates.extend(_clean_poster_text(c.text) for c in analysis.contributions[:3])
    candidates.extend(_clean_poster_text(fig.caption) for fig in analysis.key_figures[:2])

    keyword_patterns = [
        r"detail injection",
        r"global context",
        r"shadow interaction",
        r"two-stage fine-tuning",
        r"latent diffusion",
        r"retinex",
        r"multi-scale channel attention",
        r"attention",
        r"architecture",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        lowered = candidate.lower()
        for pattern in keyword_patterns:
            match = re.search(pattern, lowered)
            if match:
                return candidate[match.start():match.end()].title()

    fallback = _clean_poster_text(analysis.contributions[0].text) if analysis.contributions else ""
    if fallback:
        words = fallback.split()
        return " ".join(words[:4]) if words else "Core Idea"
    return "Core Idea"


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return parts[0].strip() if parts else text.strip()


def _first_clause(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[;:\uFF1B\uFF1A]|\s+-\s+|\s+and\s+|\s+while\s+|\s+whereas\s+", text, maxsplit=1, flags=re.IGNORECASE)
    candidate = parts[0].strip() if parts else text.strip()
    return re.sub(r"\s+", " ", candidate)


def _apply_highlight_spans(text: str, highlights: list[tuple[str, str]]) -> str:
    if not text or not highlights:
        return text or ""

    highlighted = text
    for phrase, css_class in sorted(highlights, key=lambda item: len(item[0]), reverse=True):
        if not phrase:
            continue
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)

        def _replace(match: re.Match[str]) -> str:
            return f'<span class="{css_class}">{match.group(0)}</span>'

        highlighted = pattern.sub(_replace, highlighted, count=1)
    return highlighted









def _build_compact_layout(doc: PaperDocument, analysis: PaperAnalysis) -> list[PosterSection]:
    return [
        _build_motivation_section(analysis),
        _build_method_overview_section(analysis),
        _build_key_idea_section(analysis),
        _build_core_section(doc, analysis),
        _build_contributions_section(analysis),
        _build_highlights_section(analysis),
        _build_project_section(analysis),
    ]


def _build_row3(analysis: PaperAnalysis) -> list[PosterSection]:
    """Compatibility wrapper for the compact layout's last row."""
    return [
        _build_contributions_section(analysis),
        _build_highlights_section(analysis),
    ]


def _place_figures(doc: PaperDocument, analysis: PaperAnalysis, sections: list[PosterSection]) -> list[FigurePlacement]:
    """Distribute up to 4 figures: 1 hero (method overview), 2 results (core),
    1 illustrative (key idea). The core section gets 2 figures whenever
    possible so the rendered poster never leaves an empty figure column.
    """
    placements: list[FigurePlacement] = []
    core_slots = 0      # sec-main-method: 最多 2 张结果图（左右列）
    hero_taken = False  # sec-method-overview: 最多 1 张 hero 图
    idea_taken = False  # sec-key-idea: 最多 1 张示意/示例图

    def _add(fig, target: str, width_ratio: float) -> None:
        nonlocal core_slots, hero_taken, idea_taken
        placements.append(FigurePlacement(
            figure_id=getattr(fig, "figure_id", ""),
            section_id=target,
            width_ratio=width_ratio,
            caption=_figure_caption(fig),
        ))
        if target == "sec-main-method":
            core_slots += 1
        elif target == "sec-method-overview":
            hero_taken = True
        elif target == "sec-key-idea":
            idea_taken = True

    prioritized = sorted(
        _figure_candidates(doc, analysis),
        key=lambda f: _figure_priority(_figure_caption(f), _figure_role(f)),
    )
    seen_signatures: set[str] = set()
    candidates: list = []
    for fig in prioritized:
        signature = _figure_signature(_figure_caption(fig), _figure_role(fig), getattr(fig, "figure_id", ""))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidates.append(fig)
        if len(candidates) >= 8:
            break

    for fig in candidates:
        if len(placements) >= 4:
            break
        caption = _figure_caption(fig)
        role = _figure_role(fig)
        if _is_result_figure(caption, role):
            if core_slots < 2:
                _add(fig, "sec-main-method", 0.95 if core_slots == 0 else 0.85)
            elif not hero_taken:
                _add(fig, "sec-method-overview", 0.9)
            elif not idea_taken:
                _add(fig, "sec-key-idea", 0.8)
        elif _is_method_figure(caption, role):
            if not hero_taken:
                _add(fig, "sec-method-overview", 0.9)
            elif core_slots < 2:
                _add(fig, "sec-main-method", 0.95 if core_slots == 0 else 0.85)
            elif not idea_taken:
                _add(fig, "sec-key-idea", 0.8)
        else:
            if not idea_taken:
                _add(fig, "sec-key-idea", 0.8)
            elif core_slots < 2:
                _add(fig, "sec-main-method", 0.95 if core_slots == 0 else 0.85)
            elif not hero_taken:
                _add(fig, "sec-method-overview", 0.9)
    return placements


def _place_formulas(analysis: PaperAnalysis) -> list[FormulaDisplay]:
    displays = []
    for f in analysis.key_formulas:
        cleaned_latex = _clean_formula_latex(f.latex)
        if not cleaned_latex:
            continue
        displays.append(FormulaDisplay(
            formula_id=f.formula_id, section_id="sec-method-overview",
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
    """Pin the default poster layout to the compact three-column story grid."""
    for sec in sections:
        if sec.type == "title":
            sec.column = 1
            sec.col_span = 3
            sec.row = 0
            sec.row_span = 1
        elif sec.type == "motivation":
            sec.column = 1
            sec.col_span = 1
            sec.row = 1
            sec.row_span = 1
        elif sec.type == "method_overview":
            sec.column = 2
            sec.col_span = 1
            sec.row = 1
            sec.row_span = 1
        elif sec.type == "key_idea":
            sec.column = 3
            sec.col_span = 1
            sec.row = 1
            sec.row_span = 1
        elif sec.type == "main_method":
            sec.column = 1
            sec.col_span = 3
            sec.row = 2
            sec.row_span = 1
        elif sec.type == "experiments":
            sec.column = 1
            sec.col_span = 1
            sec.row = 2
            sec.row_span = 1
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
        elif sec.type == "project_link":
            sec.column = 3
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
        for c in analysis.contributions[:2]:
            text = _clean_poster_text(c.text)
            if text:
                lines.append(text)
    if analysis.experiments and analysis.experiments.main_results:
        lines.append(_first_sentence(_clean_poster_text(analysis.experiments.main_results)))
    if analysis.experiments and analysis.experiments.takeaways:
        lines.extend(_clean_poster_text(t) for t in analysis.experiments.takeaways[:2] if _clean_poster_text(t))
    if analysis.code_url:
        lines.append(f"Code: {analysis.code_url}")
    if not lines:
        if analysis.problem_statement:
            lines.append(_clean_poster_text(analysis.problem_statement))
        elif analysis.method_overview:
            lines.append(_clean_poster_text(analysis.method_overview[:180]))
        else:
            lines.append("See paper for details.")
    return lines[:4]


def _join_with_paragraphs(*parts: str) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(cleaned)





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



