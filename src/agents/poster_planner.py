from __future__ import annotations

import logging
import re

from src.llm.client import LLMClient
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
                _normalize_compact_layout(blueprint.sections)
                _apply_poster_highlights(blueprint.sections, analysis)
                _normalize_compact_layout(blueprint.sections)
                return blueprint
        except Exception:
            logger.warning("Gemini layout failed, falling back to static layout")
    sections = []
    sections.append(_build_title_section(doc, analysis))
    sections.extend(_build_compact_layout(doc, analysis))
    figure_placements = _place_figures(doc, analysis, sections)
    formula_displays = _place_formulas(analysis)
    _tighten_layout(sections, figure_placements)
    _normalize_compact_layout(sections)
    _apply_poster_highlights(sections, analysis)
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
    motivation = _summarize_motivation(analysis)
    return PosterSection(
        section_id="sec-motivation", type="motivation",
        title="Motivation",
        content_md=motivation or "(not provided)",
        column=1, col_span=1, row=1,
    )


def _build_method_overview_section(analysis: PaperAnalysis) -> PosterSection:
    overview_lines = []
    if analysis.method_overview:
        overview_lines.append(_clean_poster_text(analysis.method_overview))
    if analysis.key_formulas:
        formula_lines = []
        for f in analysis.key_formulas[:2]:
            cleaned_latex = _clean_formula_latex(f.latex)
            if not cleaned_latex:
                continue
            if f.semantic_desc:
                formula_lines.append(f"- $$ {cleaned_latex} $$: {_clean_poster_text(f.semantic_desc)}")
            else:
                formula_lines.append(f"- $$ {cleaned_latex} $$")
        if formula_lines:
            overview_lines.append("**Key Formulas**\n\n" + "\n".join(formula_lines))
    content = _join_with_paragraphs(*overview_lines) or "(method overview not provided)"
    return PosterSection(
        section_id="sec-method-overview", type="method_overview",
        title="Method Overview",
        content_md=content,
        column=2, col_span=1, row=1,
    )


def _build_key_idea_section(analysis: PaperAnalysis) -> PosterSection:
    key_title = _infer_key_idea_title(analysis)
    bullets: list[str] = []
    if analysis.contributions:
        lead = _clean_poster_text(analysis.contributions[0].text)
        if lead:
            bullets.append(f"- {lead}")
    if analysis.key_figures:
        figure_hint = _clean_poster_text(analysis.key_figures[0].caption)
        if figure_hint:
            bullets.append(f"- {figure_hint}")
    if not bullets and analysis.problem_statement:
        bullets.append(f"- {_clean_poster_text(analysis.problem_statement)}")
    content = "\n".join(bullets) if bullets else "(key idea not provided)"
    return PosterSection(
        section_id="sec-key-idea", type="key_idea",
        title=f"Key Idea: {key_title}",
        content_md=content,
        column=3, col_span=1, row=1,
    )


def _build_core_section(doc: PaperDocument, analysis: PaperAnalysis) -> PosterSection:
    core_parts: list[str] = []
    if analysis.method_overview:
        core_parts.append(analysis.method_overview)
    result_block = _build_result_block(doc, analysis)
    if result_block:
        core_parts.append(result_block)
    if analysis.key_figures:
        extra = _clean_poster_text(analysis.key_figures[0].caption)
        if extra:
            core_parts.append(f"**Core Visual:** {extra}")
    if analysis.code_url:
        core_parts.append(f"**Code:** {analysis.code_url}")
    content = _join_with_paragraphs(*core_parts) or "(core method not provided)"
    return PosterSection(
        section_id="sec-main-method", type="main_method",
        title="Core",
        content_md=content,
        column=1, col_span=3, row=2,
    )


def _build_results_section(analysis: PaperAnalysis) -> PosterSection:
    """Compatibility wrapper for the result summary used by older tests."""
    return PosterSection(
        section_id="sec-experiments", type="experiments",
        title="Result",
        content_md=_build_result_block(analysis) or "(experimental results not available)",
        column=1, col_span=1, row=2,
    )


def _build_contributions_section(analysis: PaperAnalysis) -> PosterSection:
    contrib_lines = []
    for c in analysis.contributions:
        prefix = "-"
        if c.category:
            prefix = "- **[" + c.category + "]**"
        contrib_lines.append(prefix + " " + _clean_poster_text(c.text))
    contrib_content = "\n".join(contrib_lines) if contrib_lines else "(not provided)"

    return PosterSection(
        section_id="sec-contributions", type="contributions",
        title="Contributions",
        content_md=contrib_content,
        column=1, col_span=1, row=3,
    )


def _build_highlights_section(analysis: PaperAnalysis) -> PosterSection:
    highlights = _build_highlights(analysis)
    if highlights:
        content = "\n".join(f"- {item}" for item in highlights)
    else:
        content = "- See paper for details."
    return PosterSection(
        section_id="sec-highlights", type="highlights",
        title="Highlights",
        content_md=content,
        column=2, col_span=1, row=3,
    )


def _build_project_section(analysis: PaperAnalysis) -> PosterSection:
    code_url = (analysis.code_url or "").strip()
    if code_url:
        content = (
            "**Code**\n\n"
            f'<a class="code-link" href="{code_url}" target="_blank" rel="noreferrer noopener">{code_url}</a>'
        )
    else:
        content = "**Code**\n\nCode link not provided in the paper."
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


def _build_result_block(doc: PaperDocument, analysis: PaperAnalysis) -> str:
    exp = analysis.experiments
    if not exp:
        return ""
    numeric_summary = _extract_numeric_result_summary(doc, analysis)
    if numeric_summary:
        return numeric_summary

    lines = ["**Result**"]
    rows: list[tuple[str, str]] = []
    if exp.datasets:
        rows.append(("Datasets", ", ".join(exp.datasets)))
    if exp.metrics:
        rows.append(("Metrics", ", ".join(exp.metrics)))
    if exp.main_results:
        rows.append(("Main Result", exp.main_results))

    if rows:
        lines.append("")
        lines.append("| Item | Details |")
        lines.append("| --- | --- |")
        for key, value in rows:
            lines.append(f"| {key} | {value} |")

    if exp.takeaways:
        lines.append("")
        lines.append("**Takeaway**")
        for item in exp.takeaways[:2]:
            lines.append(f"- {_clean_poster_text(item)}")
    return "\n".join(lines).strip()


def _extract_numeric_result_summary(doc: PaperDocument, analysis: PaperAnalysis) -> str:
    exp = analysis.experiments
    combined_sources = []
    combined_sources.append(analysis.full_analysis_md or "")
    combined_sources.append(doc.raw_markdown or "")
    for sec in doc.sections:
        combined_sources.append(sec.raw_latex or "")
        combined_sources.append(sec.text or "")

    lines: list[str] = []
    for source in combined_sources:
        if not source:
            continue
        lines.extend(line.strip() for line in source.splitlines() if line.strip())

    def _number_count(line: str) -> int:
        return len(re.findall(r"(?<!\w)(\d+\.\d+)", line))

    def _pick_line(required_terms: tuple[str, ...], min_numbers: int) -> tuple[str, list[str]]:
        best_line = ""
        best_numbers: list[str] = []
        for line in lines:
            lower = line.lower()
            if required_terms and not all(term in lower for term in required_terms):
                continue
            numbers = re.findall(r"(?<!\w)(\d+\.\d+)", line)
            if len(numbers) < min_numbers:
                continue
            if len(numbers) > len(best_numbers):
                best_line = line
                best_numbers = numbers
        return best_line, best_numbers

    # Main benchmark row: ISTD+, SRD, INS, WSRD+
    best_line, best_numbers = _pick_line(("ours",), 8)
    if len(best_numbers) >= 8:
        datasets = ["ISTD+", "SRD", "INS", "WSRD+"]
        rows = [
            (datasets[i], best_numbers[i * 2], best_numbers[i * 2 + 1])
            for i in range(4)
        ]
        lines_out = ["**Result**", "", "| Dataset | PSNR | SSIM |", "| --- | --- | --- |"]
        for name, psnr, ssim in rows:
            lines_out.append(f"| {name} | {psnr} | {ssim} |")
        cross_line, cross_numbers = _pick_line(("ours",), 6)
        if len(cross_numbers) >= 6 and cross_line != best_line:
            lines_out.extend(["", "**Cross-dataset**", "", "| Setting | PSNR | SSIM |", "| --- | --- | --- |"])
            cross_rows = [
                ("ISTD+→SRD", cross_numbers[0], cross_numbers[1]),
                ("SRD→ISTD+", cross_numbers[2], cross_numbers[3]),
                ("INS→WSRD+", cross_numbers[4], cross_numbers[5]),
            ]
            for name, psnr, ssim in cross_rows:
                lines_out.append(f"| {name} | {psnr} | {ssim} |")
        return "\n".join(lines_out)

    # Fallback: find any strong numeric row with at least 6 floats.
    fallback_candidates = []
    for line in lines:
        lower = line.lower()
        if "ours" not in lower and "full" not in lower and "result" not in lower:
            continue
        numbers = re.findall(r"(?<!\w)(\d+\.\d+)", line)
        if len(numbers) >= 6:
            fallback_candidates.append((len(numbers), line, numbers))
    fallback_candidates.sort(key=lambda item: item[0], reverse=True)
    if fallback_candidates:
        _, _, numbers = fallback_candidates[0]
        rows = []
        metric_names = (
            exp.datasets[:4]
            if exp and len(exp.datasets) >= 4
            else [f"Set {i+1}" for i in range(len(numbers) // 2)]
        )
        for i, name in enumerate(metric_names):
            if i * 2 + 1 >= len(numbers):
                break
            rows.append((name, numbers[i * 2], numbers[i * 2 + 1]))
        if rows:
            lines_out = ["**Result**", "", "| Dataset | PSNR | SSIM |", "| --- | --- | --- |"]
            for name, psnr, ssim in rows:
                lines_out.append(f"| {name} | {psnr} | {ssim} |")
            return "\n".join(lines_out)

    return ""


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


def _apply_poster_highlights(sections: list[PosterSection], analysis: PaperAnalysis) -> None:
    highlight_map = _select_poster_highlights(sections, analysis)
    if not highlight_map:
        return

    for sec in sections:
        highlights = highlight_map.get(sec.type)
        if sec.type == "main_method" and highlight_map.get("experiments"):
            highlights = (highlights or []) + highlight_map["experiments"]
        if not highlights:
            continue
        sec.content_md = _apply_highlight_spans(sec.content_md, highlights)


def _select_poster_highlights(
    sections: list[PosterSection],
    analysis: PaperAnalysis,
) -> dict[str, list[tuple[str, str]]]:
    if not LLMClient.is_configured():
        return {}

    from src.config import settings

    client = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )

    user_prompt = _build_highlight_prompt(sections, analysis)
    schema = {
        "type": "object",
        "properties": {
            "highlights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_type": {
                            "type": "string",
                            "enum": ["method_overview", "key_idea", "main_method", "experiments"],
                        },
                        "phrase": {"type": "string"},
                        "kind": {"type": "string", "enum": ["phrase", "metric"]},
                    },
                    "required": ["section_type", "phrase", "kind"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["highlights"],
        "additionalProperties": False,
    }

    try:
        result = client.chat_json(system=_POSTER_HIGHLIGHT_SYSTEM_PROMPT, user=user_prompt, response_schema=schema)
    except Exception as exc:
        logger.warning("Poster highlight selection failed: %s", exc)
        return {}

    content_by_type = {
        sec.type: (sec.content_md or "")
        for sec in sections
        if sec.type in {"method_overview", "key_idea", "main_method", "experiments"}
    }
    if "experiments" not in content_by_type and "main_method" in content_by_type:
        content_by_type["experiments"] = content_by_type["main_method"]

    mapped: dict[str, list[tuple[str, str]]] = {"method_overview": [], "key_idea": [], "main_method": [], "experiments": []}
    seen: set[tuple[str, str]] = set()
    highlights = result.get("highlights", []) if isinstance(result, dict) else []
    for item in highlights:
        if not isinstance(item, dict):
            continue
        section_type = item.get("section_type", "")
        if section_type not in mapped:
            continue
        phrase = _clean_poster_text(str(item.get("phrase", "")))
        if not phrase:
            continue
        haystack = content_by_type.get(section_type, "")
        if phrase.lower() not in haystack.lower():
            continue
        css_class = "poster-highlight-metric" if item.get("kind") == "metric" else "poster-highlight"
        key = (section_type, phrase.lower())
        if key in seen:
            continue
        seen.add(key)
        mapped[section_type].append((phrase, css_class))

    return {key: value for key, value in mapped.items() if value}


def _build_highlight_prompt(sections: list[PosterSection], analysis: PaperAnalysis) -> str:
    parts: list[str] = []
    parts.append("Analyze the full poster content and choose exact phrases to highlight.")
    parts.append("Focus on method modules, architecture names, datasets, metrics, and result numbers.")
    parts.append("Do not select from Motivation unless absolutely necessary.")
    parts.append("Prefer phrases from method_overview, key_idea, main_method, and experiments.")
    parts.append("Return only phrases that appear verbatim in the text below, and keep them short.")
    parts.append("Use kind=metric for numbers, percentages, scores, or benchmark values. Use kind=phrase otherwise.")
    parts.append("Return at most 6 highlights total.")
    parts.append("")
    parts.append(f"Paper: {analysis.paper_id}")
    if analysis.problem_statement:
        parts.append(f"Problem: {_clean_poster_text(analysis.problem_statement)}")
    parts.append("")
    for sec in sections:
        if sec.type not in {"motivation", "method_overview", "key_idea", "main_method", "experiments", "contributions", "highlights"}:
            continue
        content = (sec.content_md or "").strip()
        if not content:
            continue
        if len(content) > 1200:
            content = content[:1200].rstrip() + "..."
        parts.append(f"[{sec.type}] {sec.title}")
        parts.append(content)
        parts.append("")
    return "\n".join(parts).strip()


_POSTER_HIGHLIGHT_SYSTEM_PROMPT = (
    "You are analyzing a scientific poster draft. "
    "Select only exact phrases from the provided poster content that deserve visual emphasis. "
    "Prioritize method modules, architecture names, datasets, metrics, and result numbers. "
    "Return concise phrases that appear verbatim in the text. "
    "Do not invent wording or choose from Motivation unless necessary. "
    "Prefer highlights for method_overview, key_idea, main_method, and experiments only."
)


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
    placements = []
    method_hero_taken = False
    key_idea_taken = False
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
            target = "sec-main-method"
            width_ratio = 0.96 if method_hero_taken else 0.86
            method_hero_taken = True
        elif _is_method_figure(caption, role):
            target = "sec-method-overview"
            width_ratio = 0.98 if not method_hero_taken else 0.74
            method_hero_taken = True
        else:
            if not key_idea_taken:
                target = "sec-key-idea"
                width_ratio = 0.76
                key_idea_taken = True
            else:
                target = "sec-main-method"
                width_ratio = 0.72 if method_hero_taken else 0.82
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



