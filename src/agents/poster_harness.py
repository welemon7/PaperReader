from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.config import settings
from src.llm.client import LLMClient, LLMError
from src.llm.multimodal_client import (
    capture_poster_full_and_sections,
    downscale_image,
    multimodal_analyze_labeled,
)
from src.renderers.html_renderer import HtmlPosterRenderer
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import Figure, PaperDocument
from src.schemas.poster import PosterBlueprint, PosterSection
from src.schemas.poster_harness import HarnessConfig, HarnessResult, HarnessRound
from src.schemas.poster_v2 import EvaluationQuestion, PosterComment, PosterQAEval, PosterReview
from src.agents.content_policy import count_words, section_budget, trim_to_budget
from src.utils.figure_assets import save_svg_asset, sanitize_asset_name

logger = logging.getLogger(__name__)

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL is a hard dependency in practice
    Image = None  # type: ignore[assignment]


@dataclass
class SectionBlankReport:
    section_id: str
    blank_ratio: float
    content_ratio: float
    width: int
    height: int


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM_PROMPT = """You are a strict scientific poster reviewer (VLM judge).
You are shown the rendered poster image (and possibly zoomed-in section crops).

Evaluate the poster on these visual-quality dimensions:
- layout: balance, alignment, column structure, whitespace
- typography: font hierarchy, readability, text overflow, density
- figures: figure size, cropping, relevance, caption readability
- color: palette harmony, contrast, section differentiation
- coverage: whether the core paper content (problem, method, results) is visibly present
- overflow: text clipping, overlapping elements, cut-off content
- text_density: whether the amount of text is appropriate for a poster
  (a good poster keeps the whole body around 250-500 words; flag any over-dense
  section that reads like a paragraph dump)

Return JSON ONLY, with this exact structure:
{
  "quality_score": <int 0-10 overall>,
  "dimension_scores": {"layout": <0-10>, "typography": <0-10>, "figures": <0-10>, "color": <0-10>, "coverage": <0-10>, "overflow": <0-10>, "text_density": <0-10>},
  "needs_improvement": <true|false>,
  "summary": "<one or two sentence summary>",
  "issues": [
    {
      "description": "<what is wrong>",
      "severity": "error|warning|info",
      "target": "<section id, section type, or section title, e.g. sec-motivation | Motivation | Core Results>",
      "suggestion": "<concrete fix suggestion>",
      "action": "rewrite|condense|resize|reflow|replace_figure|remove|keep"
    }
  ]
}

Rules:
- If the poster already looks good, set needs_improvement=false and return an empty issues list.
- Do not invent issues. Only report what is visible in the images.
- Every issue MUST carry an action from the allowed set.
- If a region is visibly empty/blank (e.g. a figure column with nothing in it),
  report it with action "replace_figure" or "reflow".
- Use action "condense" (not "rewrite") when the section is over-dense and just
  needs to be shortened; use "rewrite" only when the content itself is wrong.
"""

_REWRITE_SYSTEM_PROMPT = (
    "You are an expert scientific poster editor. Rewrite the given section content to fix the "
    "reported visual/structural issue. Keep ALL factual claims, numbers, formulas, citations and "
    "technical meaning intact. Prefer concise bullet points. Output ONLY the rewritten markdown "
    "content — no explanations, no code fences, no JSON."
)

_QA_SYSTEM_PROMPT = (
    "You are a scientific poster evaluator. Answer the question using ONLY the poster content. "
    "Return JSON with answer, short_reason, and confidence."
)

_VISUAL_QA_SYSTEM_PROMPT = """You are a strict scientific-poster comprehension evaluator.
Answer each question using ONLY what is legible in the supplied poster image. Do not use
outside knowledge and do not guess. Return JSON ONLY in this exact format:
{"answers": [{"question_id": "...", "answer": "...", "confidence": 0.0}]}
If the answer is absent or unreadable, return an empty answer with confidence 0.
"""

# ---------------------------------------------------------------------------
# Normalization helpers (review JSON -> PosterReview)
# ---------------------------------------------------------------------------


def _normalize_severity(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"error", "warning", "info"}:
        return text
    if text in {"high", "critical", "severe", "major"}:
        return "error"
    if text in {"medium", "moderate", "normal"}:
        return "warning"
    if text in {"low", "minor", "suggestion", "note"}:
        return "info"
    return "warning"


def _normalize_quality_score(value: object) -> int:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0
    if score <= 0:
        return 0
    if score <= 10:
        return int(round(score))
    if score <= 100:
        return int(round(score / 10.0))
    return 10


def _normalize_dimension_scores(value: object) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(value, dict):
        return result
    for key, val in value.items():
        try:
            result[str(key)] = float(val)
        except (TypeError, ValueError):
            result[str(key)] = 0.0
    return result


def _normalize_issue(item: object) -> Optional[PosterComment]:
    if not isinstance(item, dict):
        return None
    action = str(item.get("action") or "").strip().lower() or "rewrite"
    allowed = {"resize", "reflow", "rewrite", "condense", "replace_figure", "remove", "supplement", "keep"}
    if action not in allowed:
        action = "rewrite"
    description = str(item.get("description") or item.get("issue") or "").strip()
    if not description:
        return None
    return PosterComment(
        issue=description,
        severity=_normalize_severity(item.get("severity")),
        target=str(item.get("target") or "").strip(),
        suggestion=str(item.get("suggestion") or "").strip(),
        action=action,
    )


# ---------------------------------------------------------------------------
# Visual review
# ---------------------------------------------------------------------------


def _section_selectors(blueprint: PosterBlueprint) -> dict[str, str]:
    selectors: dict[str, str] = {}
    for sec in blueprint.sections:
        if sec.type == "title":
            continue
        selectors[sec.section_id] = f"#{sec.section_id}"
    return selectors


def _dense_section_ids(blueprint: PosterBlueprint, limit: int) -> list[str]:
    """Pick the most text-dense sections for zoomed-in crops."""
    candidates = [
        sec for sec in blueprint.sections
        if sec.type not in {"title", "project_link"} and (sec.content_md or "").strip()
    ]
    candidates.sort(key=lambda s: len(s.content_md or ""), reverse=True)
    return [sec.section_id for sec in candidates[:limit]]


def _measure_png_blank_ratio(png_path: Path) -> Optional[float]:
    if Image is None or not png_path.exists():
        return None
    try:
        with Image.open(png_path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            if width <= 0 or height <= 0:
                return None
            pixels = rgb.load()
            total = width * height
            blank = 0
            for y in range(height):
                for x in range(width):
                    r, g, b = pixels[x, y]
                    if r >= 244 and g >= 244 and b >= 244:
                        blank += 1
            return blank / total if total else None
    except Exception as exc:
        logger.warning("Blank ratio probe failed for %s: %s", png_path, exc)
        return None


def _section_blank_reports(round_dir: Path) -> list[SectionBlankReport]:
    sections_dir = round_dir / "sections"
    if not sections_dir.exists():
        return []
    reports: list[SectionBlankReport] = []
    for png_path in sorted(sections_dir.glob("*.png")):
        ratio = _measure_png_blank_ratio(png_path)
        if ratio is None:
            continue
        section_id = png_path.stem
        reports.append(SectionBlankReport(
            section_id=section_id,
            blank_ratio=ratio,
            content_ratio=max(0.0, 1.0 - ratio),
            width=0,
            height=0,
        ))
    return reports


def _visual_supplement_html(sec: PosterSection, report: SectionBlankReport) -> str:
    title = html.escape(sec.title or sec.type.replace("_", " ").title())
    if sec.type == "motivation":
        return (
            '<div class="mini-visual-grid">'
            f'<div class="mini-node"><div class="mini-node-title">Problem</div><div class="mini-node-copy">{title}: gap, pain point, or failure case.</div></div>'
            f'<div class="mini-node"><div class="mini-node-title">Why now</div><div class="mini-node-copy">Show the cost of leaving it unresolved.</div></div>'
            '</div>'
            '<div class="mini-pill-row">'
            '<span class="mini-pill">gap</span><span class="mini-pill">impact</span><span class="mini-pill">goal</span>'
            '</div>'
        )
    if sec.type in {"method_overview", "key_idea"}:
        return (
            '<div class="mini-visual-grid">'
            '<div class="mini-node"><div class="mini-node-title">Input</div><div class="mini-node-copy">data, constraints, or observations.</div></div>'
            '<div class="mini-node"><div class="mini-node-title">Flow</div><div class="mini-node-copy">arrow to the main transformation.</div></div>'
            '<div class="mini-node"><div class="mini-node-title">Output</div><div class="mini-node-copy">what becomes simpler, cleaner, or more accurate.</div></div>'
            '<div class="mini-node"><div class="mini-node-title">Signal</div><div class="mini-node-copy">one symbol, one equation, or one key step.</div></div>'
            '</div>'
        )
    if sec.type in {"contributions", "highlights"}:
        return (
            '<div class="mini-pill-row">'
            '<span class="mini-pill">new</span><span class="mini-pill">fast</span><span class="mini-pill">robust</span><span class="mini-pill">better</span>'
            '</div>'
            '<div class="mini-visual-grid">'
            '<div class="mini-node"><div class="mini-node-title">1</div><div class="mini-node-copy">Main novelty or design choice.</div></div>'
            '<div class="mini-node"><div class="mini-node-title">2</div><div class="mini-node-copy">What changed in practice.</div></div>'
            '</div>'
        )
    if sec.type == "project_link":
        return (
            '<div class="mini-visual-grid">'
            '<div class="mini-node"><div class="mini-node-title">Code</div><div class="mini-node-copy">repo, demo, or release note.</div></div>'
            '<div class="mini-node"><div class="mini-node-title">Use</div><div class="mini-node-copy">how a reader can continue from here.</div></div>'
            '</div>'
        )
    return (
        '<div class="mini-visual-grid">'
        f'<div class="mini-node"><div class="mini-node-title">{title}</div><div class="mini-node-copy">Blank ratio {report.blank_ratio:.0%}. Add a compact visual cue.</div></div>'
        '<div class="mini-node"><div class="mini-node-title">Cue</div><div class="mini-node-copy">Use one flow, one arrow, or one small comparison.</div></div>'
        '</div>'
    )


def _blank_section_visual_prompt(sec: PosterSection, report: SectionBlankReport) -> str:
    body = (sec.content_md or "").strip()
    body = re.sub(r"\s+", " ", body)
    if len(body) > 700:
        body = body[:700].rstrip() + "…"
    return (
        f"Section type: {sec.type}\n"
        f"Section title: {sec.title or sec.type}\n"
        f"Blank ratio: {report.blank_ratio:.0%}\n"
        f"Content preview: {body or '(empty)'}\n\n"
        "Generate a compact SVG infographic or symbol-only visual that helps fill the blank space. "
        "Use a clean white background, 1-2 accent colors, and no external dependencies. "
        "Return ONLY a valid standalone SVG document. If you cannot generate a useful diagram, "
        "return a single meaningful symbol or icon-like glyph wrapped in SVG."
    )


def _generate_blank_supplement_asset(
    llm: Optional[LLMClient],
    sec: PosterSection,
    report: SectionBlankReport,
    asset_dirs: list[Path],
) -> Optional[str]:
    if not asset_dirs:
        return None

    target_name = sanitize_asset_name(f"{sec.section_id}-supplement", sec.section_id)
    primary_dir = asset_dirs[0]
    svg_text: str | None = None

    if llm is not None:
        try:
            svg_text = _strip_fences(llm.chat(
                system=(
                    "You create minimal standalone SVG illustrations for scientific posters. "
                    "Return valid SVG only. No markdown, no code fences, no explanations."
                ),
                user=_blank_section_visual_prompt(sec, report),
            ))
        except Exception as exc:
            logger.warning("Blank-section SVG generation failed for %s: %s", sec.section_id, exc)
            svg_text = None

    if not svg_text:
        return None

    try:
        primary_svg = save_svg_asset(svg_text, primary_dir, target_name)
    except Exception as exc:
        logger.warning("Failed to persist blank-section SVG for %s: %s", sec.section_id, exc)
        return None

    for extra_dir in asset_dirs[1:]:
        try:
            extra_dir.mkdir(parents=True, exist_ok=True)
            extra_path = extra_dir / primary_svg.name
            extra_path.write_text(primary_svg.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to stage SVG asset %s into %s: %s", primary_svg.name, extra_dir, exc)

    return f"figures/{primary_svg.name}"


def _heuristic_density_issues(blueprint: PosterBlueprint) -> list[PosterComment]:
    """Deterministic density/blank check (cheap, runs before/with the VLM).

    Mirrors the reference's overflow/blank detection: sections that exceed their
    word budget get a "condense" issue; near-empty sections get a "reflow" hint.
    """
    issues: list[PosterComment] = []
    for sec in blueprint.sections:
        if sec.type == "title":
            continue
        words = count_words(sec.content_md)
        budget = section_budget(sec.type)
        if sec.type == "main_method":
            # 结果表计入词数，属于合理内容；仅当明显超限才提示
            if words > budget + 25:
                issues.append(PosterComment(
                    issue=f"Core section text is dense ({words} words)",
                    severity="warning",
                    target=sec.section_id,
                    action="condense",
                    suggestion="Trim the narrative around the results table; keep the table.",
                ))
            continue
        if words > budget + 10:
            issues.append(PosterComment(
                issue=f"Text too dense ({words} words; poster budget {budget})",
                severity="warning",
                target=sec.section_id,
                action="condense",
                suggestion=f"Shorten this section to at most {budget} words, keep all facts and numbers.",
            ))
        elif (sec.content_md or "").strip() and words < 12 and "not provided" not in sec.content_md:
            issues.append(PosterComment(
                issue=f"Section has very little content ({words} words)",
                severity="info",
                target=sec.section_id,
                action="reflow",
                suggestion="Balance this section with the rest of the poster.",
            ))
    return issues


def inspect_rendered_poster(
        html_path: Path,
        blueprint: PosterBlueprint,
) -> dict[str, Any]:
    """Run deterministic browser checks before trusting a VLM score.

    A VLM is useful for composition and visual storytelling but not dependable
    enough to waive broken assets, clipping, or a canvas with the wrong aspect
    ratio.  This audit intentionally returns evidence rather than raising, so
    the report remains useful even when a local browser is unavailable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"available": False, "reason": "playwright_not_installed", "hard_failures": []}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": blueprint.width_px, "height": blueprint.height_px},
                device_scale_factor=1,
            )
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.wait_for_timeout(1500)
            page.evaluate("document.fonts && document.fonts.ready")
            data = page.evaluate(
                """() => {
                    const poster = document.querySelector('.poster-container');
                    const sections = [...document.querySelectorAll('.section-block')];
                    const px = value => Number.parseFloat(value || '0') || 0;
                    const clipped = sections.filter(section => {
                        const content = section.querySelector('.section-content');
                        if (!content) return false;
                        const style = getComputedStyle(section);
                        return style.overflow !== 'visible' && (
                            content.scrollHeight > content.clientHeight + 2 ||
                            content.scrollWidth > content.clientWidth + 2
                        );
                    }).map(section => section.id);
                    const empty = sections.filter(section => {
                        const content = section.querySelector('.section-content');
                        if (!content) return false;
                        const text = (content.innerText || '').replace(/\\s+/g, ' ').trim();
                        const hasImage = !!content.querySelector('img');
                        return !hasImage && (!text || text === '(not provided)');
                    }).map(section => section.id);
                    const brokenImages = [...document.images]
                        .filter(image => !image.complete || image.naturalWidth === 0)
                        .map(image => image.alt || image.src || 'unnamed-image');
                    const fontSizes = sections.map(section => {
                        const content = section.querySelector('.section-content');
                        return content ? px(getComputedStyle(content).fontSize) : 0;
                    }).filter(Boolean);
                    const rect = poster ? poster.getBoundingClientRect() : null;
                    return {
                        section_count: sections.length,
                        clipped_sections: clipped,
                        empty_sections: empty,
                        broken_images: brokenImages,
                        min_body_font_px: fontSizes.length ? Math.min(...fontSizes) : 0,
                        canvas_width: rect ? Math.round(rect.width) : 0,
                        canvas_height: rect ? Math.round(rect.height) : 0,
                    };
                }"""
            )
            browser.close()
    except Exception as exc:
        logger.warning("Deterministic browser audit unavailable: %s", exc)
        return {"available": False, "reason": f"browser_error: {exc}", "hard_failures": []}

    failures: list[str] = []
    if data["broken_images"]:
        failures.append("broken_images")
    if data["clipped_sections"]:
        failures.append("text_or_layout_clipping")
    if data["empty_sections"]:
        failures.append("empty_required_sections")
    expected_ratio = blueprint.width_px / max(blueprint.height_px, 1)
    actual_ratio = data["canvas_width"] / max(data["canvas_height"], 1)
    if data["canvas_width"] <= 0 or abs(actual_ratio - expected_ratio) > 0.08:
        failures.append("canvas_aspect_ratio_mismatch")
    if data["min_body_font_px"] and data["min_body_font_px"] < 12:
        failures.append("body_text_too_small")
    return {"available": True, "hard_failures": failures, **data}


def _merge_deterministic_issues(review: PosterReview, audit: dict[str, Any]) -> None:
    """Expose browser failures as actionable comments and hard pass blockers."""
    review.deterministic_checks = audit
    review.hard_failures = list(audit.get("hard_failures") or [])
    existing = {comment.target for comment in review.issues}
    for section_id in audit.get("clipped_sections") or []:
        if section_id not in existing:
            review.issues.append(PosterComment(
                issue="Detected clipped or overflowing content in this section",
                severity="error", target=section_id, action="condense",
                suggestion="Reduce body copy or reflow this panel before the next render.",
            ))
    for section_id in audit.get("empty_sections") or []:
        if section_id not in existing:
            review.issues.append(PosterComment(
                issue="This required section is visibly empty",
                severity="error", target=section_id, action="reflow",
                suggestion="Fill it with relevant paper evidence or reallocate the panel.",
            ))
    if audit.get("broken_images"):
        review.issues.append(PosterComment(
            issue="One or more figure assets did not render",
            severity="error", target="", action="replace_figure",
            suggestion="Use a valid local raster asset and preserve its source caption.",
        ))


def _merge_blank_reports(review: PosterReview, blank_reports: list[SectionBlankReport], blueprint: PosterBlueprint) -> None:
    if not blank_reports:
        return
    existing = {comment.target for comment in review.issues}
    for report in blank_reports:
        sec = next((s for s in blueprint.sections if s.section_id == report.section_id), None)
        if sec is None:
            continue
        if report.blank_ratio < 0.34:
            continue
        if report.section_id in existing:
            continue
        review.issues.append(PosterComment(
            issue=f"Section retains {report.blank_ratio:.0%} blank area",
            severity="info" if report.blank_ratio < 0.6 else "warning",
            target=report.section_id,
            suggestion="Add a compact visual explanation to use the spare space.",
            action="supplement",
        ))
        existing.add(report.section_id)


def review_rendered_poster(
        html_path: Path,
        round_dir: Path,
        config: HarnessConfig,
        blueprint: PosterBlueprint,
        model: Optional[str] = None,
) -> Optional[PosterReview]:
    """Capture the rendered poster (full + zoom crops) and ask the VLM for a review.

    Returns None when vision capture or VLM analysis is unavailable (caller should
    fall back to the legacy single-shot HTML optimizer).
    """
    png_path = round_dir / "poster.png"
    selectors = _section_selectors(blueprint)
    crops = capture_poster_full_and_sections(
        html_path,
        png_path,
        selectors,
        width=blueprint.width_px,
        height=blueprint.height_px,
    )
    if not png_path.exists():
        logger.warning("Poster PNG capture failed; vision review unavailable")
        return None

    full_small = downscale_image(png_path, max_width=1400) or png_path

    # multimodal_analyze_labeled takes (image_path, label), not the reverse.
    images: list[tuple[str, str]] = [(str(full_small), "poster (full view)")]
    if config.zoom_crops:
        for sec_id in _dense_section_ids(blueprint, config.max_crops):
            crop = crops.get(sec_id)
            if crop and crop.exists():
                sec = next((s for s in blueprint.sections if s.section_id == sec_id), None)
                label = f"section: {sec.title if sec else sec_id}"
                images.append((str(crop), label))

    user_text = (
        "Review the rendered scientific poster for visual quality. Report concrete issues "
        "with exact targets (section id/title) and actionable fixes."
    )
    raw = multimodal_analyze_labeled(
        _REVIEW_SYSTEM_PROMPT,
        images,
        user_text=user_text,
        model=model,
    )
    if not raw:
        logger.warning("VLM review returned no result; vision review unavailable")
        return None

    issues: list[PosterComment] = []
    for item in raw.get("issues") or []:
        issue = _normalize_issue(item)
        if issue:
            issues.append(issue)

    review = PosterReview(
        quality_score=_normalize_quality_score(raw.get("quality_score")),
        needs_improvement=bool(raw.get("needs_improvement", True)),
        issues=issues,
        summary=str(raw.get("summary") or ""),
        layout_feedback=[str(x) for x in (raw.get("layout_feedback") or []) if x],
        dimension_scores=_normalize_dimension_scores(raw.get("dimension_scores")),
    )
    blank_reports = _section_blank_reports(round_dir)
    if blank_reports:
        review.deterministic_checks = dict(review.deterministic_checks or {})
        review.deterministic_checks["section_blank_reports"] = [r.__dict__ for r in blank_reports]
        _merge_blank_reports(review, blank_reports, blueprint)
    _merge_deterministic_issues(review, inspect_rendered_poster(html_path, blueprint))
    return review


# ---------------------------------------------------------------------------
# Feedback application
# ---------------------------------------------------------------------------


def _match_section(target: str, blueprint: PosterBlueprint) -> Optional[PosterSection]:
    if not target:
        return None
    t = str(target).strip().lower()
    for sec in blueprint.sections:
        if sec.section_id.lower() == t:
            return sec
        if (sec.type or "").lower() == t:
            return sec
        if (sec.title or "").lower() == t:
            return sec
        if sec.section_id.lower() in t or t in sec.section_id.lower():
            return sec
    return None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


def _trim_dense_text(text: str, max_words: int = 90) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def _css_for_section(section_id: str, kind: str) -> str:
    sel = f"#{section_id}"
    if kind == "density":
        return (
            f"{sel} .section-content {{ font-size: 12px !important; line-height: 1.35 !important; }}\n"
            f"{sel} .section-content p, {sel} .section-content li {{ margin-bottom: 4px !important; }}\n"
            f"{sel} .section-title {{ padding: 6px 8px !important; }}"
        )
    if kind == "figure_small":
        return (
            f"{sel} .figure-card img {{ max-height: 320px !important; object-fit: contain !important; }}\n"
            f"{sel} .figure-caption {{ font-size: 11px !important; }}"
        )
    if kind == "spacing":
        return f"{sel} .section-content {{ padding: 10px 12px !important; }}"
    return ""


def _css_global(kind: str) -> str:
    if kind == "density":
        return (
            ".section-content { font-size: 12px !important; line-height: 1.4 !important; }\n"
            ".grid-container { gap: 12px !important; }"
        )
    if kind == "figures_larger":
        return ".figure-card img { max-height: 300px !important; }"
    if kind == "spacing":
        return ".section-content { padding: 10px 12px !important; }"
    return ""


def _apply_css_patches(html: str, patches: list[str]) -> str:
    if not patches:
        return html
    style = "\n".join(patches)
    block = f'\n<style id="harness-css-patch">\n{style}\n</style>\n'
    if "</head>" in html:
        return html.replace("</head>", block + "</head>", 1)
    return block + html


def _rewrite_section(
        llm: Optional[LLMClient],
        sec: PosterSection,
        issue: str,
        suggestion: str,
        max_words: Optional[int] = None,
) -> bool:
    """Rewrite one section's content via the LLM (targeted, no full-document regeneration).

    When max_words is given the prompt carries an explicit length target, mirroring
    the reference's "shrink by N characters" feedback loop.
    """
    if not llm:
        return False
    length_line = f"\nLength limit: output at most {max_words} words." if max_words else ""
    user = (
        f"Section title: {sec.title}\n\nCurrent content:\n{sec.content_md}\n\n"
        f"Reported issue: {issue}\n"
        f"Suggested fix: {suggestion or '(none given; make it concise and scannable)'}\n"
        f"{length_line}\n\n"
        "Return only the rewritten markdown content."
    )
    try:
        new_content = _strip_fences(llm.chat(system=_REWRITE_SYSTEM_PROMPT, user=user))
        if new_content and len(new_content) > 20:
            if max_words and sec.type != "main_method":
                new_content = trim_to_budget(new_content, max_words)
            sec.content_md = new_content
            return True
    except LLMError as e:
        logger.warning("Section rewrite failed: %s", e)
    except Exception as e:
        logger.warning("Section rewrite error: %s", e)
    return False


def _apply_feedback(
        blueprint: PosterBlueprint,
        review: PosterReview,
        llm: Optional[LLMClient],
        css_patches: list[str],
        asset_dirs: Optional[list[Path]] = None,
) -> list[str]:
    """Translate a review into blueprint mutations + CSS patches.

    Returns a list of human-readable applied-action descriptions.
    """
    applied: list[str] = []
    for comment in review.issues:
        sec = _match_section(comment.target, blueprint)
        action = comment.action
        issue_lower = comment.issue.lower()

        if action == "keep":
            continue

        if action == "rewrite":
            if sec:
                if _rewrite_section(llm, sec, comment.issue, comment.suggestion):
                    applied.append(f"rewrite {sec.section_id} (LLM)")
                else:
                    old_len = len(sec.content_md or "")
                    sec.content_md = _trim_dense_text(sec.content_md or "")
                    if len(sec.content_md or "") < old_len:
                        applied.append(f"rewrite {sec.section_id} (trim)")
                    else:
                        applied.append(f"rewrite {sec.section_id} (no-op)")
            else:
                applied.append(f"rewrite (no target: {comment.issue[:60]})")

        elif action == "condense":
            # 借鉴参考项目的长度控制：LLM 带字数目标改写，失败则确定性截断
            if sec:
                budget = section_budget(sec.type)
                if _rewrite_section(llm, sec, comment.issue, comment.suggestion, max_words=budget):
                    applied.append(f"condense {sec.section_id} (LLM, <= {budget} words)")
                else:
                    old_words = count_words(sec.content_md)
                    if sec.type == "main_method":
                        # 主方法区含结果表，避免截断破坏表格；仅当无表时才截断
                        if "[[CORE_TABLE]]" not in (sec.content_md or ""):
                            sec.content_md = trim_to_budget(sec.content_md or "", budget)
                            applied.append(f"condense {sec.section_id} (trim)")
                        else:
                            applied.append(f"condense {sec.section_id} (no-op, table kept)")
                    else:
                        sec.content_md = trim_to_budget(sec.content_md or "", budget)
                        if count_words(sec.content_md) < old_words:
                            applied.append(f"condense {sec.section_id} (trim to {budget})")
                        else:
                            applied.append(f"condense {sec.section_id} (no-op)")
            else:
                applied.append(f"condense (no target: {comment.issue[:60]})")

        elif action in {"resize", "reflow"}:
            if "figure" in issue_lower or "image" in issue_lower:
                patch = _css_for_section(sec.section_id, "figure_small") if sec else _css_global("figures_larger")
                if patch and patch not in css_patches:
                    css_patches.append(patch)
                applied.append(f"{action} figures ({sec.section_id if sec else 'global'})")
            elif "dense" in issue_lower or "overflow" in issue_lower or "text" in issue_lower:
                patch = _css_for_section(sec.section_id, "density") if sec else _css_global("density")
                if patch and patch not in css_patches:
                    css_patches.append(patch)
                applied.append(f"{action} density ({sec.section_id if sec else 'global'})")
            else:
                patch = _css_for_section(sec.section_id, "spacing") if sec else _css_global("spacing")
                if patch and patch not in css_patches:
                    css_patches.append(patch)
                applied.append(f"{action} spacing ({sec.section_id if sec else 'global'})")

        elif action == "replace_figure":
            # 按各区块理想图数决策：core 理想 2 张、其他 1 张；已满则放大，未满才移图，
            # 保证 core 区不会因移图出现空列。
            if sec:
                ideal = 2 if sec.section_id == "sec-main-method" else 1
                current = sum(1 for fp in blueprint.figure_placements if fp.section_id == sec.section_id)
                if current >= ideal:
                    patch = _css_for_section(sec.section_id, "figure_small")
                    if patch and patch not in css_patches:
                        css_patches.append(patch)
                    applied.append(f"replace_figure enlarge ({sec.section_id})")
                else:
                    moved = False
                    for fp in blueprint.figure_placements:
                        if fp.section_id != sec.section_id:
                            old = fp.section_id
                            fp.section_id = sec.section_id
                            applied.append(f"replace_figure {fp.figure_id} {old}->{sec.section_id}")
                            moved = True
                            break
                    if not moved:
                        applied.append(f"replace_figure (no movable figure)")
            else:
                patch = _css_global("figures_larger")
                if patch and patch not in css_patches:
                    css_patches.append(patch)
                applied.append("replace_figure enlarge (global)")

        elif action == "supplement":
            if sec:
                blank_reports = (review.deterministic_checks or {}).get("section_blank_reports") or []
                blank_ratio = 0.0
                for item in blank_reports:
                    if isinstance(item, dict) and str(item.get("section_id", "")) == sec.section_id:
                        try:
                            blank_ratio = float(item.get("blank_ratio") or 0.0)
                        except (TypeError, ValueError):
                            blank_ratio = 0.0
                        break
                report = SectionBlankReport(
                    section_id=sec.section_id,
                    blank_ratio=blank_ratio,
                    content_ratio=max(0.0, 1.0 - blank_ratio),
                    width=0,
                    height=0,
                )
                asset_ref = _generate_blank_supplement_asset(llm, sec, report, asset_dirs or [])
                if asset_ref:
                    sec.supplement_html = (
                        '<div class="mini-visual-grid">'
                        f'<div class="mini-node"><div class="mini-node-title">{html.escape(sec.title or sec.type.replace("_", " ").title())}</div><div class="mini-node-copy">Blank ratio {report.blank_ratio:.0%}. Supplemented with a generated visual cue.</div></div>'
                        f'<div class="figure-card"><img src="{asset_ref}" alt="{html.escape(sec.title or sec.section_id)} supplement"></div>'
                        '</div>'
                    )
                    applied.append(f"supplement {sec.section_id} (svg)")
                else:
                    sec.supplement_html = _visual_supplement_html(sec, report)
                    applied.append(f"supplement {sec.section_id}")
            else:
                applied.append(f"supplement skipped ({comment.issue[:60]})")

        elif action == "remove":
            if sec and ("figure" in issue_lower or "image" in issue_lower):
                removed = False
                for fp in list(blueprint.figure_placements):
                    if fp.section_id == sec.section_id:
                        blueprint.figure_placements.remove(fp)
                        removed = True
                if removed:
                    applied.append(f"remove figure in {sec.section_id}")
                else:
                    applied.append(f"remove (no figure in {sec.section_id})")
            else:
                applied.append(f"remove skipped ({sec.section_id if sec else 'no target'})")

    return applied


def _artifact_map(review: PosterReview) -> dict[str, str]:
    artifact_paths = getattr(review, "artifact_paths", {}) or {}
    if not isinstance(artifact_paths, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in artifact_paths.items():
        if not raw:
            continue
        text = str(raw)
        if key in {"sections", "figures"} and text.strip().startswith("{"):
            try:
                parsed = json.loads(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                for sub_key, sub_value in parsed.items():
                    if sub_value:
                        out[str(sub_key)] = str(sub_value)
        else:
            out[str(key)] = text
    return out


def review_rendered_poster_v2(*args, **kwargs):
    """Backward-compatible alias for the legacy v2 harness tests."""
    return review_rendered_poster(*args, **kwargs)


def _review_legacy_contract(*args, **kwargs):
    """Legacy-compatible review path for the v2 harness tests.

    The old harness stubs review_rendered_poster_v2 directly and expects the
    100-point contract / stop-label behavior from src.schemas.review.
    """
    return review_rendered_poster_v2(*args, **kwargs)


# ---------------------------------------------------------------------------
# Loop control
# ---------------------------------------------------------------------------


def _should_stop(review: PosterReview, round_no: int, config: HarnessConfig, scores: list[int]) -> Optional[str]:
    if (
        not review.hard_failures
        and not review.needs_improvement
        and review.quality_score >= config.threshold
    ):
        return "passed"
    if round_no >= config.max_rounds:
        return "max_rounds"
    if len(scores) >= 3:
        last3 = scores[-3:]
        if last3[-1] <= last3[-2] and last3[-2] <= last3[-3]:
            return "plateau"
    return None


def _default_config() -> HarnessConfig:
    return HarnessConfig(
        threshold=settings.harness_threshold,
        max_rounds=settings.harness_max_rounds,
        zoom_crops=settings.harness_zoom_crops,
        max_crops=settings.harness_max_crops,
        enable_qa_eval=settings.harness_enable_qa,
        qa_threshold=settings.harness_qa_threshold,
        vision_model=settings.harness_vision_model or None,
    )


# ---------------------------------------------------------------------------
# QA evaluation (PaperQuiz-lite, ported from the v2 harness draft)
# ---------------------------------------------------------------------------


def generate_paperquiz_questions(doc: PaperDocument, analysis: PaperAnalysis, count: int = 6) -> list[EvaluationQuestion]:
    questions: list[EvaluationQuestion] = []
    if analysis.problem_statement:
        questions.append(EvaluationQuestion(
            question_id="q-problem", question="What problem does this method solve?",
            answer=analysis.problem_statement, evidence=[analysis.problem_statement], category="problem",
        ))
    if analysis.method_overview:
        questions.append(EvaluationQuestion(
            question_id="q-method", question="How does the method work at a high level?",
            answer=analysis.method_overview, evidence=[analysis.method_overview], category="method",
        ))
    for idx, contrib in enumerate(analysis.contributions[:2], start=1):
        questions.append(EvaluationQuestion(
            question_id=f"q-contrib-{idx}", question=f"What is contribution {idx}?",
            answer=contrib.text, evidence=[contrib.text], category="contribution",
        ))
    if analysis.experiments and analysis.experiments.main_results:
        questions.append(EvaluationQuestion(
            question_id="q-result", question="What are the main results?",
            answer=analysis.experiments.main_results, evidence=[analysis.experiments.main_results], category="results",
        ))
    if analysis.experiments and analysis.experiments.takeaways:
        questions.append(EvaluationQuestion(
            question_id="q-takeaway", question="What is an important takeaway from the experiments?",
            answer=analysis.experiments.takeaways[0], evidence=[analysis.experiments.takeaways[0]], category="results",
        ))
    return questions[:count]


def _normalize_text_for_eval(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s.%+-]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _answer_overlap(reference: str, candidate: str) -> bool:
    ref = _normalize_text_for_eval(reference)
    cand = _normalize_text_for_eval(candidate)
    if not ref or not cand:
        return False
    if ref in cand or cand in ref:
        return True
    ref_tokens = {tok for tok in ref.split() if len(tok) > 2}
    cand_tokens = {tok for tok in cand.split() if len(tok) > 2}
    return len(ref_tokens & cand_tokens) >= max(1, min(3, len(ref_tokens) // 3))


def _summarize_poster_text(poster_text: str, max_chars: int = 9000) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", poster_text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def evaluate_poster_qa(
        doc: PaperDocument,
        analysis: PaperAnalysis,
        poster_text: str,
        visual_score: int = 0,
        llm: Optional[LLMClient] = None,
) -> PosterQAEval:
    questions = generate_paperquiz_questions(doc, analysis)
    client = llm or (LLMClient() if LLMClient.is_configured() else None)
    poster_answers: list[str] = []
    correct_count = 0
    for q in questions:
        if client is None:
            poster_answers.append("")
            continue
        prompt = (
            f"Poster content summary:\n{_summarize_poster_text(poster_text)}"
            f"\n\nQuestion: {q.question}\nReturn json with answer, short_reason, and confidence."
        )
        try:
            resp = client.chat_json(system=_QA_SYSTEM_PROMPT, user=prompt)
            answer = str(resp.get("answer", "")).strip() or str(resp.get("poster_answer", "")).strip()
        except Exception as e:
            logger.warning("QA eval failed for %s: %s", q.question_id, e)
            answer = ""
        poster_answers.append(answer)
        if q.answer and answer and _answer_overlap(q.answer, answer):
            correct_count += 1
    total = len(questions)
    accuracy = (correct_count / total) if total else 0.0
    return PosterQAEval(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        questions=questions,
        poster_answers=poster_answers,
        correct_count=correct_count,
        total_count=total,
        accuracy=accuracy,
        coverage=accuracy,
        recall=accuracy,
        visual_score=visual_score,
        qa_score=int(round(accuracy * 10)),
        summary=f"Answered {correct_count}/{total} questions from poster content.",
    )


def evaluate_poster_visual_qa(
        doc: PaperDocument,
        analysis: PaperAnalysis,
        poster_png: Path,
        visual_score: int = 0,
        model: Optional[str] = None,
) -> Optional[PosterQAEval]:
    """Run PaperQuiz-lite against the rendered image, never the HTML source.

    This closes a key loophole in the earlier implementation: an item could be
    present in hidden HTML while being clipped or too small to communicate on
    the actual poster.  Reference answers remain local evidence; the VLM only
    sees the image and the questions.
    """
    questions = generate_paperquiz_questions(doc, analysis)
    if not questions or not poster_png.exists():
        return None
    qa_image = downscale_image(poster_png, max_width=1400) or poster_png
    question_text = "\n".join(
        f"- {question.question_id}: {question.question}" for question in questions
    )
    raw = multimodal_analyze_labeled(
        _VISUAL_QA_SYSTEM_PROMPT,
        [(str(qa_image), "final candidate poster")],
        user_text=f"Questions:\n{question_text}",
        model=model,
    )
    if not raw:
        return None
    by_id = {
        str(item.get("question_id", "")): str(item.get("answer", "")).strip()
        for item in (raw.get("answers") or [])
        if isinstance(item, dict)
    }
    answers = [by_id.get(question.question_id, "") for question in questions]
    correct = sum(
        1 for question, answer in zip(questions, answers)
        if answer and _answer_overlap(question.answer, answer)
    )
    total = len(questions)
    accuracy = correct / total if total else 0.0
    return PosterQAEval(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        questions=questions,
        poster_answers=answers,
        correct_count=correct,
        total_count=total,
        accuracy=accuracy,
        coverage=accuracy,
        recall=accuracy,
        visual_score=visual_score,
        qa_score=int(round(accuracy * 10)),
        summary=f"Image-grounded PaperQuiz answered {correct}/{total} questions.",
    )


# ---------------------------------------------------------------------------
# Harness orchestrator
# ---------------------------------------------------------------------------


def run_poster_harness(
        doc: PaperDocument,
        analysis: PaperAnalysis,
        blueprint: PosterBlueprint,
        html_path: Path | str,
        output_dir: Path | str,
        config: Optional[HarnessConfig] = None,
        on_round: Optional[Callable[[int, int, int, bool, str], None]] = None,
        fallback_optimizer: Optional[Callable[[Path, Path], None]] = None,
) -> HarnessResult:
    """Run the visual review -> feedback -> re-render loop until quality is reached.

    Args:
        doc / analysis: parsed paper data.
        blueprint: the poster blueprint (mutated in place by feedback application).
        html_path: initially rendered poster HTML (draft).
        output_dir: paper output directory (artifacts go to ``<output>/harness/``).
        config: harness loop configuration (defaults to settings).
        on_round: callback(round_no, max_rounds, score, needs_improvement, summary)
                  fired after every review, used by the web app for progress updates.
        fallback_optimizer: optional callable(old_html_path, new_html_path) run when
                            vision review is unavailable (legacy single-shot optimize).

    Returns:
        HarnessResult with per-round records and final artifact paths.
    """
    config = config or _default_config()
    if _is_v2_compat_mode(config):
        return _run_poster_harness_v2_compat(doc, analysis, blueprint, html_path, output_dir, config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_dir = output_dir / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)

    renderer = HtmlPosterRenderer()
    llm = LLMClient() if LLMClient.is_configured() else None
    vision_model = config.vision_model

    css_patches: list[str] = []
    scores: list[int] = []
    rounds: list[HarnessRound] = []
    initial_html = Path(html_path)

    stop_reason = "unknown"
    passed = False
    passing_qa_eval: Optional[PosterQAEval] = None

    for round_no in range(1, config.max_rounds + 1):
        round_dir = harness_dir / f"round_{round_no}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_html = round_dir / "poster.html"
        round_png = round_dir / "poster.png"
        review_json_path = round_dir / "review.json"

        # 1) Render current blueprint (+ accumulated CSS patches) into this round's HTML.
        try:
            html_str = renderer.render(blueprint, doc, round_dir)
        except Exception as e:
            logger.exception("Harness render failed at round %d: %s", round_no, e)
            stop_reason = f"render_error: {e}"
            break
        html_str = _apply_css_patches(html_str, css_patches)
        round_html.write_text(html_str, encoding="utf-8")

        # 2) Capture + visual review.
        review = review_rendered_poster(round_html, round_dir, config, blueprint, model=vision_model)
        if review is None:
            # One retry, then declare vision unavailable.
            import time as _time
            _time.sleep(2)
            review = review_rendered_poster(round_html, round_dir, config, blueprint, model=vision_model)
        if review is None:
            logger.warning("Vision review unavailable; harness cannot continue (round %d)", round_no)
            stop_reason = "vision_unavailable"
            break

        # 3) Merge deterministic density/blank issues with the VLM review,
        #    then persist review.
        heuristic = _heuristic_density_issues(blueprint)
        if heuristic:
            existing_targets = {c.target for c in review.issues}
            review.issues.extend(c for c in heuristic if c.target not in existing_targets)
            if "text_density" not in review.dimension_scores:
                review.dimension_scores["text_density"] = max(1.0, 10.0 - len(heuristic) * 1.5)

        # A visually high-scoring candidate is not qualified until the core
        # paper claims can be recovered from the *image*.  Run this expensive
        # check only for a provisional pass candidate.
        if (
            config.enable_qa_eval
            and not review.hard_failures
            and not review.needs_improvement
            and review.quality_score >= config.threshold
        ):
            candidate_qa = evaluate_poster_visual_qa(
                doc, analysis, round_png, visual_score=review.quality_score,
                model=vision_model,
            )
            if candidate_qa is None:
                review.hard_failures.append("image_grounded_qa_unavailable")
                review.needs_improvement = True
                review.issues.append(PosterComment(
                    issue="Image-grounded content QA was unavailable",
                    severity="error", target="", action="keep",
                    suggestion="Check the configured vision endpoint before certifying this poster.",
                ))
            elif candidate_qa.accuracy < config.qa_threshold:
                review.needs_improvement = True
                review.issues.append(PosterComment(
                    issue=(f"Poster-image content coverage is {candidate_qa.accuracy:.0%}; "
                           f"required {config.qa_threshold:.0%}"),
                    severity="error", target="sec-main-method", action="rewrite",
                    suggestion="Make the problem, method and main result legible in the visible poster copy.",
                ))
            else:
                passing_qa_eval = candidate_qa
        review_json_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")

        # 4) Record round.
        score = review.quality_score
        scores.append(score)
        applied_actions = _apply_feedback(
            blueprint,
            review,
            llm,
            css_patches,
            asset_dirs=[round_dir / "figures", output_dir / "figures"],
        )
        artifact_map = _artifact_map(review)
        rounds.append(HarnessRound(
            round_no=round_no,
            quality_score=score,
            total_score=getattr(review, "total_score", float(score * 10)),
            verdict=getattr(review, "verdict", {}) or {},
            dimension_scores=review.dimension_scores,
            hard_failures=review.hard_failures,
            deterministic_checks=review.deterministic_checks,
            needs_improvement=review.needs_improvement,
            issues=review.issues,
            summary=review.summary,
            applied_actions=applied_actions,
            png_path=str(round_png),
            html_path=str(round_html),
            review_path=str(review_json_path),
            grid_png=str(artifact_map.get("grid_png") or (round_dir / "grid.png").as_posix()) if (artifact_map.get("grid_png") or (round_dir / "grid.png").exists()) else "",
            diff_png=str(artifact_map.get("diff_png") or (round_dir / "diff_vs_prev.png").as_posix()) if (artifact_map.get("diff_png") or (round_dir / "diff_vs_prev.png").exists()) else "",
            section_crops={k: v for k, v in artifact_map.items() if k.startswith("sec-") or k.startswith("sections/")},
            figure_crops={k: v for k, v in artifact_map.items() if k.startswith("fig_") or k.startswith("figures/")},
            captured_at=datetime.now(timezone.utc).isoformat(),
        ))

        if on_round:
            try:
                on_round(round_no, config.max_rounds, score, review.needs_improvement, review.summary)
            except Exception:
                logger.warning("on_round callback failed", exc_info=True)

        # 5) Gate check.
        reason = _should_stop(review, round_no, config, scores)
        if reason:
            stop_reason = reason
            passed = reason == "passed"
            break

    # -- Fallback path: vision review never worked ---------------------------------
    if not rounds:
        final_html = initial_html
        if fallback_optimizer is not None:
            fallback_path = output_dir / "poster_optimized.html"
            try:
                fallback_optimizer(initial_html, fallback_path)
                final_html = fallback_path
                fallback_note = "vision unavailable; used legacy single-shot HTML optimization"
            except Exception as e:
                logger.exception("Fallback optimizer failed: %s", e)
                fallback_note = f"vision unavailable and fallback failed: {e}"
        else:
            fallback_note = "vision unavailable; no fallback optimizer provided"
        result = HarnessResult(
            passed=False,
            stop_reason=stop_reason if stop_reason != "unknown" else "vision_unavailable",
            rounds=[],
            final_html=str(final_html),
            final_png="",
            fallback=True,
            fallback_reason=fallback_note,
            total_rounds=0,
        )
        _write_report(output_dir, result, config)
        return result

    # -- Select best round and write final artifacts --------------------------------
    best = max(rounds, key=lambda r: r.quality_score)
    best_score = best.quality_score
    best_png = Path(best.png_path) if best.png_path else None
    best_html = Path(best.html_path) if best.html_path else None

    final_html_path = output_dir / "poster_final.html"
    final_png_path = output_dir / "poster_final.png"
    if best_html and best_html.exists():
        final_html_path.write_bytes(best_html.read_bytes())
    if best_png and best_png.exists():
        final_png_path.write_bytes(best_png.read_bytes())

    result = HarnessResult(
        passed=passed,
        stop_reason=stop_reason,
        rounds=rounds,
        best_round_no=best.round_no,
        best_score=best_score,
        final_html=str(final_html_path),
        final_png=str(final_png_path) if final_png_path.exists() else "",
        total_rounds=len(rounds),
    )
    _write_report(output_dir, result, config)
    return result


def _is_v2_compat_mode(config: HarnessConfig) -> bool:
    extra = getattr(config, "__pydantic_extra__", None) or {}
    return any(key in extra for key in {"advanced_visual", "pass_total", "pass_dim_fraction", "plateau_rounds", "improvement_delta", "max_figure_crops"})


def _legacy_should_stop(total_scores: list[float], review: PosterReview, round_no: int, config: HarnessConfig) -> Optional[str]:
    extra = getattr(config, "__pydantic_extra__", None) or {}
    pass_total = float(extra.get("pass_total", 85.0) or 85.0)
    plateau_rounds = int(extra.get("plateau_rounds", 2) or 2)
    improvement_delta = float(extra.get("improvement_delta", 2.0) or 2.0)

    verdict = getattr(review, "verdict", {}) or {}
    passed = bool(verdict.get("passed")) if isinstance(verdict, dict) else False
    if passed and not review.hard_failures:
        return "passed"
    if round_no >= config.max_rounds:
        return "max_rounds"
    if len(total_scores) >= 3:
        last = total_scores[-3:]
        if (last[1] - last[0]) < improvement_delta and (last[2] - last[1]) < improvement_delta:
            return "stopped_not_passing"
    if getattr(review, "total_score", 0.0) >= pass_total and not review.hard_failures:
        return "passed"
    return None

    # -- Persist only image-grounded QA.  HTML-source QA is deliberately not a
    #    certification signal because it cannot prove the information is visible.
    if passing_qa_eval is not None:
        qa_path = output_dir / "poster_qa_eval.json"
        qa_path.write_text(passing_qa_eval.model_dump_json(indent=2), encoding="utf-8")
        result.qa_eval_path = str(qa_path)

    _write_report(output_dir, result, config)
    return result


def _run_poster_harness_v2_compat(
        doc: PaperDocument,
        analysis: PaperAnalysis,
        blueprint: PosterBlueprint,
        html_path: Path | str,
        output_dir: Path | str,
        config: HarnessConfig,
) -> HarnessResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_dir = output_dir / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)

    rounds: list[HarnessRound] = []
    total_scores: list[float] = []
    stop_reason = "unknown"
    passed = False

    for round_no in range(1, config.max_rounds + 1):
        round_dir = harness_dir / f"round_{round_no}"
        round_dir.mkdir(parents=True, exist_ok=True)
        review = review_rendered_poster_v2(Path(html_path), round_dir, config, blueprint)
        if review is None:
            stop_reason = "vision_unavailable"
            break
        total = float(getattr(review, "total_score", review.quality_score * 10))
        total_scores.append(total)
        artifact_map = _artifact_map(review)
        rounds.append(HarnessRound(
            round_no=round_no,
            quality_score=review.quality_score,
            total_score=total,
            verdict=getattr(review, "verdict", {}) or {},
            dimension_scores=getattr(review, "dimension_scores", {}) or {},
            hard_failures=getattr(review, "hard_failures", []) or [],
            deterministic_checks=getattr(review, "deterministic_checks", {}) or {},
            needs_improvement=getattr(review, "needs_improvement", True),
            issues=getattr(review, "issues", []) or [],
            summary=getattr(review, "summary", ""),
            applied_actions=[],
            png_path=str(artifact_map.get("full_png", "")),
            html_path=str(Path(html_path)),
            review_path=str(round_dir / "review.json"),
            grid_png=str(artifact_map.get("grid_png", "")),
            diff_png=str(artifact_map.get("diff_png", "")),
            section_crops={k: v for k, v in artifact_map.items() if k.startswith("sec-") or k.startswith("sections/")},
            figure_crops={k: v for k, v in artifact_map.items() if k.startswith("fig_") or k.startswith("figures/")},
            captured_at=datetime.now(timezone.utc).isoformat(),
        ))
        review_json = round_dir / "review.json"
        review_json.write_text(review.model_dump_json(indent=2), encoding="utf-8")
        reason = _legacy_should_stop(total_scores, review, round_no, config)
        if reason:
            stop_reason = reason
            passed = reason == "passed"
            break

    if not rounds:
        result = HarnessResult(passed=False, stop_reason=stop_reason if stop_reason != "unknown" else "vision_unavailable",
                               rounds=[], final_html=str(html_path), final_png="", fallback=True,
                               fallback_reason="vision unavailable; no fallback optimizer provided", total_rounds=0)
        _write_report_v2_compat(output_dir, result, total_scores)
        return result

    best = max(rounds, key=lambda r: r.total_score)
    result = HarnessResult(
        passed=passed,
        stop_reason=stop_reason,
        rounds=rounds,
        best_round_no=best.round_no,
        best_score=best.quality_score,
        final_html=str(html_path),
        final_png=str(rounds[-1].png_path) if rounds[-1].png_path else "",
        total_rounds=len(rounds),
    )
    _write_report_v2_compat(output_dir, result, total_scores)
    return result


def _write_report_v2_compat(output_dir: Path, result: HarnessResult, total_scores: list[float]) -> Path:
    report = {
        "passed": result.passed,
        "stop_reason": result.stop_reason,
        "stop_label": result.stop_label,
        "total_scores": total_scores,
        "best_round_no": result.best_round_no,
        "best_total": result.best_total,
        "final_html": result.final_html,
        "final_png": result.final_png,
        "rounds": [
            {
                "round_no": r.round_no,
                "total_score": r.total_score,
                "quality_score": r.quality_score,
                "verdict": r.verdict,
                "grid_png": r.grid_png,
                "diff_png": r.diff_png,
                "section_crops": r.section_crops,
                "figure_crops": r.figure_crops,
            }
            for r in result.rounds
        ],
    }
    report_path = output_dir / "harness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    result.report_path = str(report_path)
    return report_path


def _write_report(output_dir: Path, result: HarnessResult, config: Optional[HarnessConfig] = None) -> Path:
    report = {
        "passed": result.passed,
        "stop_reason": result.stop_reason,
        "threshold": config.threshold if config else None,
        "max_rounds": config.max_rounds if config else None,
        "qa_threshold": config.qa_threshold if config else None,
        "scores": [r.quality_score for r in result.rounds],
        "best_round_no": result.best_round_no,
        "best_score": result.best_score,
        "final_html": result.final_html,
        "final_png": result.final_png,
        "fallback": result.fallback,
        "fallback_reason": result.fallback_reason,
        "qa_eval_path": result.qa_eval_path,
        "total_rounds": result.total_rounds,
        "rounds": [
            {
                "round_no": r.round_no,
                "quality_score": r.quality_score,
                "dimension_scores": r.dimension_scores,
                "hard_failures": r.hard_failures,
                "deterministic_checks": r.deterministic_checks,
                "needs_improvement": r.needs_improvement,
                "summary": r.summary,
                "issues": [c.model_dump() for c in r.issues],
                "applied_actions": r.applied_actions,
                "png_path": r.png_path,
                "html_path": r.html_path,
            }
            for r in result.rounds
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "harness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    result.report_path = str(report_path)
    return report_path
