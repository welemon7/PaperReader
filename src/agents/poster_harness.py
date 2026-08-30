from __future__ import annotations

import html
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.llm.client import LLMClient, LLMError
from src.llm.multimodal_client import (
    capture_poster_full_and_sections,
)
from src.visual.capture import measure_section_content_size
from src.renderers.html_renderer import HtmlPosterRenderer
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import Figure, PaperDocument
from src.schemas.poster import PosterBlueprint, PosterSection
from src.schemas.poster_harness import HarnessConfig, HarnessResult, HarnessRound
from src.schemas.poster_v2 import PosterReview
from src.agents.content_policy import count_words
from src.utils.figure_assets import save_svg_asset, sanitize_asset_name
from src.agents.svg_skill_adapter import (
    normalize_svg_dimensions,
    svg_generation_guidance,
    validate_svg_document,
)

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError:  # pragma: no cover - PIL is a hard dependency in practice
    Image = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]


_INVALID_BLANK_RATIO_THRESHOLD = 0.35
_BLANK_GRAY_VARIANCE_THRESHOLD = 10.0
_MORPHOLOGY_KERNEL_SIZE = 15
_HIGHLIGHTS_FILE = Path(__file__).resolve().parents[2] / "highlights.md"


@dataclass
class SectionBlankReport:
    section_id: str
    section_type: str
    section_title: str
    blank_ratio: float
    content_ratio: float
    width: int
    height: int
    text_words: int
    figure_count: int
    has_figures: bool
    crop_path: str = ""
    blank_cells: list[dict[str, Any]] = field(default_factory=list)
    blank_regions: list[dict[str, Any]] = field(default_factory=list)
    core_blank_review: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlankRegionCandidate:
    section_id: str
    section_type: str
    section_title: str
    blank_ratio: float
    content_ratio: float
    width: int
    height: int
    text_words: int
    figure_count: int
    has_figures: bool
    local_context: str
    nearby_context: str
    global_context: str
    key_signals: list[str] = field(default_factory=list)
    blank_cells: list[dict[str, Any]] = field(default_factory=list)
    crop_path: str = ""
    blank_regions: list[dict[str, Any]] = field(default_factory=list)
    core_blank_review: dict[str, Any] = field(default_factory=dict)


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


def _section_lookup(blueprint: PosterBlueprint) -> dict[str, PosterSection]:
    return {sec.section_id: sec for sec in blueprint.sections}


def _dense_section_ids(blueprint: PosterBlueprint, limit: int) -> list[str]:
    """Pick the most text-dense sections for zoomed-in crops."""
    candidates = [
        sec for sec in blueprint.sections
        if sec.type not in {"title", "project_link"} and (sec.content_md or "").strip()
    ]
    candidates.sort(key=lambda s: len(s.content_md or ""), reverse=True)
    return [sec.section_id for sec in candidates[:limit]]


def _section_words(sec: PosterSection) -> int:
    return count_words(sec.content_md)


def _section_figure_count(section_id: str, blueprint: PosterBlueprint) -> int:
    return sum(1 for fp in blueprint.figure_placements if fp.section_id == section_id)


def _section_has_figures(section_id: str, blueprint: PosterBlueprint) -> bool:
    return _section_figure_count(section_id, blueprint) > 0


def _section_neighbor_text(blueprint: PosterBlueprint, sec: PosterSection, radius: int = 1) -> str:
    ordered = sorted(
        [s for s in blueprint.sections if s.type != "title"],
        key=lambda s: (s.row, s.column, s.section_id),
    )
    try:
        idx = next(i for i, item in enumerate(ordered) if item.section_id == sec.section_id)
    except StopIteration:
        return ""
    parts: list[str] = []
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        pos = idx + offset
        if 0 <= pos < len(ordered):
            other = ordered[pos]
            text = _section_content_preview(other, limit=260)
            parts.append(f"{other.section_id} ({other.type}): {text}")
    return "\n".join(parts)


def _global_poster_context(doc: PaperDocument, analysis: PaperAnalysis, blueprint: PosterBlueprint) -> str:
    contributions = " | ".join(
        _short if (_short := re.sub(r"\s+", " ", c.text).strip()) else ""
        for c in analysis.contributions[:4]
        if (c.text or "").strip()
    )
    experiments = analysis.experiments.main_results if analysis.experiments and analysis.experiments.main_results else ""
    method = (analysis.method_overview or "").strip()
    problem = (analysis.problem_statement or "").strip()
    return (
        f"Paper title: {doc.title}\n"
        f"Problem: {problem or '(not provided)'}\n"
        f"Method: {method or '(not provided)'}\n"
        f"Main results: {experiments or '(not provided)'}\n"
        f"Contributions: {contributions or '(not provided)'}\n"
        f"Poster title: {blueprint.poster_title}\n"
        f"Poster sections: {', '.join(sec.type for sec in blueprint.sections if sec.type != 'title')}"
    )


def _fallback_poster_context(blueprint: PosterBlueprint) -> str:
    return (
        f"Poster title: {blueprint.poster_title}\n"
        f"Poster sections: {', '.join(sec.type for sec in blueprint.sections if sec.type != 'title')}\n"
        f"Layout rows: {', '.join(str(sec.row) for sec in blueprint.sections if sec.type != 'title')}"
    )


def _blank_ratio_threshold(sec: PosterSection) -> float:
    del sec
    return _INVALID_BLANK_RATIO_THRESHOLD


def _should_supplement_report(report: SectionBlankReport, sec: PosterSection) -> bool:
    if sec.type in {"contributions", "highlights", "project_link"}:
        return False
    return report.blank_ratio >= _INVALID_BLANK_RATIO_THRESHOLD


def _connected_white_regions(binary: Any, white_threshold: int = 240) -> list[dict[str, Any]]:
    """Return 8-connected white components from a PIL grayscale image."""
    width, height = binary.size
    pixels = binary.load()
    seen = bytearray(width * height)
    regions: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if seen[index] or pixels[x, y] < white_threshold:
                continue
            seen[index] = 1
            stack = [(x, y)]
            area = 0
            left = right = x
            top = bottom = y
            while stack:
                current_x, current_y = stack.pop()
                area += 1
                left = min(left, current_x)
                right = max(right, current_x)
                top = min(top, current_y)
                bottom = max(bottom, current_y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if not dx and not dy:
                            continue
                        next_x = current_x + dx
                        next_y = current_y + dy
                        if not (0 <= next_x < width and 0 <= next_y < height):
                            continue
                        next_index = next_y * width + next_x
                        if seen[next_index] or pixels[next_x, next_y] < white_threshold:
                            continue
                        seen[next_index] = 1
                        stack.append((next_x, next_y))
            regions.append({
                "component_area": area,
                "x": left,
                "y": top,
                "width": right - left + 1,
                "height": bottom - top + 1,
                "touches_border": left == 0 or top == 0 or right == width - 1 or bottom == height - 1,
            })
    return regions


def _location_hint_to_region(location: str) -> dict[str, float]:
    """Convert a coarse VLM location into a conservative normalized crop hint."""
    location = (location or "none").strip().lower()
    if location == "none":
        return {}
    horizontal = "right" if "right" in location else ("left" if "left" in location else "center")
    vertical = "bottom" if "bottom" in location else ("top" if "top" in location else "center")
    x = 0.5 if horizontal == "right" else (0.0 if horizontal == "left" else 0.25)
    y = 0.5 if vertical == "bottom" else (0.0 if vertical == "top" else 0.25)
    width = 0.5 if horizontal in {"left", "right"} else 0.5
    height = 0.5 if vertical in {"top", "bottom"} else 0.5
    return {"x": x, "y": y, "width": width, "height": height}


def _normalize_region_hint(value: object, location: str = "none") -> dict[str, float]:
    if not isinstance(value, dict):
        return _location_hint_to_region(location)
    result: dict[str, float] = {}
    for key in ("x", "y", "width", "height"):
        try:
            result[key] = min(1.0, max(0.0, float(value.get(key))))
        except (TypeError, ValueError):
            return _location_hint_to_region(location)
    if result["width"] <= 0 or result["height"] <= 0:
        return _location_hint_to_region(location)
    result["width"] = round(min(result["width"], 1.0 - result["x"]), 6)
    result["height"] = round(min(result["height"], 1.0 - result["y"]), 6)
    return result


def _largest_pure_background_rectangle(
    gray: Any,
    *,
    white_threshold: int,
    min_area_ratio: float,
) -> dict[str, Any] | None:
    """Find the largest all-background rectangle for an edge-connected component.

    Text often leaves the outer white background as one connected component.
    Its full bounding box contains text and therefore fails the variance test,
    while a bottom or side strip can still be a genuine empty rectangle.  This
    maximal-rectangle pass recovers that geometry without treating pale charts
    as blank: the final rectangle is still checked against raw-pixel variance.
    """
    width, height = gray.size
    pixels = gray.load()
    total = max(1, width * height)
    histogram = [0] * width
    best: dict[str, Any] | None = None
    for y in range(height):
        for x in range(width):
            histogram[x] = histogram[x] + 1 if pixels[x, y] >= white_threshold else 0
        stack: list[tuple[int, int]] = []
        for index in range(width + 1):
            current = histogram[index] if index < width else 0
            start = index
            while stack and stack[-1][1] > current:
                previous_start, previous_height = stack.pop()
                rect_width = index - previous_start
                area = rect_width * previous_height
                if area >= (best or {}).get("area_pixels", 0):
                    candidate = {
                        "x": previous_start,
                        "y": y - previous_height + 1,
                        "width": rect_width,
                        "height": previous_height,
                        "area_pixels": area,
                    }
                    crop = gray.crop((candidate["x"], candidate["y"], candidate["x"] + rect_width, candidate["y"] + previous_height))
                    variance = float(ImageStat.Stat(crop).var[0])
                    if variance < _BLANK_GRAY_VARIANCE_THRESHOLD and area / total >= min_area_ratio:
                        candidate["gray_variance"] = round(variance, 4)
                        best = candidate
                start = previous_start
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    return best


def _analyze_blank_regions(
    png_path: Path,
    *,
    white_threshold: int = 240,
    kernel_size: int = _MORPHOLOGY_KERNEL_SIZE,
    region_hint: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """Detect pure-background regions using closing, components and variance.

    The binary convention is black=foreground and white=background.  PIL's
    MinFilter followed by MaxFilter is the equivalent of a 15x15 closing for
    that convention: it dilates black text strokes so nearby strokes connect,
    then restores their approximate edges before white components are measured.
    """
    if Image is None or ImageFilter is None or ImageStat is None or not png_path.exists():
        return []
    try:
        with Image.open(png_path) as source:
            full_gray = source.convert("L")
            full_width, full_height = full_gray.size
            total = float(max(1, full_width * full_height))
            base_x = base_y = 0
            gray = full_gray
            if region_hint:
                hint_x = min(1.0, max(0.0, float(region_hint.get("x") or 0.0)))
                hint_y = min(1.0, max(0.0, float(region_hint.get("y") or 0.0)))
                hint_w = min(1.0 - hint_x, max(0.0, float(region_hint.get("width") or 0.0)))
                hint_h = min(1.0 - hint_y, max(0.0, float(region_hint.get("height") or 0.0)))
                base_x = int(full_width * hint_x)
                base_y = int(full_height * hint_y)
                right = max(base_x + 1, min(full_width, base_x + int(full_width * hint_w)))
                bottom = max(base_y + 1, min(full_height, base_y + int(full_height * hint_h)))
                gray = full_gray.crop((base_x, base_y, right, bottom))
            width, height = gray.size
            if width <= 0 or height <= 0:
                return []
            binary = gray.point(lambda value: 0 if value < white_threshold else 255)
            closed = binary.filter(ImageFilter.MinFilter(kernel_size)).filter(ImageFilter.MaxFilter(kernel_size))
            min_area_ratio = min(0.01, _INVALID_BLANK_RATIO_THRESHOLD / 10.0)
            regions: list[dict[str, Any]] = []
            for component in _connected_white_regions(closed, white_threshold):
                x = int(component["x"])
                y = int(component["y"])
                right = x + int(component["width"])
                bottom = y + int(component["height"])
                variance = float(ImageStat.Stat(gray.crop((x, y, right, bottom))).var[0])
                area = int(component["component_area"])
                if variance < _BLANK_GRAY_VARIANCE_THRESHOLD and area / total >= min_area_ratio:
                    region = {
                        "x": x + base_x,
                        "y": y + base_y,
                        "width": int(component["width"]),
                        "height": int(component["height"]),
                        "area_pixels": area,
                        "area_ratio": round(area / total, 6),
                        "blank_ratio": round(area / total, 6),
                        "gray_variance": round(variance, 4),
                        "method": "closed_white_component",
                        "touches_border": bool(component["touches_border"]),
                    }
                    regions.append(region)

            # Recover the largest pure strip inside a text-connected background.
            rectangle = _largest_pure_background_rectangle(
                gray,
                white_threshold=white_threshold,
                min_area_ratio=min_area_ratio,
            )
            if rectangle:
                rectangle["x"] = int(rectangle["x"]) + base_x
                rectangle["y"] = int(rectangle["y"]) + base_y
                rectangle["area_ratio"] = round(rectangle["area_pixels"] / total, 6)
                rectangle["blank_ratio"] = rectangle["area_ratio"]
                rectangle["method"] = "pure_background_rectangle"
                rectangle["touches_border"] = (
                    rectangle["x"] == 0 or rectangle["y"] == 0
                    or rectangle["x"] + rectangle["width"] == width
                    or rectangle["y"] + rectangle["height"] == height
                )
                largest_region_area = max(
                    (int(region.get("area_pixels") or 0) for region in regions),
                    default=0,
                )
                if int(rectangle["area_pixels"]) > largest_region_area:
                    regions.append(rectangle)

            regions.sort(key=lambda item: float(item.get("area_pixels") or 0), reverse=True)
            return regions
    except Exception as exc:
        logger.warning("Blank region analysis failed for %s: %s", png_path, exc)
        return []


def _analyze_blank_cells(png_path: Path, *, white_threshold: int = 240, grid_x: int = 4, grid_y: int = 4) -> list[dict[str, Any]]:
    """Backward-compatible name for the precise blank-region analyzer."""
    del grid_x, grid_y
    return _analyze_blank_regions(png_path, white_threshold=white_threshold)


def _image_size(png_path: Path) -> tuple[int, int]:
    if Image is None or not png_path.exists():
        return 0, 0
    try:
        with Image.open(png_path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return 0, 0


def _make_blank_region_candidates(
    blueprint: PosterBlueprint,
    doc: PaperDocument,
    analysis: PaperAnalysis,
    round_dir: Path,
    core_blank_review: Optional[object] = None,
) -> list[BlankRegionCandidate]:
    reports = _section_blank_reports(round_dir, blueprint, core_blank_review)
    if not reports:
        return []
    lookup = _section_lookup(blueprint)
    global_context = _global_poster_context(doc, analysis, blueprint)
    candidates: list[BlankRegionCandidate] = []
    for report in reports:
        sec = lookup.get(report.section_id)
        if sec is None:
            continue
        if not _should_supplement_report(report, sec):
            continue
        local_context = _section_content_preview(sec, limit=650)
        nearby_context = _section_neighbor_text(blueprint, sec, radius=1)
        signals: list[str] = []
        if analysis.problem_statement:
            signals.append(_section_content_preview(PosterSection(section_id="tmp", type=sec.type, title=sec.title, content_md=analysis.problem_statement), limit=180))
        if analysis.method_overview:
            signals.append(_section_content_preview(PosterSection(section_id="tmp", type=sec.type, title=sec.title, content_md=analysis.method_overview), limit=180))
        if analysis.experiments and analysis.experiments.main_results:
            signals.append(_section_content_preview(PosterSection(section_id="tmp", type=sec.type, title=sec.title, content_md=analysis.experiments.main_results), limit=180))
        crop_path = report.crop_path or str((round_dir / "sections" / f"{sec.section_id}.png").resolve())
        candidates.append(BlankRegionCandidate(
            section_id=sec.section_id,
            section_type=sec.type,
            section_title=sec.title or sec.type,
            blank_ratio=report.blank_ratio,
            content_ratio=report.content_ratio,
            width=report.width,
            height=report.height,
            text_words=report.text_words,
            figure_count=report.figure_count,
            has_figures=report.has_figures,
            local_context=local_context,
            nearby_context=nearby_context,
            global_context=global_context,
            key_signals=[s for s in signals if s],
            blank_cells=list(report.blank_cells),
            blank_regions=list(report.blank_regions),
            crop_path=crop_path,
            core_blank_review=dict(report.core_blank_review),
        ))
    candidates.sort(key=lambda c: (c.blank_ratio, -c.text_words, -c.figure_count), reverse=True)
    return candidates


def _measure_png_blank_ratio(png_path: Path) -> Optional[float]:
    if Image is None or not png_path.exists():
        return None
    try:
        with Image.open(png_path) as img:
            gray = img.convert("L")
            width, height = gray.size
            if width <= 0 or height <= 0:
                return None
            pixels = gray.load()
            total = width * height
            blank = 0
            for y in range(height):
                for x in range(width):
                    if pixels[x, y] >= 240:
                        blank += 1
            return blank / total if total else None
    except Exception as exc:
        logger.warning("Blank ratio probe failed for %s: %s", png_path, exc)
        return None


def _section_blank_reports(
    round_dir: Path,
    blueprint: PosterBlueprint,
    core_blank_review: Optional[object] = None,
) -> list[SectionBlankReport]:
    sections_dir = round_dir / "sections"
    if not sections_dir.exists():
        sections_dir = round_dir
    lookup = _section_lookup(blueprint)
    reports: list[SectionBlankReport] = []
    for png_path in sorted(sections_dir.glob("*.png")):
        ratio = _measure_png_blank_ratio(png_path)
        if ratio is None:
            continue
        section_id = png_path.stem
        sec = lookup.get(section_id)
        if sec is None and section_id.startswith("poster_"):
            sec = lookup.get(section_id.removeprefix("poster_"))
        if sec is None:
            continue
        width, height = _image_size(png_path)
        blank_regions = _analyze_blank_regions(png_path)
        blank_cells = list(blank_regions)
        region_ratio = min(
            1.0,
            sum(float(region.get("area_ratio") or 0.0) for region in blank_regions),
        )
        reports.append(SectionBlankReport(
            section_id=sec.section_id,
            section_type=sec.type,
            section_title=sec.title or sec.type,
            blank_ratio=region_ratio,
            content_ratio=max(0.0, 1.0 - region_ratio),
            width=width,
            height=height,
            text_words=_section_words(sec),
            figure_count=_section_figure_count(sec.section_id, blueprint),
            has_figures=_section_has_figures(sec.section_id, blueprint),
            crop_path=str(png_path.resolve()),
            blank_cells=blank_cells,
            blank_regions=blank_regions,
            core_blank_review={},
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


def _section_content_preview(sec: PosterSection, limit: int = 900) -> str:
    raw = (sec.content_md or "").strip()
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return "(empty)"
    if len(raw) <= limit:
        return raw
    return raw[:limit].rstrip() + "…"


def _blank_region_visual_prompt(candidate: BlankRegionCandidate) -> str:
    blank_cells = candidate.blank_regions[:8] or candidate.blank_cells[:8]
    cell_lines = ", ".join(
        f"x={cell.get('x', 0)}, y={cell.get('y', 0)}, "
        f"w={cell.get('width', 0)}, h={cell.get('height', 0)}, "
        f"area={float(cell.get('area_ratio') or cell.get('blank_ratio') or 0.0):.0%}, "
        f"variance={float(cell.get('gray_variance') or 0.0):.2f}"
        for cell in blank_cells
        if isinstance(cell, dict)
    ) or "none"
    return (
        f"Target section: {candidate.section_id} ({candidate.section_type})\n"
        f"Section title: {candidate.section_title}\n"
        f"Canvas size: {candidate.width}x{candidate.height}\n"
        f"Blank ratio: {candidate.blank_ratio:.0%}\n"
        f"Content ratio: {candidate.content_ratio:.0%}\n"
        f"Text words: {candidate.text_words}\n"
        f"Figures in section: {candidate.figure_count}\n"
        f"Detected blank cells: {cell_lines}\n\n"
        f"Core VLM blank-location prior: {candidate.core_blank_review or '(not applicable)'}\n\n"
        "Local context:\n"
        f"{candidate.local_context or '(empty)'}\n\n"
        "Nearby context:\n"
        f"{candidate.nearby_context or '(none)'}\n\n"
        "Global context:\n"
        f"{candidate.global_context or '(none)'}\n\n"
        "Key signals:\n"
        f"{chr(10).join('- ' + s for s in candidate.key_signals) if candidate.key_signals else '(none)'}\n\n"
        "Design task: generate a polished self-contained SVG that fills the detected blank region with a visual grounded in the poster content. "
        "Choose the simplest visual grammar that preserves the content relationships, such as a flow, comparison, timeline, architecture, chart, or symbolic concept. "
        "Let the supplied dimensions determine orientation, density, spacing, and label wrapping. Preserve important content rather than imposing a fixed label count or palette. "
        "Match the visual emphasis to the blank geometry; do not add decorative elements unrelated to the content. "
        "Return ONLY a valid standalone SVG document."
    )


def _fallback_supplement_svg(candidate: BlankRegionCandidate) -> str:
    region = max(
        candidate.blank_regions or candidate.blank_cells or [{}],
        key=lambda item: float(item.get("area_pixels") or item.get("area_ratio") or 0.0),
    )
    target_width = max(1, int(region.get("width") or candidate.width or 360))
    target_height = max(1, int(region.get("height") or candidate.height or 220))
    accent = {
        "motivation": "#d1495b",
        "method_overview": "#1f4a75",
        "key_idea": "#8b5cf6",
        "main_method": "#0f766e",
        "experiments": "#a16207",
        "contributions": "#2563eb",
        "highlights": "#7c3aed",
        "project_link": "#059669",
    }.get(candidate.section_type, "#16324f")
    secondary = "#c9a84c"
    if candidate.section_type in {"contributions", "highlights", "project_link"}:
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{target_width}" height="{target_height}" viewBox="0 0 360 220" preserveAspectRatio="none">
  <rect width="360" height="220" rx="18" fill="#ffffff"/>
  <rect x="24" y="26" width="312" height="168" rx="14" fill="#f8fafc" stroke="#e2e8f0"/>
  <rect x="46" y="58" width="54" height="18" rx="9" fill="{accent}" opacity="0.18"/>
  <rect x="110" y="58" width="76" height="18" rx="9" fill="{secondary}" opacity="0.22"/>
  <rect x="196" y="58" width="92" height="18" rx="9" fill="{accent}" opacity="0.18"/>
  <rect x="46" y="98" width="256" height="14" rx="7" fill="{accent}" opacity="0.22"/>
  <rect x="46" y="126" width="198" height="14" rx="7" fill="{secondary}" opacity="0.26"/>
  <rect x="46" y="154" width="232" height="14" rx="7" fill="{accent}" opacity="0.20"/>
</svg>'''
    if candidate.section_type in {"motivation", "method_overview", "key_idea"}:
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{target_width}" height="{target_height}" viewBox="0 0 360 220" preserveAspectRatio="none">
  <rect width="360" height="220" rx="18" fill="#ffffff"/>
  <rect x="28" y="28" width="304" height="164" rx="16" fill="#f8fafc" stroke="#e2e8f0"/>
  <rect x="52" y="58" width="62" height="84" rx="10" fill="{accent}" opacity="0.18" stroke="{accent}"/>
  <rect x="150" y="58" width="62" height="84" rx="10" fill="{secondary}" opacity="0.28" stroke="{secondary}"/>
  <rect x="248" y="58" width="62" height="84" rx="10" fill="{accent}" opacity="0.18" stroke="{accent}"/>
  <path d="M114 100H144" stroke="{accent}" stroke-width="8" stroke-linecap="round"/>
  <path d="M212 100H242" stroke="{secondary}" stroke-width="8" stroke-linecap="round"/>
  <path d="M138 86l10 14-10 14" fill="none" stroke="{accent}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M236 86l10 14-10 14" fill="none" stroke="{secondary}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{target_width}" height="{target_height}" viewBox="0 0 360 220" preserveAspectRatio="none">
  <rect width="360" height="220" rx="18" fill="#ffffff"/>
  <rect x="24" y="28" width="312" height="164" rx="16" fill="#f8fafc" stroke="#e2e8f0"/>
  <rect x="54" y="52" width="252" height="18" rx="9" fill="{accent}" opacity="0.2"/>
  <rect x="54" y="84" width="154" height="18" rx="9" fill="{secondary}" opacity="0.25"/>
  <rect x="54" y="116" width="216" height="18" rx="9" fill="{accent}" opacity="0.16"/>
  <rect x="54" y="148" width="120" height="18" rx="9" fill="{secondary}" opacity="0.22"/>
</svg>'''


def _size_supplement_svg(svg_text: str, candidate: BlankRegionCandidate) -> str:
    """Give generated SVGs the detected blank rectangle's intrinsic size."""
    region = max(
        candidate.blank_regions or candidate.blank_cells or [{}],
        key=lambda item: float(item.get("area_pixels") or item.get("area_ratio") or 0.0),
    )
    width = max(1, int(region.get("width") or candidate.width or 360))
    height = max(1, int(region.get("height") or candidate.height or 220))
    match = re.search(r"<svg\b([^>]*)>", svg_text, flags=re.IGNORECASE)
    if not match:
        return svg_text
    attrs = match.group(1)
    for name, value in (("width", width), ("height", height)):
        pattern = rf"\s{name}\s*=\s*(['\"]).*?\1"
        replacement = f' {name}="{value}"'
        if re.search(pattern, attrs, flags=re.IGNORECASE):
            attrs = re.sub(pattern, replacement, attrs, count=1, flags=re.IGNORECASE)
        else:
            attrs += replacement
    return svg_text[:match.start(1)] + attrs + svg_text[match.end(1):]


def _supplement_overlay_html(asset_ref: str, candidate: BlankRegionCandidate, alt: str) -> str:
    """Build a non-flow SVG overlay positioned in the section crop coordinates."""
    description = _supplement_description(candidate, alt)
    regions = candidate.blank_regions or candidate.blank_cells
    if not regions:
        return f'<div class="figure-card"><div class="figure-description">{html.escape(description)}</div><img src="{html.escape(asset_ref, quote=True)}" alt="{html.escape(alt, quote=True)} supplement"></div>'
    region = max(
        regions,
        key=lambda item: float(item.get("area_pixels") or item.get("area_ratio") or 0.0),
    )
    region_x = int(region.get("x") or 0)
    region_y = int(region.get("y") or 0)
    region_right = region_x + max(1, int(region.get("width") or candidate.width or 360))
    region_bottom = region_y + max(1, int(region.get("height") or candidate.height or 220))
    canvas_width = max(1, candidate.width or region_right)
    canvas_height = max(1, candidate.height or region_bottom)
    # Section crops include the border, title band, and content padding. Map the
    # detected crop coordinates into the content box so the overlay stays in
    # the section without changing its grid/flex dimensions.
    content_left = 18
    content_top = 54
    content_right = max(content_left, canvas_width - 18)
    content_bottom = max(content_top, canvas_height - 16)
    left_edge = max(region_x, content_left)
    top_edge = max(region_y, content_top)
    right_edge = min(region_right, content_right)
    bottom_edge = min(region_bottom, content_bottom)
    if right_edge <= left_edge or bottom_edge <= top_edge:
        return ""
    left = left_edge - content_left
    top = top_edge - content_top
    width = right_edge - left_edge
    height = bottom_edge - top_edge
    return (
        '<div class="blank-region-supplement" '
        f'style="left:{left}px;top:{top}px;width:{width}px;height:{height}px;">'
        f'<div class="figure-description">{html.escape(description)}</div>'
        f'<img src="{html.escape(asset_ref, quote=True)}" alt="{html.escape(alt, quote=True)} supplement">'
        '</div>'
    )


def _supplement_description(candidate: BlankRegionCandidate, fallback: str) -> str:
    """Return one concise sentence from the visual-generation context."""
    source = next((item for item in candidate.key_signals if item and item != "(empty)"), "")
    if not source:
        source = candidate.local_context or candidate.section_title or fallback
    source = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", source)).strip()
    source = re.split(r"(?<=[.!?])\s+", source, maxsplit=1)[0].rstrip(".!?")
    source = re.sub(r"^(?:shows|illustrates|depicts|compares|presents|visualizes)\s+", "", source, flags=re.I)
    if not source:
        source = fallback
    return source.rstrip(".!?") + "."


def _clean_llm_figure_title(value: str, fallback: str) -> str:
    text = re.sub(r"```(?:text|plain)?", "", value or "", flags=re.I)
    text = re.sub(r"^(?:title|caption)\s*:\s*", "", text.strip(), flags=re.I)
    quoted = re.search(r"[\"']([^\"']+)[\"']", text)
    if quoted:
        text = quoted.group(1)
    text = text.strip().strip('"\'')
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)
    if not words:
        return fallback
    return " ".join(words[:20]).rstrip(".!?") + "."


def _llm_supplement_title(llm: Optional[LLMClient], candidate: BlankRegionCandidate, fallback: str) -> str:
    if llm is None:
        return fallback
    source = "\n".join(candidate.key_signals) or candidate.local_context or candidate.section_title
    try:
        result = llm.chat(
            system="You write short scientific figure titles.",
            user=(
                "Write a concise English title for this generated scientific SVG. "
                "Use at most 20 words. Return title text only, with no label, quotes, or explanation.\n\n"
                f"SVG generation content:\n{source}"
            ),
        )
        return _clean_llm_figure_title(result, fallback)
    except (LLMError, OSError) as exc:
        logger.warning("SVG title generation skipped: %s", exc)
        return fallback


def _generate_blank_supplement_asset(
    llm: Optional[LLMClient],
    candidate: BlankRegionCandidate,
    asset_roots: list[Path],
) -> Optional[str]:
    if not asset_roots:
        return None

    target_name = sanitize_asset_name(f"{candidate.section_id}_supplement", candidate.section_id)
    region = max(
        candidate.blank_regions or candidate.blank_cells or [{}],
        key=lambda item: float(item.get("area_pixels") or item.get("area_ratio") or 0.0),
    )
    target_width = max(1, int(region.get("width") or candidate.width or 360))
    target_height = max(1, int(region.get("height") or candidate.height or 220))
    primary_root = asset_roots[0]
    primary_supplement_dir = primary_root / "supplement"
    primary_browser_dir = primary_root / "figures"
    # Reuse an existing generated asset when the preliminary pass is rerun.
    for root in asset_roots:
        existing = root / "supplement" / f"{target_name}.svg"
        if existing.exists():
            for target_root in asset_roots:
                (target_root / "supplement").mkdir(parents=True, exist_ok=True)
                (target_root / "figures").mkdir(parents=True, exist_ok=True)
                (target_root / "supplement" / existing.name).write_bytes(existing.read_bytes())
                (target_root / "figures" / existing.name).write_bytes(existing.read_bytes())
            return f"figures/{existing.name}"
    svg_text: str | None = None

    if llm is not None:
        try:
            svg_text = _strip_fences(llm.chat(
                system=(
                    "You are a content-driven SVG information designer. The user content and SVG skill "
                    "define the visual; do not reuse a generic template. Create a distinct composition "
                    "that represents the supplied relationships and fits the exact target region. "
                    "Return valid SVG only. No markdown, no code fences, no explanations."
                ),
                user=(
                    _blank_region_visual_prompt(candidate)
                    + "\n\n"
                    + svg_generation_guidance(target_width, target_height)
                ),
            ))
        except Exception as exc:
            logger.warning("Blank-section SVG generation failed for %s: %s", candidate.section_id, exc)
            svg_text = None

    if svg_text:
        svg_text = normalize_svg_dimensions(svg_text, target_width, target_height)
        valid, reason = validate_svg_document(svg_text, target_width, target_height)
        if not valid:
            logger.warning("Rejected generated SVG for %s: %s", candidate.section_id, reason)
            svg_text = None
    if not svg_text:
        svg_text = _fallback_supplement_svg(candidate)
    svg_text = _size_supplement_svg(svg_text, candidate)

    try:
        primary_svg = save_svg_asset(svg_text, primary_supplement_dir, target_name)
    except Exception as exc:
        logger.warning("Failed to persist blank-section SVG for %s: %s", candidate.section_id, exc)
        return None

    try:
        primary_browser_dir.mkdir(parents=True, exist_ok=True)
        browser_path = primary_browser_dir / primary_svg.name
        browser_path.write_text(primary_svg.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to stage SVG asset %s into %s: %s", primary_svg.name, primary_browser_dir, exc)

    for extra_root in asset_roots[1:]:
        try:
            supplement_dir = extra_root / "supplement"
            supplement_dir.mkdir(parents=True, exist_ok=True)
            extra_path = supplement_dir / primary_svg.name
            extra_path.write_text(primary_svg.read_text(encoding="utf-8"), encoding="utf-8")
            browser_dir = extra_root / "figures"
            browser_dir.mkdir(parents=True, exist_ok=True)
            browser_path = browser_dir / primary_svg.name
            browser_path.write_text(primary_svg.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to stage SVG asset %s into %s: %s", primary_svg.name, extra_root, exc)

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
                    const highlights = document.querySelector('#sec-highlights .section-content');
                    const highlightsRect = highlights ? highlights.getBoundingClientRect() : null;
                    const highlightsStyle = highlights ? getComputedStyle(highlights) : null;
                    const rect = poster ? poster.getBoundingClientRect() : null;
                    return {
                        section_count: sections.length,
                        clipped_sections: clipped,
                        empty_sections: empty,
                        broken_images: brokenImages,
                        min_body_font_px: fontSizes.length ? Math.min(...fontSizes) : 0,
                        canvas_width: rect ? Math.round(rect.width) : 0,
                        canvas_height: rect ? Math.round(rect.height) : 0,
                        highlights_region_width: highlightsRect && highlightsStyle
                            ? Math.max(1, Math.round(highlightsRect.width - px(highlightsStyle.paddingLeft) - px(highlightsStyle.paddingRight)))
                            : 0,
                        highlights_region_height: highlightsRect && highlightsStyle
                            ? Math.max(1, Math.round(highlightsRect.height - px(highlightsStyle.paddingTop) - px(highlightsStyle.paddingBottom)))
                            : 0,
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
        if not _should_supplement_report(report, sec):
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
        doc: Optional[PaperDocument] = None,
        analysis: Optional[PaperAnalysis] = None,
) -> Optional[PosterReview]:
    """Capture section crops and record deterministic blank-region findings."""
    png_path = round_dir / "poster.png"
    sections_dir = round_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    selectors = _section_selectors(blueprint)
    crops = capture_poster_full_and_sections(
        html_path,
        png_path,
        selectors,
        width=blueprint.width_px,
        height=blueprint.height_px,
    )
    if not png_path.exists():
        logger.warning("Poster PNG capture failed; supplement pass unavailable")
        return None

    for sec_id, crop in crops.items():
        if not crop.exists():
            continue
        target = sections_dir / f"{sec_id}.png"
        try:
            if crop.resolve() != target.resolve():
                shutil.copy2(crop, target)
        except Exception as exc:
            logger.warning("Failed to stage section crop %s: %s", sec_id, exc)

    review = PosterReview(
        quality_score=10,
        needs_improvement=False,
        issues=[],
        summary="Preliminary Supplement completed deterministic blank-region analysis.",
    )
    blank_reports = _section_blank_reports(round_dir, blueprint)
    if blank_reports:
        review.deterministic_checks["section_blank_reports"] = [asdict(r) for r in blank_reports]
        if doc is not None and analysis is not None:
            review.deterministic_checks["blank_region_candidates"] = [
                asdict(c) for c in _make_blank_region_candidates(
                    blueprint, doc, analysis, round_dir
                )
            ]
    return review


def _refresh_round_artifacts(
        html_path: Path,
        round_dir: Path,
        blueprint: PosterBlueprint,
) -> None:
    """Re-capture the poster after feedback has been applied.

    The harness first captures a pre-feedback review image, then mutates the
    blueprint and re-renders HTML. Without a second capture pass, the saved
    section crops stay stale even though the HTML now includes the supplement.
    """
    png_path = round_dir / "poster.png"
    sections_dir = round_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    selectors = _section_selectors(blueprint)
    crops = capture_poster_full_and_sections(
        html_path,
        png_path,
        selectors,
        width=blueprint.width_px,
        height=blueprint.height_px,
    )
    for sec_id, crop in crops.items():
        if not crop.exists():
            continue
        target = sections_dir / f"{sec_id}.png"
        try:
            if crop.resolve() != target.resolve():
                shutil.copy2(crop, target)
        except Exception as exc:
            logger.warning("Failed to refresh section crop %s: %s", sec_id, exc)


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


def _highlight_candidates() -> list[str]:
    text = _highlight_source()
    candidates = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if line:
            candidates.append(line)
    return candidates


def _highlight_source() -> str:
    try:
        return _HIGHLIGHTS_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_four_highlights(raw: str, candidates: list[str]) -> list[str]:
    selected: list[str] = []
    try:
        payload = json.loads(_strip_fences(raw))
        values = payload.get("highlights") if isinstance(payload, dict) else payload
        if isinstance(values, list):
            selected = [str(value).strip() for value in values if str(value).strip()]
    except (json.JSONDecodeError, TypeError, ValueError):
        selected = []
    candidate_set = set(candidates)
    selected = [item for item in selected if item in candidate_set]
    if not selected:
        selected = candidates[:4]
    selected = list(dict.fromkeys(selected))
    return (selected + [item for item in candidates if item not in selected])[:4]


def _generate_highlights(
    llm: Optional[LLMClient],
    doc: PaperDocument,
    analysis: PaperAnalysis,
    blueprint: PosterBlueprint,
    html_path: Path,
    output_roots: list[Path],
) -> bool:
    """Choose four source highlights and render them as a self-contained SVG."""
    sec = next((item for item in blueprint.sections if item.type == "highlights"), None)
    if sec is None:
        return False
    candidates = _highlight_candidates()
    source_text = _highlight_source()
    if not candidates:
        return False

    size = measure_section_content_size(
        html_path,
        "#sec-highlights .section-content",
        width=blueprint.width_px,
        height=blueprint.height_px,
    )
    width = int(size.get("width") or 0)
    height = int(size.get("height") or 0)
    sec.highlights_region_width = width
    sec.highlights_region_height = height
    if width <= 0 or height <= 0:
        logger.warning("Highlights region has no measurable content size")
        sec.highlights_items = candidates[:4]
        sec.highlights_svg_ref = ""
        return False

    selected = candidates[:4]
    if llm is not None:
        try:
            selection = llm.chat(
                system="You select concise, paper-grounded highlights for a scientific poster.",
                user=(
                    "Read the paper core and the candidate highlights below. Select exactly four "
                    "candidate lines that best represent the paper's actual core. Preserve the "
                    "selected wording exactly. Return JSON only: {\"highlights\": [\"...\", \"...\", "
                    "\"...\", \"...\"]}.\n\n"
                    f"Paper core:\n{_global_poster_context(doc, analysis, blueprint)}\n\n"
                    f"Full highlights.md source:\n{source_text}\n\n"
                    f"Candidate highlights parsed from highlights.md:\n" + "\n".join(
                        f"{index}. {value}" for index, value in enumerate(candidates, 1)
                    )
                ),
            )
            selected = _parse_four_highlights(selection, candidates)
        except Exception as exc:
            logger.warning("Highlight selection failed: %s", exc)
    selected = (selected + candidates)[:4]
    sec.highlights_items = selected
    sec.highlights_svg_ref = ""

    if llm is None:
        return False
    try:
        svg_text = _strip_fences(llm.chat(
            system=(
                "You are a scientific SVG information designer. Return only one complete, valid, "
                "self-contained SVG document. Visualize the supplied four paper highlights clearly."
            ),
            user=(
                f"Paper core:\n{_global_poster_context(doc, analysis, blueprint)}\n\n"
                f"Full highlights.md source:\n{source_text}\n\n"
                f"Selected four highlights:\n" + "\n".join(f"- {item}" for item in selected)
                + "\n\n"
                + svg_generation_guidance(width, height)
            ),
        ))
        svg_text = normalize_svg_dimensions(svg_text, width, height)
        valid, reason = validate_svg_document(svg_text, width, height)
        if not valid:
            logger.warning("Rejected generated highlights SVG: %s", reason)
            return False
        name = sanitize_asset_name(f"{sec.section_id}_highlights", "highlights") + ".svg"
        for root in output_roots:
            figures = Path(root) / "figures"
            figures.mkdir(parents=True, exist_ok=True)
            (figures / name).write_text(svg_text, encoding="utf-8")
        sec.highlights_svg_ref = f"figures/{name}"
        return True
    except Exception as exc:
        logger.warning("Highlights SVG generation failed: %s", exc)
        return False


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
        asset_roots: Optional[list[Path]] = None,
        doc: Optional[PaperDocument] = None,
        analysis: Optional[PaperAnalysis] = None,
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
                if sec.type in {"contributions", "highlights", "project_link"}:
                    sec.supplement_html = ""
                    applied.append(f"supplement skipped ({sec.section_id}: no generated visual)")
                    continue
                candidate_map: dict[str, BlankRegionCandidate] = {}
                for item in (review.deterministic_checks or {}).get("blank_region_candidates") or []:
                    if isinstance(item, dict) and item.get("section_id"):
                        try:
                            candidate_map[str(item["section_id"])] = BlankRegionCandidate(**item)
                        except Exception:
                            continue
                if sec.section_id in candidate_map:
                    candidate = candidate_map[sec.section_id]
                else:
                    blank_reports = (review.deterministic_checks or {}).get("section_blank_reports") or []
                    blank_ratio = 0.0
                    blank_cells: list[dict[str, Any]] = []
                    blank_regions: list[dict[str, Any]] = []
                    crop_path = ""
                    width = 0
                    height = 0
                    text_words = _section_words(sec)
                    figure_count = _section_figure_count(sec.section_id, blueprint)
                    has_figures = _section_has_figures(sec.section_id, blueprint)
                    for item in blank_reports:
                        if isinstance(item, dict) and str(item.get("section_id", "")) == sec.section_id:
                            try:
                                blank_ratio = float(item.get("blank_ratio") or 0.0)
                                width = int(item.get("width") or 0)
                                height = int(item.get("height") or 0)
                                text_words = int(item.get("text_words") or text_words)
                                figure_count = int(item.get("figure_count") or figure_count)
                                has_figures = bool(item.get("has_figures", has_figures))
                                blank_cells = list(item.get("blank_cells") or [])
                                blank_regions = list(item.get("blank_regions") or blank_cells)
                                crop_path = str(item.get("crop_path") or crop_path)
                            except (TypeError, ValueError):
                                blank_ratio = 0.0
                            break
                    candidate = BlankRegionCandidate(
                        section_id=sec.section_id,
                        section_type=sec.type,
                        section_title=sec.title or sec.type,
                        blank_ratio=blank_ratio,
                        content_ratio=max(0.0, 1.0 - blank_ratio),
                        width=width,
                        height=height,
                        text_words=text_words,
                        figure_count=figure_count,
                        has_figures=has_figures,
                        local_context=_section_content_preview(sec, limit=650),
                        nearby_context=_section_neighbor_text(blueprint, sec, radius=1),
                        global_context=(_global_poster_context(doc, analysis, blueprint) if doc and analysis else _fallback_poster_context(blueprint)),
                        key_signals=[],
                        blank_cells=blank_cells,
                        blank_regions=blank_regions,
                        crop_path=crop_path,
                    )
                asset_ref = _generate_blank_supplement_asset(llm, candidate, asset_roots or [])
                if asset_ref:
                    fallback_title = _supplement_description(candidate, sec.title or sec.section_id)
                    supplement_title = _llm_supplement_title(llm, candidate, fallback_title)
                    sec.supplement_html = _supplement_overlay_html(
                        asset_ref,
                        candidate,
                        sec.title or sec.section_id,
                    )
                    sec.supplement_html = sec.supplement_html.replace(
                        html.escape(fallback_title), html.escape(supplement_title), 1
                    )
                    applied.append(f"supplement {sec.section_id} (svg)")
                else:
                    sec.supplement_html = ""
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


def _legacy_visual_harness(
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

        # Measure the actual content box, then replace the legacy Highlights
        # markers with a paper-grounded SVG (or four textual fallbacks).
        if round_no == 1:
            generated_svg = _generate_highlights(
                llm, doc, analysis, blueprint, round_html, [round_dir, output_dir]
            )
            if generated_svg or next(
                (sec for sec in blueprint.sections if sec.type == "highlights"),
                None,
            ) is not None:
                try:
                    html_str = renderer.render(blueprint, doc, round_dir)
                    html_str = _apply_css_patches(html_str, css_patches)
                    round_html.write_text(html_str, encoding="utf-8")
                except Exception as exc:
                    logger.warning("Highlights rerender failed: %s", exc)

        # 2) Capture + visual review.
        review = review_rendered_poster(
            round_html,
            round_dir,
            config,
            blueprint,
            model=vision_model,
            doc=doc,
            analysis=analysis,
        )
        if review is None:
            # One retry, then declare vision unavailable.
            import time as _time
            _time.sleep(2)
            review = review_rendered_poster(
                round_html,
                round_dir,
                config,
                blueprint,
                model=vision_model,
                doc=doc,
                analysis=analysis,
            )
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
            asset_roots=[round_dir, output_dir],
            doc=doc,
            analysis=analysis,
        )
        if applied_actions:
            try:
                html_str = renderer.render(blueprint, doc, round_dir)
                html_str = _apply_css_patches(html_str, css_patches)
                round_html.write_text(html_str, encoding="utf-8")
                _refresh_round_artifacts(round_html, round_dir, blueprint)
            except Exception as exc:
                logger.warning("Post-feedback rerender failed at round %d: %s", round_no, exc)
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


def run_poster_harness(
    doc: PaperDocument,
    analysis: PaperAnalysis,
    blueprint: PosterBlueprint,
    html_path: Path | str,
    output_dir: Path | str,
    config: Optional[HarnessConfig] = None,
    on_round: Optional[object] = None,
    fallback_optimizer: Optional[object] = None,
) -> HarnessResult:
    """Run the one-pass Preliminary Supplement flow.

    The pass uses local screenshot/pixel analysis only. LLM calls are limited to
    the existing highlights and blank-region SVG/title generation helpers.
    """
    del config, fallback_optimizer
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    round_dir = output_dir / "harness" / "round_1"
    round_dir.mkdir(parents=True, exist_ok=True)
    draft = Path(html_path)
    renderer = HtmlPosterRenderer()
    llm = LLMClient() if LLMClient.is_configured() else None
    round_html = round_dir / "poster.html"

    try:
        round_html.write_text(renderer.render(blueprint, doc, round_dir), encoding="utf-8")
        _generate_highlights(llm, doc, analysis, blueprint, round_html, [round_dir, output_dir])
        round_html.write_text(renderer.render(blueprint, doc, round_dir), encoding="utf-8")

        review = review_rendered_poster(
            round_html, round_dir, HarnessConfig(), blueprint, doc=doc, analysis=analysis
        )
        if review is None:
            raise RuntimeError("Unable to capture poster sections for blank-region analysis")

        candidates = [
            BlankRegionCandidate(**item)
            for item in review.deterministic_checks.get("blank_region_candidates", [])
            if isinstance(item, dict)
        ]
        actions: list[str] = []
        for candidate in candidates:
            sec = _section_lookup(blueprint).get(candidate.section_id)
            if sec is None:
                continue
            asset_ref = _generate_blank_supplement_asset(
                llm, candidate, [round_dir, output_dir]
            )
            if not asset_ref:
                continue
            fallback_title = _supplement_description(candidate, sec.title or sec.section_id)
            title = _llm_supplement_title(llm, candidate, fallback_title)
            sec.supplement_html = _supplement_overlay_html(
                asset_ref, candidate, sec.title or sec.section_id
            ).replace(html.escape(fallback_title), html.escape(title), 1)
            actions.append(f"supplement {sec.section_id} (svg)")

        final_html = output_dir / "poster_final.html"
        final_png = output_dir / "poster_final.png"
        final_html.write_text(renderer.render(blueprint, doc, output_dir), encoding="utf-8")
        _refresh_round_artifacts(final_html, output_dir / "harness" / "round_1", blueprint)
        captured = output_dir / "harness" / "round_1" / "poster.png"
        if captured.exists():
            final_png.write_bytes(captured.read_bytes())

        round_record = HarnessRound(
            round_no=1,
            quality_score=10,
            total_score=100.0,
            needs_improvement=False,
            deterministic_checks=review.deterministic_checks,
            summary="Preliminary Supplement completed.",
            applied_actions=actions,
            png_path=str(captured) if captured.exists() else "",
            html_path=str(final_html),
            review_path="",
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        result = HarnessResult(
            passed=True,
            stop_reason="supplement_complete",
            rounds=[round_record],
            best_round_no=1,
            best_score=10,
            final_html=str(final_html),
            final_png=str(final_png) if final_png.exists() else "",
            total_rounds=1,
        )
        _write_report(output_dir, result)
        if on_round:
            on_round(1, 1, 10, False, result.rounds[0].summary)
        return result
    except Exception as exc:
        logger.exception("Preliminary Supplement failed: %s", exc)
        result = HarnessResult(
            passed=False,
            stop_reason="supplement_error",
            final_html=str(draft),
            fallback=False,
            fallback_reason=str(exc),
        )
        _write_report(output_dir, result)
        return result
