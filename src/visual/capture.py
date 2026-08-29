"""Chromium probing and true-size poster capture.

The harness must never ask a VLM to judge a poster it cannot prove was rendered.
This module:

- probes whether a headless Chromium can actually launch (cached),
- captures the rendered poster at its true canvas size with ``device_scale_factor``
  so the full-resolution image matches the print dimensions (e.g. 48x36 in),
- additionally captures every section crop, every figure-region crop,
- draws a grid-overlay image showing section boundaries for the VLM,
- writes a before/after side-by-side diff against the previous round,
- persists everything under ``<round_dir>/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

try:  # PIL is a hard dependency of the visual pipeline.
    from PIL import Image, ImageDraw, ImageFont  # noqa: F401
except ImportError:  # pragma: no cover - env dependent
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_BROWSER_PROBE_CACHE: Optional["BrowserStatus"] = None


class BrowserStatus(BaseModel):
    """Result of the Chromium launch probe."""

    available: bool = False
    reason: str = ""
    error: str = ""


def probe_chromium(force: bool = False) -> BrowserStatus:
    """Launch a headless Chromium once and cache the outcome.

    Returns:
        BrowserStatus. ``available=False`` carries a stable ``reason``
        (``playwright_not_installed`` or ``chromium_launch_failed``) plus the
        error detail so callers can surface an actionable message instead of
        silently degrading.
    """
    global _BROWSER_PROBE_CACHE
    if _BROWSER_PROBE_CACHE is not None and not force:
        return _BROWSER_PROBE_CACHE
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - env dependent
        _BROWSER_PROBE_CACHE = BrowserStatus(
            available=False, reason="playwright_not_installed", error=str(exc)
        )
        return _BROWSER_PROBE_CACHE
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        _BROWSER_PROBE_CACHE = BrowserStatus(available=True, reason="ok")
    except Exception as exc:  # pragma: no cover - env dependent
        _BROWSER_PROBE_CACHE = BrowserStatus(
            available=False,
            reason="chromium_launch_failed",
            error=str(exc)[:800],
        )
    return _BROWSER_PROBE_CACHE


def chromium_available(force: bool = False) -> bool:
    return probe_chromium(force).available


class CaptureBundle(BaseModel):
    """Everything captured in one browser session for one harness round."""

    available: bool = False
    reason: str = ""
    full_png: str = ""
    zoom_png: str = ""
    grid_png: str = ""
    diff_png: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    figures: dict[str, str] = Field(default_factory=dict)
    canvas: dict = Field(default_factory=dict)
    error: str = ""


def _as_file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def measure_section_content_size(
    html_path: Path,
    selector: str,
    width: int = 1920,
    height: int = 1080,
) -> dict[str, int]:
    """Measure a rendered element's content box in CSS pixels."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.goto(_as_file_uri(html_path), wait_until="load")
            _wait_render(page)
            result = page.evaluate(
                """(selector) => {
                    const el = document.querySelector(selector);
                    if (!el) return null;
                    const r = el.getBoundingClientRect();
                    const style = getComputedStyle(el);
                    const px = value => Number.parseFloat(value || '0') || 0;
                    return {
                        width: Math.max(1, Math.round(r.width - px(style.paddingLeft) - px(style.paddingRight))),
                        height: Math.max(1, Math.round(r.height - px(style.paddingTop) - px(style.paddingBottom)))
                    };
                }""",
                selector,
            )
            browser.close()
            return result or {}
    except Exception as exc:
        logger.warning("Section content measurement failed for %s: %s", selector, exc)
        return {}


def _wait_render(page, wait_ms: int = 2500) -> None:
    """Wait for fonts and (best-effort) MathJax before screenshotting."""
    page.wait_for_timeout(wait_ms)
    try:
        page.evaluate("document.fonts && document.fonts.ready")
    except Exception:
        pass
    try:
        page.evaluate(
            """() => {
                if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
                    return window.MathJax.startup.promise.then(() => true);
                }
                return true;
            }"""
        )
    except Exception:
        pass


def capture_poster_bundle(
    html_path: Path,
    out_dir: Path,
    width_px: int = 1920,
    height_px: int = 1440,
    section_selectors: Optional[dict[str, str]] = None,
    figure_selectors: Optional[list[str]] = None,
    device_scale_factor: int = 2,
    max_zoom_width: int = 2048,
    prev_full_png: Optional[Path] = None,
) -> CaptureBundle:
    """Capture the full poster plus crops in a single browser session.

    Args:
        html_path: rendered poster HTML file.
        out_dir: round directory; artifacts are written under ``out_dir``
            (``poster_full.png``, ``poster_zoom.png``, ``grid.png``,
            ``diff_vs_prev.png``, ``sections/<name>.png``, ``figures/<i>.png``).
        width_px / height_px: poster canvas in CSS pixels (print size at 40 dpi).
        section_selectors: name -> CSS selector; every one is screenshotted.
        figure_selectors: CSS selectors resolved to individual images; each
            matching element is screenshotted as ``figures/<index>.png``.
        device_scale_factor: 2 => the full PNG has twice the CSS resolution
            (e.g. 1920x1440 -> 3840x2880) so VLM crops stay legible.
        max_zoom_width: the VLM-facing downscaled copy's max width.
        prev_full_png: previous round's full PNG for a side-by-side diff.

    Returns:
        CaptureBundle with ``available=False`` and a stable ``reason`` when the
        browser is missing or capture failed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = probe_chromium()
    if not status.available:
        return CaptureBundle(
            available=False,
            reason=status.reason,
            error=status.error,
            canvas={"width": width_px, "height": height_px},
        )

    section_selectors = section_selectors or {}
    figure_selectors = figure_selectors or []
    full_png = out_dir / "poster_full.png"
    zoom_png = out_dir / "poster_zoom.png"
    grid_png = out_dir / "grid.png"
    diff_png = out_dir / "diff_vs_prev.png"
    sections_dir = out_dir / "sections"
    figures_dir = out_dir / "figures"
    sections_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    bundle = CaptureBundle(
        available=True,
        reason="ok",
        canvas={"width": width_px, "height": height_px, "device_scale_factor": device_scale_factor},
    )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        bundle.available = False
        bundle.reason = "playwright_not_installed"
        bundle.error = str(exc)
        return bundle

    bboxes: dict[str, list[float]] = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": width_px, "height": height_px},
                device_scale_factor=device_scale_factor,
            )
            page.goto(_as_file_uri(html_path), wait_until="load")
            _wait_render(page)

            # Measure the poster canvas as actually laid out.
            canvas = page.evaluate(
                """() => {
                    const el = document.querySelector('.poster-container');
                    if (!el) return {width: 0, height: 0};
                    const r = el.getBoundingClientRect();
                    return {width: Math.round(r.width), height: Math.round(r.height),
                            x: Math.round(r.x), y: Math.round(r.y)};
                }"""
            )
            bundle.canvas.update(canvas or {})

            # 1) True-size full poster.
            poster = page.locator(".poster-container")
            if poster.count() > 0:
                poster.first.screenshot(path=str(full_png))
            else:
                page.screenshot(path=str(full_png), full_page=True)

            # 2) Section crops + bounding boxes for the grid overlay.
            for name, selector in section_selectors.items():
                locator = page.locator(selector)
                if locator.count() == 0:
                    logger.warning("Section selector not found: %s", selector)
                    continue
                crop = sections_dir / f"{name}.png"
                locator.first.screenshot(path=str(crop))
                bundle.sections[name] = str(crop)
                try:
                    box = locator.first.bounding_box()
                    if box:
                        bboxes[name] = [box["x"], box["y"], box["width"], box["height"]]
                except Exception:
                    pass

            # 3) Figure-region crops (every matched image element).
            fig_index = 0
            for selector in figure_selectors:
                locator = page.locator(selector)
                count = locator.count()
                for i in range(count):
                    crop = figures_dir / f"fig_{fig_index:02d}.png"
                    try:
                        locator.nth(i).screenshot(path=str(crop))
                        bundle.figures[f"fig_{fig_index:02d}"] = str(crop)
                        fig_index += 1
                    except Exception as exc:
                        logger.warning("Figure crop %d failed: %s", i, exc)
            browser.close()
    except Exception as exc:  # pragma: no cover - browser dependent
        logger.warning("Poster capture failed: %s", exc)
        bundle.available = False
        bundle.reason = "capture_failed"
        bundle.error = str(exc)[:800]
        return bundle

    if not full_png.exists():
        bundle.available = False
        bundle.reason = "capture_empty"
        return bundle

    # 4) VLM-facing downscaled copy.
    zoom = _downscale(full_png, zoom_png, max_zoom_width)
    if zoom:
        bundle.zoom_png = str(zoom)

    # 5) Grid overlay (section boundaries on a canvas-size copy).
    overlay = _draw_grid_overlay(full_png, grid_png, bboxes, width_px, height_px)
    if overlay:
        bundle.grid_png = str(overlay)

    # 6) Before/after diff against the previous round.
    if prev_full_png and Path(prev_full_png).exists():
        diff = _side_by_side(Path(prev_full_png), full_png, diff_png)
        if diff:
            bundle.diff_png = str(diff)

    bundle.full_png = str(full_png)
    logger.info(
        "Captured poster bundle: %d sections, %d figures (full=%s)",
        len(bundle.sections), len(bundle.figures), full_png.name,
    )
    return bundle


def _downscale(src: Path, dst: Path, max_width: int) -> Optional[Path]:
    if Image is None:
        return None
    try:
        with Image.open(src) as img:
            width, height = img.size
            if width <= max_width:
                return src
            ratio = max_width / float(width)
            img = img.resize((max_width, int(height * ratio)), Image.LANCZOS)
            img.save(dst, "PNG")
        return dst
    except Exception as exc:
        logger.warning("Downscale failed for %s: %s", src, exc)
        return None


def _draw_grid_overlay(
    full_png: Path,
    dst: Path,
    bboxes: dict[str, list[float]],
    canvas_width: int,
    canvas_height: int,
) -> Optional[Path]:
    """Draw section boundaries over a canvas-size copy of the poster."""
    if Image is None or ImageDraw is None:
        return None
    if not bboxes:
        return None
    try:
        with Image.open(full_png) as img:
            scale_x = canvas_width / float(img.size[0]) if img.size[0] else 1.0
            scale_y = canvas_height / float(img.size[1]) if img.size[1] else 1.0
            base = img.resize((canvas_width, canvas_height), Image.LANCZOS).convert("RGB")
    except Exception as exc:
        logger.warning("Grid overlay base failed: %s", exc)
        return None

    draw = ImageDraw.Draw(base)
    palette = [
        (214, 83, 0), (0, 121, 191), (0, 150, 90), (153, 0, 153),
        (230, 126, 0), (66, 133, 244), (219, 68, 55), (52, 168, 83),
    ]
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for idx, (name, box) in enumerate(sorted(bboxes.items())):
        x, y, w, h = box
        color = palette[idx % len(palette)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x + 6, y + 4), name, fill=color, font=font)
    try:
        base.save(dst, "PNG")
        return dst
    except Exception as exc:
        logger.warning("Grid overlay save failed: %s", exc)
        return None


def _side_by_side(prev: Path, current: Path, dst: Path, total_width: int = 1800) -> Optional[Path]:
    if Image is None:
        return None
    try:
        with Image.open(prev) as a, Image.open(current) as b:
            a = a.convert("RGB")
            b = b.convert("RGB")
            half = total_width // 2
            a = _fit_width(a, half)
            b = _fit_width(b, half)
            height = max(a.height, b.height)
            canvas = Image.new("RGB", (total_width + 8, height), "white")
            canvas.paste(a, (0, 0))
            canvas.paste(b, (half + 8, 0))
            canvas.save(dst, "PNG")
        return dst
    except Exception as exc:
        logger.warning("Diff image failed: %s", exc)
        return None


def _fit_width(img, target_width: int):
    if img.width <= target_width:
        return img
    ratio = target_width / float(img.width)
    return img.resize((target_width, int(img.height * ratio)), Image.LANCZOS)
