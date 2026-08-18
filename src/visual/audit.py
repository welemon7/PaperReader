"""Deterministic browser-geometry audit (no model involved).

Runs a single ``page.evaluate`` that measures the rendered poster: canvas
bounds, sibling overlap, text clipping (including ancestors with
``overflow: hidden``), blank space, image resolution vs display size, minimum
body font size, broken/missing assets and figure-caption rendering.

Every failure is classified ``error`` (hard blocker, cannot be waived by any
VLM) or ``warning`` (quality hint).  The audit must be *available* for the
harness to certify a poster.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.schemas.review import DeterministicAudit, DeterministicCheck
from src.visual.capture import probe_chromium

logger = logging.getLogger(__name__)

# Minimum body font size in CSS px on a 1920px-wide (48in @ 40dpi) canvas.
MIN_BODY_FONT_PX = 13
MIN_TITLE_FONT_PX = 20
# A figure upscaled beyond 2x its native resolution is flagged.
MAX_UPSCALE_RATIO = 2.0
# Overlap IOU threshold for sibling section blocks.
OVERLAP_IOU = 0.02
# Fraction of a section's content area that may be blank before we complain.
BLANK_SPACE_WARNING = 0.55
BLANK_SPACE_ERROR = 0.85

_AUDIT_JS = r"""
() => {
    const poster = document.querySelector('.poster-container');
    const sections = [...document.querySelectorAll('.section-block')];
    const px = v => Number.parseFloat(v || '0') || 0;
    const rect = el => { const r = el.getBoundingClientRect(); return {x:r.x, y:r.y, w:r.width, h:r.height}; };
    const isClipped = el => {
        // Walk ancestors; if any clips (overflow != visible) and the element
        // sticks out of it, the content is cut.
        let node = el.parentElement;
        while (node && node !== poster && node !== document.body) {
            const st = getComputedStyle(node);
            if (st.overflow !== 'visible' && st.overflow !== '') {
                const r = el.getBoundingClientRect();
                const c = node.getBoundingClientRect();
                if (r.bottom > c.bottom + 2 || r.right > c.right + 2 ||
                    r.top < c.top - 2 || r.left < c.left - 2) {
                    return true;
                }
            }
            node = node.parentElement;
        }
        return false;
    };
    const textEls = el => [...el.querySelectorAll('p, li, td, th, h1, h2, h3, h4, div, span')]
        .filter(n => (n.textContent || '').replace(/\s+/g, '').length > 0);
    const fontOf = el => px(getComputedStyle(el).fontSize);

    const result = {
        canvas: poster ? rect(poster) : null,
        canvas_expected: null,
        section_boxes: {},
        overlap: [],
        clipped_sections: [],
        blank_sections: [],
        broken_images: [],
        upscaled_images: [],
        min_body_font: 0,
        min_title_font: 0,
        caption_rendered: [],
        img_count: document.images.length,
        section_count: sections.length,
    };

    sections.forEach(sec => {
        const id = sec.id || '(unnamed)';
        const box = rect(sec);
        result.section_boxes[id] = [box.x, box.y, box.w, box.h];

        // text clipping (own overflow hidden + ancestor clipping)
        const content = sec.querySelector('.section-content');
        if (content) {
            const st = getComputedStyle(content);
            if (st.overflow !== 'visible' &&
                (content.scrollHeight > content.clientHeight + 2 ||
                 content.scrollWidth > content.clientWidth + 2)) {
                result.clipped_sections.push(id);
            }
            for (const t of textEls(content)) {
                if (isClipped(t)) { result.clipped_sections.push(id); break; }
            }
        }

        // blank space: content area vs used area
        if (content) {
            const cRect = content.getBoundingClientRect();
            const kids = [...content.children].filter(c => getComputedStyle(c).display !== 'none');
            let usedH = 0;
            if (kids.length) {
                let top = Infinity, bottom = -Infinity;
                kids.forEach(k => {
                    const r = k.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0) { top = Math.min(top, r.top); bottom = Math.max(bottom, r.bottom); }
                });
                usedH = bottom > top ? (bottom - top) : 0;
            }
            const areaH = Math.max(1, cRect.height);
            if (usedH > 0 && (areaH - usedH) / areaH > 0.85 && (content.innerText || '').replace(/\s+/g,'').length > 0) {
                result.blank_sections.push({id, ratio: +(1 - usedH/areaH).toFixed(2)});
            }
        }

        // fonts
        content && textEls(content).forEach(t => {
            const f = fontOf(t);
            if (f > 0) {
                const isTitle = t.closest('.section-title') !== null;
                if (isTitle) result.min_title_font = result.min_title_font === 0 ? f : Math.min(result.min_title_font, f);
                else if (t.closest('.section-content')) result.min_body_font = result.min_body_font === 0 ? f : Math.min(result.min_body_font, f);
            }
        });
    });

    // sibling overlap (only same row area, coarse: all pairs)
    const keys = Object.keys(result.section_boxes);
    for (let i = 0; i < keys.length; i++) {
        for (let j = i + 1; j < keys.length; j++) {
            const a = result.section_boxes[keys[i]], b = result.section_boxes[keys[j]];
            const ix = Math.max(0, Math.min(a[0]+a[2], b[0]+b[2]) - Math.max(a[0], b[0]));
            const iy = Math.max(0, Math.min(a[1]+a[3], b[1]+b[3]) - Math.max(a[1], b[1]));
            const inter = ix * iy;
            if (inter <= 0) continue;
            const union = a[2]*a[3] + b[2]*b[3] - inter;
            const iou = union > 0 ? inter / union : 0;
            if (iou > 0.02) result.overlap.push({a: keys[i], b: keys[j], iou: +iou.toFixed(3)});
        }
    }

    // images
    [...document.images].forEach(img => {
        if (!img.complete || img.naturalWidth === 0) {
            result.broken_images.push(img.alt || img.src || '(unnamed)');
        } else {
            const r = img.getBoundingClientRect();
            if (r.width > 0) {
                const ratio = img.naturalWidth / r.width;
                if (ratio < 1 / 2) {
                    result.upscaled_images.push({src: img.src || img.alt, ratio: +ratio.toFixed(2)});
                }
            }
        }
    });

    // captions must NOT be rendered under figures (product requirement)
    [...document.querySelectorAll('.figure-caption')].forEach(c => {
        if ((c.textContent || '').trim()) result.caption_rendered.push(c.textContent.trim().slice(0, 60));
    });

    return result;
}
"""


def run_deterministic_audit(
    html_path: Path,
    width_px: int = 1920,
    height_px: int = 1440,
) -> DeterministicAudit:
    """Open the rendered poster in a headless Chromium and measure geometry."""
    status = probe_chromium()
    if not status.available:
        return DeterministicAudit(
            available=False,
            reason=status.reason,
            checks=[DeterministicCheck(
                name="browser_available", passed=False, severity="error",
                detail=status.reason or "chromium unavailable",
            )],
        )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return DeterministicAudit(
            available=False, reason="playwright_not_installed",
            checks=[DeterministicCheck(name="browser_available", passed=False, severity="error", detail=str(exc))],
        )

    audit = DeterministicAudit(available=True, reason="ok")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width_px, "height": height_px})
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.wait_for_timeout(1800)
            page.evaluate("document.fonts && document.fonts.ready")
            data = page.evaluate(_AUDIT_JS)
            browser.close()
    except Exception as exc:
        logger.warning("Deterministic audit unavailable: %s", exc)
        audit.available = False
        audit.reason = "browser_error"
        audit.add(DeterministicCheck(
            name="browser_available", passed=False, severity="error", detail=str(exc)[:400],
        ))
        return audit

    # --- map raw measurements to checks -----------------------------------
    if not data.get("canvas"):
        audit.add(DeterministicCheck(
            name="canvas_present", passed=False, severity="error",
            detail="no .poster-container found",
        ))
    else:
        canvas = data["canvas"]
        expected = {"x": 0, "y": 0, "w": width_px, "h": height_px}
        data["canvas_expected"] = expected
        within = (
            abs(canvas["w"] - expected["w"]) <= 8
            and canvas["h"] >= expected["h"] - 8
        )
        audit.add(DeterministicCheck(
            name="canvas_size",
            passed=within,
            severity="error" if not within else "warning",
            detail=f"canvas {canvas['w']}x{canvas['h']} (expected {expected['w']}x{expected['h']})",
            data={"actual": canvas, "expected": expected},
        ))

    audit.add(DeterministicCheck(
        name="section_count",
        passed=data.get("section_count", 0) >= 4,
        severity="error",
        detail=f"{data.get('section_count', 0)} sections rendered",
        data={"count": data.get("section_count", 0)},
    ))

    clipped = data.get("clipped_sections") or []
    audit.add(DeterministicCheck(
        name="text_clipping",
        passed=len(clipped) == 0,
        severity="error",
        detail=f"clipped: {clipped}" if clipped else "no clipping",
        data={"clipped": clipped},
    ))

    overlap = data.get("overlap") or []
    audit.add(DeterministicCheck(
        name="element_overlap",
        passed=len(overlap) == 0,
        severity="error",
        detail=f"overlap: {overlap}" if overlap else "no overlap",
        data={"overlap": overlap},
    ))

    broken = data.get("broken_images") or []
    audit.add(DeterministicCheck(
        name="broken_images",
        passed=len(broken) == 0,
        severity="error",
        detail=f"broken: {broken}" if broken else "all images loaded",
        data={"broken": broken},
    ))

    upscaled = data.get("upscaled_images") or []
    audit.add(DeterministicCheck(
        name="image_resolution",
        passed=len(upscaled) == 0,
        severity="warning",
        detail=f"upscaled: {upscaled}" if upscaled else "images at native resolution",
        data={"upscaled": upscaled},
    ))

    blank = data.get("blank_sections") or []
    audit.add(DeterministicCheck(
        name="blank_space",
        passed=len(blank) == 0,
        severity="warning",
        detail=f"large blank area: {blank}" if blank else "no excessive blank space",
        data={"blank": blank},
    ))

    min_body = float(data.get("min_body_font") or 0)
    min_title = float(data.get("min_title_font") or 0)
    audit.add(DeterministicCheck(
        name="min_body_font",
        passed=min_body >= MIN_BODY_FONT_PX,
        severity="error",
        detail=f"min body font {min_body:.1f}px (>= {MIN_BODY_FONT_PX}px)",
        data={"min_body_font": min_body, "min_title_font": min_title},
    ))
    audit.add(DeterministicCheck(
        name="min_title_font",
        passed=min_title == 0 or min_title >= MIN_TITLE_FONT_PX,
        severity="warning",
        detail=f"min section-title font {min_title:.1f}px",
        data={"min_title_font": min_title},
    ))

    captions = data.get("caption_rendered") or []
    audit.add(DeterministicCheck(
        name="figure_caption_hidden",
        passed=len(captions) == 0,
        severity="error",
        detail=f"captions rendered: {captions}" if captions else "no figure captions rendered",
        data={"caption_rendered": captions},
    ))

    return audit
