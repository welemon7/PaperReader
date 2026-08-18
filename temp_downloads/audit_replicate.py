"""Replicate the audit's isClipped logic and report exact offenders."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright

HTML = Path(sys.argv[1]).resolve()

JS = r"""
() => {
    const poster = document.querySelector('.poster-container');
    const out = [];
    [...document.querySelectorAll('.section-block')].forEach(sec => {
        const content = sec.querySelector('.section-content');
        if (!content) return;
        const clippedSections = [];
        if (getComputedStyle(content).overflow !== 'visible' &&
            (content.scrollHeight > content.clientHeight + 2 ||
             content.scrollWidth > content.clientWidth + 2)) {
            clippedSections.push({why: 'content-scroll', scrollH: content.scrollHeight, clientH: content.clientHeight});
        }
        const isClipped = el => {
            let node = el.parentElement;
            while (node && node !== poster && node !== document.body) {
                const st = getComputedStyle(node);
                if (st.overflow !== 'visible' && st.overflow !== '') {
                    const r = el.getBoundingClientRect();
                    const c = node.getBoundingClientRect();
                    if (r.bottom > c.bottom + 2 || r.right > c.right + 2 ||
                        r.top < c.top - 2 || r.left < c.left - 2) {
                        return {node: (node.className||node.tagName).toString().slice(0,30),
                                el: el.tagName + '.' + (el.className||'').toString().slice(0,20),
                                overBottom: Math.round(r.bottom - c.bottom),
                                overRight: Math.round(r.right - c.right)};
                    }
                }
                node = node.parentElement;
            }
            return null;
        };
        const textEls = [...content.querySelectorAll('p, li, td, th, h1, h2, h3, h4, div, span')]
            .filter(n => (n.textContent || '').replace(/\s+/g, '').length > 0);
        for (const t of textEls) {
            const clip = isClipped(t);
            if (clip) { clippedSections.push({why: 'isClipped', ...clip, text: t.textContent.slice(0,40)}); }
        }
        if (clippedSections.length) out.push({section: sec.id, findings: clippedSections});
    });
    return out;
}
"""

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1920, "height": 1440})
    page.goto(HTML.as_uri(), wait_until="load")
    page.wait_for_timeout(2500)
    page.evaluate("document.fonts && document.fonts.ready")
    print(json.dumps(page.evaluate(JS), indent=1, ensure_ascii=False).encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    b.close()
