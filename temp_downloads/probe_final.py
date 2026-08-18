"""Probe what clips in a rendered HTML (geometry detail)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright

HTML = Path(sys.argv[1] if len(sys.argv) > 1 else "temp_downloads/final_check/poster.html").resolve()

JS = r"""
() => {
    const out = [];
    [...document.querySelectorAll('.section-block')].forEach(sec => {
        const sbox = sec.getBoundingClientRect();
        const content = sec.querySelector('.section-content');
        if (!content) return;
        [...content.children].forEach(ch => {
            const r = ch.getBoundingClientRect();
            if (r.width === 0 && r.height === 0) return;
            const ob = +(r.bottom - sbox.bottom).toFixed(1);
            if (ob > 2) {
                out.push({
                    section: sec.id, child: (ch.className || ch.tagName),
                    y: Math.round(r.y), h: Math.round(r.height),
                    sectionBottom: Math.round(sbox.bottom), overflowBottom: ob,
                });
            }
        });
        // scroll overflow detail
        if (content.scrollHeight > content.clientHeight + 2) {
            out.push({section: sec.id, kind: 'scroll', scrollH: content.scrollHeight, clientH: content.clientHeight});
        }
    });
    return out;
}
"""

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1440})
    page.goto(HTML.as_uri(), wait_until="load")
    page.wait_for_timeout(2000)
    page.evaluate("document.fonts && document.fonts.ready")
    print(json.dumps(page.evaluate(JS), indent=1))
    browser.close()
