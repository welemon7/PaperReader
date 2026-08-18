"""Inspect formula-box inner structure heights."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright

HTML = Path(sys.argv[1]).resolve()

JS = r"""
() => {
    const box = document.querySelector('#sec-method-overview .formula-box');
    if (!box) return {error: 'no box'};
    const walk = (el, depth) => {
        const r = el.getBoundingClientRect();
        const info = {tag: el.tagName, cls: (el.className||'').toString().slice(0,40),
                      h: Math.round(r.height), text: (el.textContent||'').slice(0,60)};
        if (depth < 3 && el.children.length) {
            info.children = [...el.children].slice(0,6).map(c => walk(c, depth+1));
        }
        return info;
    };
    return walk(box, 0);
}
"""

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1920, "height": 1440})
    page.goto(HTML.as_uri(), wait_until="load")
    page.wait_for_timeout(3000)
    page.evaluate("document.fonts && document.fonts.ready")
    print(json.dumps(page.evaluate(JS), indent=1, ensure_ascii=False))
    b.close()
