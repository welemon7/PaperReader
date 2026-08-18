"""Dump every child box + inline style per section-content."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright

HTML = Path(sys.argv[1]).resolve()

JS = r"""
() => {
    const out = [];
    [...document.querySelectorAll('.section-block')].forEach(sec => {
        const content = sec.querySelector('.section-content');
        if (!content) return;
        const cbox = content.getBoundingClientRect();
        const children = [...content.children].map(ch => {
            const r = ch.getBoundingClientRect();
            return {
                cls: ch.className.split(' ')[0],
                relY: Math.round(r.top - cbox.top), relH: Math.round(r.height),
                absBottom: Math.round(r.bottom),
                scrollH: ch.scrollHeight,
                inline: (ch.getAttribute('style') || '').slice(0, 90),
            };
        });
        out.push({
            section: sec.id,
            contentTop: Math.round(cbox.top),
            contentBottom: Math.round(cbox.bottom),
            clientH: content.clientHeight, scrollH: content.scrollHeight,
            children,
        });
    });
    return out;
}
"""

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1920, "height": 1440})
    page.goto(HTML.as_uri(), wait_until="load")
    page.wait_for_timeout(2000)
    result = page.evaluate(JS)
    for sec in result:
        if sec["scrollH"] > sec["clientH"] + 2 or any(c["absBottom"] > sec["contentBottom"] + 2 for c in sec["children"]):
            print(json.dumps(sec, indent=1, ensure_ascii=False))
    b.close()
