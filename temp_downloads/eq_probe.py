"""Measure per-child heights inside the method/motivation text blocks."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright

HTML = Path(sys.argv[1]).resolve()

JS = r"""
() => {
    const out = [];
    ['sec-motivation', 'sec-method-overview'].forEach(secId => {
        const block = document.querySelector('#' + secId + ' .text-block');
        if (!block) return;
        const kids = [...block.children].map(ch => ({
            tag: ch.tagName,
            cls: ch.className || '',
            h: Math.round(ch.getBoundingClientRect().height),
        }));
        out.push({section: secId, blockH: Math.round(block.getBoundingClientRect().height), kids});
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
    print(json.dumps(page.evaluate(JS), indent=1))
    b.close()
