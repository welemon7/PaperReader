"""Per-panel: text-block box vs actual content height (absolute layout)."""
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
        const blocks = [...content.querySelectorAll('.text-block')];
        const info = blocks.map(b => {
            const style = getComputedStyle(b);
            const est = parseFloat(style.height);
            // natural height: measure scrollHeight of inner content
            const inner = b.firstElementChild || b;
            return {boxH: Math.round(b.getBoundingClientRect().height),
                    scrollH: b.scrollHeight,
                    font: style.fontSize,
                    overflow: style.overflow};
        });
        out.push({section: sec.id, blocks: info, contentClientH: content.clientHeight, contentScrollH: content.scrollHeight});
    });
    return out;
}
"""

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1920, "height": 1440})
    page.goto(HTML.as_uri(), wait_until="load")
    page.wait_for_timeout(2000)
    print(json.dumps(page.evaluate(JS), indent=1))
    b.close()
