"""Compare solver estimates vs actual browser render heights."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.sqlite import PaperDatabase
from src.agents.poster_planner import generate_blueprint, normalize_analysis_for_poster
from src.layout.builder import build_scene
from src.layout.solver import solve_layout
from src.renderers.scene_renderer import SceneRenderer
from playwright.sync_api import sync_playwright

db = PaperDatabase()
doc = db.get_paper_by_arxiv("2412.17630")
analysis = db.get_analysis_by_arxiv("2412.17630")
db.close()
analysis = normalize_analysis_for_poster(analysis)
bp = generate_blueprint(doc, analysis)
out = Path("temp_downloads/est_check")
import shutil
shutil.rmtree(out, ignore_errors=True)
scene = build_scene(bp, doc, analysis, out)
layout = solve_layout(scene)
SceneRenderer().render_to_file(scene, layout, doc, out / "poster.html")

for e in layout.elements_in("sec-main-method"):
    print(f"solver: {e.element_id:28s} {e.kind:6s} y={e.y} h={e.h} bottom={e.y+e.h}")

JS = r"""
() => {
    const sec = document.querySelector('#sec-main-method .section-content');
    const out = [];
    [...sec.children].forEach(ch => {
        const r = ch.getBoundingClientRect();
        out.push({cls: ch.className.split(' ')[0], y: Math.round(r.y - sec.getBoundingClientRect().top), h: Math.round(r.height)});
    });
    return {clientH: sec.clientHeight, scrollH: sec.scrollHeight, children: out};
}
"""
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1920, "height": 1440})
    page.goto((out / "poster.html").resolve().as_uri(), wait_until="load")
    page.wait_for_timeout(2000)
    print("browser:", json.dumps(page.evaluate(JS), indent=1))
    b.close()
