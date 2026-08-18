"""Validate the FINAL code with a real browser (no LLM): synthetic analysis."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.sqlite import PaperDatabase
from src.schemas.analysis import PaperAnalysis, Contribution, ExperimentSummary
from src.agents.poster_planner import generate_blueprint, normalize_analysis_for_poster
from src.layout.builder import build_scene
from src.layout.solver import solve_layout
from src.renderers.scene_renderer import SceneRenderer
from src.visual.capture import capture_poster_bundle
from src.visual.audit import run_deterministic_audit

ARXIV = sys.argv[1] if len(sys.argv) > 1 else "2412.17630"
db = PaperDatabase()
doc = db.get_paper_by_arxiv(ARXIV)
db.close()
assert doc is not None, "paper not in DB"

analysis = PaperAnalysis(
    paper_id=doc.paper_id, arxiv_id=doc.arxiv_id, title_zh="",
    problem_statement="We address the ill-posed task of removing undesired shadows while preserving fine image details.",
    contributions=[
        Contribution(text="A detail-preserving latent diffusion framework for shadow removal", category="method"),
        Contribution(text="A global context injection module that stabilizes training", category="method"),
        Contribution(text="Strong quantitative gains over prior shadow removal methods", category="system"),
    ],
    method_overview="We introduce a two-stage architecture that combines latent diffusion with detail injection to remove shadows while keeping texture and color faithful.",
    key_formulas=[], key_figures=[],
    experiments=ExperimentSummary(
        main_results="Outperforms prior methods on PSNR and SSIM across ISTD+, SRD and WSRD+",
        takeaways=["Better shadow removal quality", "Fine detail preservation", "Improved cross-dataset robustness"],
        datasets=["ISTD+", "SRD", "WSRD+"], metrics=["PSNR", "SSIM"],
    ),
    conclusion="The proposed framework achieves state-of-the-art shadow removal with detail preservation.",
    full_analysis_md="",
)
analysis = normalize_analysis_for_poster(analysis)
blueprint = generate_blueprint(doc, analysis)

out = Path("temp_downloads/final_check")
import shutil
shutil.rmtree(out, ignore_errors=True)
scene = build_scene(blueprint, doc, analysis, out)
layout = solve_layout(scene)
html = SceneRenderer().render(scene, layout, doc, out)
(out / "poster.html").write_text(html, encoding="utf-8")

bundle = capture_poster_bundle(
    out / "poster.html", out, width_px=1920, height_px=1440,
    section_selectors={p.panel_id: f"#{p.panel_id}" for p in scene.panels},
    figure_selectors=[".figure-card img"],
)
print("capture:", bundle.available, "sections:", len(bundle.sections), "figures:", len(bundle.figures))

audit = run_deterministic_audit(out / "poster.html", 1920, 1440)
print("audit available:", audit.available)
for c in audit.checks:
    print(f"  {'PASS' if c.passed else 'FAIL':4s} {c.severity:7s} {c.name:22s} {c.detail[:90]}")
print("hard_failures:", audit.hard_failures)
