"""Scene builder + solver + renderer invariants (no browser required)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.agents.poster_planner import generate_blueprint
from src.layout.builder import build_scene
from src.layout.scene import CANVAS_HEIGHT, CANVAS_WIDTH, PosterScene
from src.layout.solver import solve_layout
from src.renderers.scene_renderer import SceneRenderer
from src.schemas.analysis import Contribution, ExperimentSummary, PaperAnalysis
from src.schemas.paper import Author, Figure, PaperDocument, Section


def _make_doc(tmp_path: Path) -> PaperDocument:
    (tmp_path / "fig_a.png").write_bytes(b"")
    Image.new("RGB", (1200, 600), (100, 150, 200)).save(tmp_path / "arch.png")
    Image.new("RGB", (800, 800), (200, 150, 100)).save(tmp_path / "result.png")
    return PaperDocument(
        paper_id="scene-test", arxiv_id="9999.99998",
        title="Scene Graph Poster Test Title",
        authors=[Author(name="Alice"), Author(name="Bob")],
        abstract="Abstract.",
        sections=[Section(section_id="s1", title="Intro", level=1, text="x", raw_latex="x")],
        figures=[
            Figure(figure_id="fig-arch", caption="Architecture overview", section_id="s1", local_path="arch.png"),
            Figure(figure_id="fig-res", caption="Results comparison", section_id="s1", local_path="result.png"),
        ],
        raw_markdown="# x",
        source_dir=str(tmp_path),
    )


def _make_analysis() -> PaperAnalysis:
    return PaperAnalysis(
        paper_id="scene-test", arxiv_id="9999.99998", title_zh="",
        problem_statement="We solve a hard problem with a two-stage pipeline that injects detail.",
        contributions=[
            Contribution(text="A better backbone with strong gains", category="method"),
            Contribution(text="A new loss function that stabilizes training", category="method"),
            Contribution(text="SOTA results on benchmarks", category="system"),
        ],
        method_overview="We introduce a two-stage architecture combining global context and detail injection.",
        key_formulas=[], key_figures=[],
        experiments=ExperimentSummary(main_results="99.2 percent accuracy, 2.1x faster", takeaways=["Works well"]),
        conclusion="Done.", full_analysis_md="# A",
    )


def test_build_scene_zones_and_canvas(tmp_path):
    doc, analysis = _make_doc(tmp_path), _make_analysis()
    blueprint = generate_blueprint(doc, analysis)
    scene = build_scene(blueprint, doc, analysis, tmp_path / "out")

    assert isinstance(scene, PosterScene)
    assert scene.canvas_width == CANVAS_WIDTH == 1920
    assert scene.canvas_height == CANVAS_HEIGHT == 1440

    zones = {p.zone for p in scene.panels}
    assert zones == {"left", "center", "right_top", "right_bottom", "bottom_left", "bottom_center", "bottom_right"}
    assert scene.panel_by_type("motivation").zone == "left"
    assert scene.panel_by_type("method_overview").zone == "center"
    assert scene.panel_by_type("main_method").zone == "right_bottom"
    assert scene.panel_by_type("project_link").zone == "bottom_right"


def test_build_scene_figures_measure_aspect(tmp_path):
    doc, analysis = _make_doc(tmp_path), _make_analysis()
    blueprint = generate_blueprint(doc, analysis)
    scene = build_scene(blueprint, doc, analysis, tmp_path / "out")

    method = scene.panel_by_type("method_overview")
    figures = [e for e in method.elements if e.kind == "figure"]
    assert figures, "method overview should carry the architecture figure"
    assert figures[0].figure_src, "figure src must be a prepared local asset"
    assert figures[0].figure_aspect == 2.0  # 1200x600

    results = scene.panel_by_type("main_method")
    res_figs = [e for e in results.elements if e.kind == "figure"]
    assert res_figs and res_figs[0].figure_aspect == 1.0  # 800x800


def test_solve_layout_boxes_within_canvas(tmp_path):
    doc, analysis = _make_doc(tmp_path), _make_analysis()
    blueprint = generate_blueprint(doc, analysis)
    scene = build_scene(blueprint, doc, analysis, tmp_path / "out")
    layout = solve_layout(scene)

    assert len(layout.panels) == 7
    for box in layout.panels:
        assert box.x >= 0 and box.y >= 0
        assert box.x + box.w <= CANVAS_WIDTH
        assert box.y + box.h <= CANVAS_HEIGHT

    # 无面板重叠
    for i in range(len(layout.panels)):
        for j in range(i + 1, len(layout.panels)):
            a, b = layout.panels[i], layout.panels[j]
            ox = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
            oy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
            assert ox * oy == 0, f"panels {a.panel_id}/{b.panel_id} overlap"


def test_solve_layout_two_up_figures_equal_height(tmp_path):
    doc, analysis = _make_doc(tmp_path), _make_analysis()
    blueprint = generate_blueprint(doc, analysis)
    scene = build_scene(blueprint, doc, analysis, tmp_path / "out")
    layout = solve_layout(scene)

    results = layout.panel("sec-main-method")
    figs = [e for e in layout.elements if e.element_id.startswith("sec-main-method.") and e.kind == "figure"]
    if len(figs) == 2:
        assert figs[0].h == figs[1].h, "two-up figure row must be aligned"
        assert figs[0].x != figs[1].x


def test_scene_renderer_no_caption_and_absolute_panels(tmp_path):
    doc, analysis = _make_doc(tmp_path), _make_analysis()
    blueprint = generate_blueprint(doc, analysis)
    out = tmp_path / "render"
    scene = build_scene(blueprint, doc, analysis, out)
    layout = solve_layout(scene)
    html = SceneRenderer().render(scene, layout, doc, out)

    assert ".figure-caption" not in html, "figure captions must not be rendered"
    assert 'class="poster-container"' in html
    assert "width: 1920px" in html and "height: 1440px" in html
    for panel_id in ("sec-motivation", "sec-method-overview", "sec-key-idea", "sec-main-method",
                     "sec-contributions", "sec-highlights", "sec-project"):
        assert f'id="{panel_id}"' in html
        assert f"position: absolute" in html or "position:absolute" in html
    # 图的 alt 保留 caption（用于 QA/审查），但页面不显示
    assert "alt=" in html
    # 主题变量注入
    assert "--poster-blue" in html
    # MathJax 保留
    assert "MathJax" in html
