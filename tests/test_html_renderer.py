from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.schemas.poster import PosterBlueprint, PosterSection
from src.schemas.paper import PaperDocument
from src.schemas.paper import Figure
from src.renderers.html_renderer import HtmlPosterRenderer
from src.utils.figure_assets import copy_or_rasterize_asset


def _make_blueprint() -> PosterBlueprint:
    return PosterBlueprint(
        paper_id="test-999", poster_title="Test Poster",
        authors_str="Alice, Bob",
        code_url="",
        sections=[
            PosterSection(section_id="s1", type="motivation", title="Motivation",
                        content_md="Our problem is **important**.", column=1, col_span=1, row=1),
            PosterSection(section_id="s2", type="method_overview", title="Method Overview",
                        content_md="We propose a method with $E=mc^2$.", column=2, col_span=1, row=1),
            PosterSection(section_id="s3", type="key_idea", title="Key Idea: Core Trick",
                        content_md="Details here.", column=3, col_span=1, row=1),
            PosterSection(section_id="s4", type="main_method", title="Core",
                        content_md="Results and details here.", column=1, col_span=3, row=2),
            PosterSection(section_id="s5", type="contributions", title="Contributions",
                        content_md="Bullets.", column=1, col_span=1, row=3),
            PosterSection(section_id="s6", type="highlights", title="Highlights",
                        content_md="Takeaways.", column=2, col_span=1, row=3),
            PosterSection(section_id="s7", type="project_link", title="Project",
                        content_md="Code link.", column=3, col_span=1, row=3),
        ],
        color_scheme={"primary": "#1a5276", "accent": "#2980b9", "background": "#ffffff",
                     "text": "#2c3e50", "section_header_bg": "#1a5276",
                     "section_header_text": "#ffffff", "border": "#d5dbdb", "highlight": "#f39c12"},
    )


class TestHtmlPosterRenderer:
    def test_render_returns_html(self):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999")
        assert "<!DOCTYPE html>" in html
        assert "Test Poster" in html
        assert "Alice, Bob" in html
        assert "Motivation" in html
        assert "Our problem is" in html
        assert "<strong>" in html

    def test_render_shows_code_pill(self):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        bp.code_url = "https://github.com/example/repo"
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999")
        assert "https://github.com/example/repo" in html
        assert html.count("hero-pill") >= 1
        assert "Code & Project" in html

    def test_render_hides_code_pill_when_missing(self):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999")
        assert "https://github.com/example/repo" not in html
        assert 'href="https://github.com/example/repo"' not in html

    def test_render_cleans_reference_fragments(self):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        bp.sections[0].content_md = r"See ~\\ref{tab:ablation} and ~\\cite{wang2018} for details."
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999")
        assert "~\\ref" not in html
        assert "~\\cite" not in html
        assert "wang2018" not in html

    def test_render_converts_textbf_markup(self):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        bp.sections[0].content_md = r"This is \textbf{important} and \emph{clear}."
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999")
        assert r"\textbf" not in html
        assert "<strong>important</strong>" in html
        assert "<em>clear</em>" in html or "<i>clear</i>" in html

    def test_render_preserves_highlight_spans(self):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        bp.sections[2].content_md = 'Details about the <span class="poster-highlight">novel approach</span> and <span class="poster-highlight-metric">97.3%</span>.'
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999")
        assert '<span class="poster-highlight">novel approach</span>' in html
        assert '<span class="poster-highlight-metric">97.3%</span>' in html

    def test_render_to_file(self, tmp_path):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        out = tmp_path / "poster.html"
        renderer = HtmlPosterRenderer()
        result = renderer.render_to_file(bp, doc, out)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Test Poster" in content

    @patch("src.renderers.html_renderer.LLMClient.is_configured", return_value=False)
    def test_render_skips_llm_when_unconfigured(self, _mock_config):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999", optimize_with_llm=True)
        assert "<!DOCTYPE html>" in html

    @patch("src.renderers.html_renderer.LLMClient.is_configured", return_value=True)
    @patch("src.renderers.html_renderer.LLMClient.chat", return_value="<!DOCTYPE html><html><body>optimized</body></html>")
    def test_render_uses_llm_optimizer(self, _mock_chat, _mock_config, tmp_path):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, tmp_path, optimize_with_llm=True)
        assert "optimized" in html

    def test_organize_rows(self):
        bp = _make_blueprint()
        rows = HtmlPosterRenderer._organize_rows(bp)
        assert len(rows) == 3
        assert [s.section_id for s in rows[0]["sections"]] == ["s1", "s2", "s3"]
        assert [s.section_id for s in rows[1]["sections"]] == ["s4"]
        assert [s.section_id for s in rows[2]["sections"]] == ["s5", "s6", "s7"]

    def test_build_layout_sorts_by_row_then_column(self):
        bp = _make_blueprint()
        layout = HtmlPosterRenderer._build_layout(bp)
        assert [s.section_id for s in layout] == ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]

    def test_render_uses_grid_areas(self):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999")
        assert "grid-area: motivation" in html
        assert "grid-area: overview" in html
        assert "grid-area: key_idea" in html
        assert "grid-area: core" in html
        assert "grid-area: contributions" in html
        assert "grid-area: highlights" in html
        assert "grid-area: project" in html
        assert 'grid-template-areas:' in html
        assert '"motivation overview key_idea"' in html
        assert '"core core core"' in html
        assert '"contributions highlights project"' in html
        assert "grid-area: main_method" not in html
        assert "grid-area: method_overview" not in html

    def test_render_merges_core_band_into_core_results(self, tmp_path):
        source_dir = tmp_path / "paper"
        source_dir.mkdir()
        left = source_dir / "left.png"
        right = source_dir / "right.png"
        left.write_bytes(b"fakepng")
        right.write_bytes(b"fakepng")
        doc = PaperDocument(
            paper_id="test-999",
            arxiv_id="9999.99999",
            title="Test",
            raw_markdown=".",
            source_dir=str(source_dir),
            figures=[
                Figure(figure_id="fig-left", caption="Result figure 1", local_path=str(left), section_id="s4"),
                Figure(figure_id="fig-right", caption="Result figure 2", local_path=str(right), section_id="s2"),
            ],
        )
        bp = _make_blueprint()
        bp.sections = [
            bp.sections[0],
            bp.sections[1],
            bp.sections[2],
            PosterSection(section_id="s4", type="main_method", title="Core Method", content_md="| Dataset | PSNR | SSIM |\n| --- | --- | --- |\n| ISTD+ | 34.1 | 0.92 |", column=1, col_span=3, row=2),
            PosterSection(section_id="s5", type="contributions", title="Contributions", content_md="Bullets.", column=1, col_span=1, row=3),
            PosterSection(section_id="s6", type="highlights", title="Highlights", content_md="Takeaways.", column=2, col_span=1, row=3),
            PosterSection(section_id="s7", type="project_link", title="Project", content_md="Code link.", column=3, col_span=1, row=3),
        ]
        bp.figure_placements = [
            type("FP", (), {"figure_id": "fig-left", "section_id": "s4", "width_ratio": 0.9, "caption": "Result figure 1"})(),
            type("FP", (), {"figure_id": "fig-right", "section_id": "s4", "width_ratio": 0.9, "caption": "Result figure 2"})(),
        ]
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, tmp_path)
        assert "Core Results" in html
        assert "core-grid" in html
        assert "Result figure 1" in html
        assert "ISTD+" in html
        assert "Result figure 2" in html
        assert html.count('class="section-block core-band"') == 1
        assert html.count('style="grid-area: core;"') == 1

    def test_render_uses_item_details_table_for_core_results(self, tmp_path):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        bp.sections[3].content_md = (
            "Core results are strong.\n\n"
            "[[CORE_TABLE]]\n\n"
            "<div class=\"item-details-wrap\">"
            "<div class=\"item-details-title\">Item Details</div>"
            "<table class=\"item-details-table\">"
            "<tbody>"
            "<tr><th>Datasets</th><td>Dataset-A<br>Dataset-B</td></tr>"
            "<tr><th>Metrics</th><td>Accuracy<br>F1</td></tr>"
            "<tr><th>Main Results</th><td>State-of-the-art</td></tr>"
            "<tr><th>Takeaways</th><td>Works well</td></tr>"
            "</tbody></table></div>"
        )
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, tmp_path)
        assert "Item Details" in html
        assert "Datasets" in html
        assert "Metrics" in html
        assert "Main Results" in html
        assert "Takeaways" in html
        assert "Dataset-A" in html
        assert "Accuracy" in html

    def test_render_reflow_content_sections(self, tmp_path):
        source_dir = tmp_path / "paper"
        source_dir.mkdir()
        overview_img = source_dir / "overview.png"
        key_img = source_dir / "detail.png"
        overview_img.write_bytes(b"fakepng")
        key_img.write_bytes(b"fakepng")

        doc = PaperDocument(
            paper_id="test-999",
            arxiv_id="9999.99999",
            title="Test",
            raw_markdown=".",
            source_dir=str(source_dir),
            figures=[
                Figure(figure_id="fig-ov", caption="Network architecture overview", local_path=str(overview_img), section_id="s2"),
                Figure(figure_id="fig-key", caption="Detailed structure", local_path=str(key_img), section_id="s3"),
                Figure(figure_id="fig-left", caption="Result figure 1", local_path=str(overview_img), section_id="s4"),
                Figure(figure_id="fig-right", caption="Result figure 2", local_path=str(key_img), section_id="s4"),
            ],
        )
        bp = _make_blueprint()
        bp.code_url = "https://github.com/example/repo"
        bp.sections[0].content_md = "The current approach is limited. It misses fine details."
        bp.sections[1].content_md = "We introduce a compact pipeline.\n\n$$ y = f(x) $$"
        bp.sections[2].content_md = "The key idea is to refine the output step by step."
        bp.sections[3].content_md = "| Dataset | PSNR | SSIM |\n| --- | --- | --- |\n| ISTD+ | 34.1 | 0.92 |\n| SRD | 33.7 | 0.91 |\n| INS | 32.8 | 0.90 |"
        bp.sections[4].content_md = "- First contribution"
        bp.sections[5].content_md = "- Highlight one"
        bp.sections[6].content_md = "Code link."
        bp.figure_placements = [
            type("FP", (), {"figure_id": "fig-ov", "section_id": "s2", "width_ratio": 0.9, "caption": "Network architecture overview"})(),
            type("FP", (), {"figure_id": "fig-key", "section_id": "s3", "width_ratio": 0.9, "caption": "Detailed structure"})(),
            type("FP", (), {"figure_id": "fig-left", "section_id": "s4", "width_ratio": 0.9, "caption": "Result figure 1"})(),
            type("FP", (), {"figure_id": "fig-right", "section_id": "s4", "width_ratio": 0.9, "caption": "Result figure 2"})(),
        ]
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, tmp_path)
        assert "formula-box" in html
        assert "callout" in html
        assert "Network architecture overview" in html
        assert "Detailed structure" in html
        assert "code-cta" in html
        assert "Paper-to-Poster · Research Reader" in html
        assert html.count("badge-pill") >= 4
        assert "Result figure 1" in html
        assert "Result figure 2" in html
        assert "Result figure 1 unavailable" not in html
        assert "Result figure 2 unavailable" not in html

    def test_render_avoids_duplicate_formula_blocks(self, tmp_path):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        bp.sections[0].content_md = "Problem statement only."
        bp.sections[1].content_md = "Method overview only."
        bp.sections[2].content_md = "Key idea only."
        bp.sections[3].content_md = "| Dataset | PSNR | SSIM |\n| --- | --- | --- |"
        bp.sections[4].content_md = "Contribution."
        bp.sections[5].content_md = "Highlight."
        bp.sections[6].content_md = "Code."
        bp.figure_placements = []
        bp.color_scheme = _make_blueprint().color_scheme
        bp.code_url = ""
        bp.sections[0].content_md += "\n\n$$ a=b $$"
        bp.sections[1].content_md += "\n\n$$ c=d $$"
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, tmp_path)
        assert html.count("formula-box") <= 2
        assert "inlineMath: [['$', '$']]" in html

    def test_resolve_figure_path_prefers_existing_pdf_or_image(self, tmp_path):
        source_dir = tmp_path / "paper"
        source_dir.mkdir()
        pdf = source_dir / "figure1.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%fake")
        resolved = HtmlPosterRenderer._resolve_figure_path("figure1", str(source_dir))
        assert resolved is not None
        assert resolved.name == "figure1.pdf"

    def test_build_figure_map_uses_browser_uri(self, tmp_path):
        source_dir = tmp_path / "paper"
        source_dir.mkdir()
        img = source_dir / "plot.png"
        img.write_bytes(b"fakepng")
        doc = PaperDocument(
            paper_id="test-999",
            arxiv_id="9999.99999",
            title="Test",
            raw_markdown=".",
            source_dir=str(source_dir),
            figures=[Figure(figure_id="fig-001", caption="Plot", local_path=str(img), section_id="s3")],
        )
        bp = _make_blueprint()
        bp.figure_placements = [
            type("FP", (), {"figure_id": "fig-001", "section_id": "s3", "width_ratio": 0.9, "caption": "Plot"})()
        ]
        fig_map = HtmlPosterRenderer._build_figure_map(bp, doc, tmp_path)
        assert fig_map["s3"][0]["src"] == "paper/plot.png"

    def test_build_figure_map_matches_semantic_blueprint_ids(self, tmp_path, monkeypatch):
        source_dir = tmp_path / "paper"
        source_dir.mkdir()
        pdf = source_dir / "framework-small.pdf"
        pdf.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF")
        doc = PaperDocument(
            paper_id="test-999",
            arxiv_id="9999.99999",
            title="Test",
            raw_markdown=".",
            source_dir=str(source_dir),
            figures=[Figure(figure_id="fig-002", label="fig:framework", caption="Framework of Data-Free Adversarial Distillation.", local_path=str(pdf), section_id="s1")],
        )
        bp = _make_blueprint()
        bp.figure_placements = [
            type("FP", (), {"figure_id": "fig:framework", "section_id": "s3", "width_ratio": 0.9, "caption": "Framework of Data-Free Adversarial Distillation."})()
        ]

        def _fake_copy_or_rasterize_asset(src, out_dir, target_name=None):
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{target_name}.png"
            target.write_bytes(b"fakepng")
            return target

        monkeypatch.setattr("src.renderers.html_renderer.copy_or_rasterize_asset", _fake_copy_or_rasterize_asset)
        fig_map = HtmlPosterRenderer._build_figure_map(bp, doc, tmp_path)
        assert fig_map["s3"][0]["caption"].startswith("Framework of Data-Free")
        assert fig_map["s3"][0]["src"].endswith(".png")
        assert "figures/" in fig_map["s3"][0]["src"]
        assert fig_map["s3"][0]["section_type"] == "key_idea"

    def test_build_figure_map_marks_method_and_results_sections(self, tmp_path):
        source_dir = tmp_path / "paper"
        source_dir.mkdir()
        img = source_dir / "plot.png"
        img.write_bytes(b"fakepng")
        doc = PaperDocument(
            paper_id="test-999",
            arxiv_id="9999.99999",
            title="Test",
            raw_markdown=".",
            source_dir=str(source_dir),
            figures=[
                Figure(figure_id="fig-101", caption="Framework overview", local_path=str(img), section_id="s2"),
                Figure(figure_id="fig-102", caption="Results on benchmarks", local_path=str(img), section_id="s3"),
            ],
        )
        bp = _make_blueprint()
        bp.sections = [
            type("Sec", (), {"section_id": "s2", "type": "main_method"})(),
            type("Sec", (), {"section_id": "s3", "type": "experiments"})(),
        ]
        bp.figure_placements = [
            type("FP", (), {"figure_id": "fig-101", "section_id": "s2", "width_ratio": 0.95, "caption": "Framework overview"})(),
            type("FP", (), {"figure_id": "fig-102", "section_id": "s3", "width_ratio": 0.96, "caption": "Results on benchmarks"})(),
        ]
        fig_map = HtmlPosterRenderer._build_figure_map(bp, doc, tmp_path)
        assert fig_map["s2"][0]["section_type"] == "main_method"
        assert fig_map["s3"][0]["section_type"] == "experiments"

    def test_copy_or_rasterize_asset_retries_on_copy_failure(self, tmp_path, monkeypatch):
        src = tmp_path / "figure.png"
        src.write_bytes(b"fakepng")
        out_dir = tmp_path / "out"

        calls = {"count": 0}

        def _fake_copyfile(_src, _dst):
            calls["count"] += 1
            if calls["count"] < 3:
                raise PermissionError("locked")
            Path(_dst).write_bytes(b"fakepng")

        monkeypatch.setattr("src.utils.figure_assets.shutil.copyfile", _fake_copyfile)
        monkeypatch.setattr("src.utils.figure_assets.shutil.copystat", lambda *args, **kwargs: None)
        result = copy_or_rasterize_asset(src, out_dir, "figure")
        assert result is not None
        assert result.exists()

    def test_copy_or_rasterize_asset_returns_target_for_same_file(self, tmp_path):
        src = tmp_path / "figure.png"
        src.write_bytes(b"fakepng")
        result = copy_or_rasterize_asset(src, tmp_path, "figure")
        assert result == src
