from __future__ import annotations

from pathlib import Path

from src.schemas.poster import PosterBlueprint, PosterSection
from src.schemas.paper import PaperDocument
from src.schemas.paper import Figure
from src.renderers.html_renderer import HtmlPosterRenderer


def _make_blueprint() -> PosterBlueprint:
    return PosterBlueprint(
        paper_id="test-999", poster_title="Test Poster",
        authors_str="Alice, Bob",
        sections=[
            PosterSection(section_id="s1", type="motivation", title="Motivation",
                        content_md="Our problem is **important**.", column=1, col_span=1, row=1),
            PosterSection(section_id="s2", type="method_overview", title="Method Overview",
                        content_md="We propose a method with $E=mc^2$.", column=2, col_span=1, row=1),
            PosterSection(section_id="s3", type="main_method", title="Method",
                        content_md="Details here.", column=1, col_span=2, row=2),
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

    def test_render_cleans_reference_fragments(self):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        bp.sections[0].content_md = r"See ~\\ref{tab:ablation} and ~\\cite{wang2018} for details."
        renderer = HtmlPosterRenderer()
        html = renderer.render(bp, doc, Path("output") / "9999.99999")
        assert "~\\ref" not in html
        assert "~\\cite" not in html
        assert "wang2018" not in html

    def test_render_to_file(self, tmp_path):
        doc = PaperDocument(paper_id="test-999", arxiv_id="9999.99999", title="Test", raw_markdown=".")
        bp = _make_blueprint()
        out = tmp_path / "poster.html"
        renderer = HtmlPosterRenderer()
        result = renderer.render_to_file(bp, doc, out)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Test Poster" in content

    def test_organize_rows(self):
        bp = _make_blueprint()
        rows = HtmlPosterRenderer._organize_rows(bp)
        assert len(rows) == 2
        assert len(rows[0]["sections"]) == 2
        assert len(rows[1]["sections"]) == 1

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
        assert fig_map["s3"][0]["section_type"] == "main_method"

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
