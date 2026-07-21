from __future__ import annotations

from pathlib import Path
from src.schemas.poster import PosterBlueprint, PosterSection
from src.schemas.paper import PaperDocument
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
        html = renderer.render(bp, doc)
        assert "<!DOCTYPE html>" in html
        assert "Test Poster" in html
        assert "Alice, Bob" in html
        assert "Motivation" in html
        assert "Our problem is" in html
        assert "<strong>" in html

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