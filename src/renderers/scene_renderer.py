"""Scene-graph poster renderer (48x36 in, absolute-positioned layout).

Renders a solved ``SceneLayout`` to a standalone HTML poster.  Key properties:

- every panel is absolutely positioned by the solver, so columns align exactly
  regardless of image sizes,
- figures live in fixed uniform frames with ``object-fit: contain``,
- figure captions are NEVER rendered under images (product requirement); the
  caption stays in ``alt`` and in the scene notes for QA/review,
- text elements carry a solver-computed font scale.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.layout.scene import PosterScene
from src.layout.solver import PANEL_TITLE_H, SceneLayout
from src.renderers.html_renderer import HtmlPosterRenderer
from src.schemas.paper import PaperDocument

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

BODY_FONT_PX = 17.0


class SceneRenderer:
    def __init__(self, template_dir: str = _TEMPLATE_DIR) -> None:
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template("scene_poster.html.j2")

    def render(
        self,
        scene: PosterScene,
        layout: SceneLayout,
        doc: PaperDocument,
        output_dir: Path,
    ) -> str:
        """Render the solved layout to a standalone HTML string."""
        panels_meta: dict[str, dict] = {}
        for panel in scene.panels:
            panels_meta[panel.panel_id] = {"title": panel.title or panel.panel_type}

        # Per-element rendered HTML (text elements: markdown -> html).
        element_html: dict[str, str] = {}
        for panel in scene.panels:
            for el in panel.elements:
                if el.kind == "text" and el.content_md:
                    cleaned = HtmlPosterRenderer._clean_html_text(el.content_md)
                    element_html[el.element_id] = HtmlPosterRenderer._markdown_with_latex(cleaned)
                elif el.content_html:
                    element_html[el.element_id] = el.content_html

        html = self.template.render(
            poster_title=scene.poster_title,
            tagline=scene.tagline or HtmlPosterRenderer._summarize_text(doc.abstract, 1),
            authors_str=scene.authors_str,
            author_line=HtmlPosterRenderer._build_author_line(scene.authors_str, doc),
            code_url=scene.code_url,
            github_src=HtmlPosterRenderer._prepare_github_asset(output_dir) if scene.code_url else "",
            canvas_width=scene.canvas_width,
            canvas_height=scene.canvas_height,
            color_scheme=scene.color_scheme or {},
            theme=scene.theme,
            panels=layout.panels,
            panels_meta=panels_meta,
            elements=layout.elements,
            element_html=element_html,
            body_font=BODY_FONT_PX,
            panel_title_h=PANEL_TITLE_H,
        )
        return html

    def render_to_file(
        self,
        scene: PosterScene,
        layout: SceneLayout,
        doc: PaperDocument,
        output_path: Path,
    ) -> Path:
        html = self.render(scene, layout, doc, output_path.parent)
        output_path.write_text(html, encoding="utf-8")
        return output_path
