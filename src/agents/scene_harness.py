"""Adapter that glues the scene pipeline into ``run_poster_harness``.

Provides the ``render_html`` / ``apply_feedback_override`` / ``rollback_patches``
callables the harness accepts, so the harness loop stays renderer-agnostic while
production renders through the scene graph: build scene (once) -> solve layout ->
render absolute-positioned HTML -> apply controlled patches -> re-solve.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from src.agents.patch_applier import apply_patches, patches_from_review
from src.layout.builder import build_scene
from src.layout.scene import PosterScene
from src.layout.solver import SceneLayout, solve_layout
from src.llm.client import LLMClient
from src.renderers.scene_renderer import SceneRenderer
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import PosterBlueprint
from src.schemas.poster_v2 import PosterReview

logger = logging.getLogger(__name__)


class SceneHarnessAdapter:
    """Stateful scene pipeline for one poster generation task."""

    def __init__(
        self,
        blueprint: PosterBlueprint,
        doc: PaperDocument,
        analysis: PaperAnalysis,
        theme: str = "academic",
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.blueprint = blueprint
        self.doc = doc
        self.analysis = analysis
        self.theme = theme
        self.llm = llm
        self.scene: Optional[PosterScene] = None
        self._assets_dir: Optional[Path] = None
        self._snapshot: Optional[PosterScene] = None
        self.renderer = SceneRenderer()

    # -- render -------------------------------------------------------------

    def render_html(self, blueprint: PosterBlueprint, doc: PaperDocument, round_dir: Path | str) -> str:
        round_dir = Path(round_dir)
        round_dir.mkdir(parents=True, exist_ok=True)
        if self.scene is None:
            self.scene = build_scene(blueprint, doc, self.analysis, round_dir, theme=self.theme)
            self._assets_dir = round_dir / "figures"
        else:
            if self._assets_dir is not None and self._assets_dir.exists():
                try:
                    shutil.copytree(self._assets_dir, round_dir / "figures", dirs_exist_ok=True)
                except Exception as exc:
                    logger.warning("Figure asset copy failed: %s", exc)
        layout = solve_layout(self.scene)
        self._layout = layout
        return self.renderer.render(self.scene, layout, doc, round_dir)

    def current_layout(self) -> Optional[SceneLayout]:
        return getattr(self, "_layout", None)

    # -- patches ------------------------------------------------------------

    def apply_feedback(self, review: PosterReview) -> list[str]:
        if self.scene is None:
            return []
        self._snapshot = self.scene.model_copy(deep=True)
        patches = patches_from_review(self.scene, review)
        if not patches:
            return []
        return apply_patches(self.scene, patches, self.llm)

    def rollback(self) -> None:
        if self._snapshot is not None:
            self.scene = self._snapshot
            self._snapshot = None
            logger.info("Scene rolled back to the pre-patch snapshot")

    def snapshot_after(self) -> None:
        """Commit the current scene as the new baseline (called on pass/stop)."""
        if self.scene is not None:
            self._snapshot = self.scene.model_copy(deep=True)
