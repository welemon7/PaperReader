"""Patch applier: review -> controlled patches -> scene mutation."""

from __future__ import annotations

from src.agents.patch_applier import apply_patches, patches_from_review
from src.layout.scene import PosterScene, SceneConstraints, SceneElement, ScenePanel
from src.schemas.poster_v2 import PosterComment, PosterReview


def _scene() -> PosterScene:
    return PosterScene(
        paper_id="p", poster_title="T",
        panels=[
            ScenePanel(
                panel_id="sec-motivation", panel_type="motivation", title="Motivation", zone="left",
                elements=[SceneElement(element_id="sec-motivation.t0", kind="text", content_md=" ".join(["word"] * 120))],
                constraints=SceneConstraints(min_font_scale=0.82),
            ),
            ScenePanel(
                panel_id="sec-main-method", panel_type="main_method", title="Core Results", zone="right_bottom",
                elements=[
                    SceneElement(element_id="sec-main-method.t0", kind="text", content_md="Short intro."),
                    SceneElement(element_id="sec-main-method.f0", kind="figure", figure_src="figures/a.png", figure_aspect=1.5),
                ],
            ),
            ScenePanel(
                panel_id="sec-project", panel_type="project_link", title="Project", zone="bottom_right",
                elements=[SceneElement(element_id="sec-project.qr", kind="qr", content_md="[QR]")],
            ),
        ],
    )


def _review(issues: list[PosterComment]) -> PosterReview:
    return PosterReview(quality_score=5, needs_improvement=True, issues=issues, summary="s")


def test_condense_patch_trims_text():
    scene = _scene()
    review = _review([PosterComment(issue="Too dense", severity="warning", target="sec-motivation",
                                    suggestion="", action="condense")])
    patches = patches_from_review(scene, review)
    assert patches and patches[0].kind == "condense_text"
    applied = apply_patches(scene, patches)
    assert applied
    el = scene.panel("sec-motivation").elements[0]
    assert len(el.content_md.split()) <= 75  # section_budget("motivation")


def test_resize_figure_patch():
    scene = _scene()
    review = _review([PosterComment(issue="Figure too large", severity="warning", target="sec-main-method",
                                    suggestion="", action="resize")])
    patches = patches_from_review(scene, review)
    assert any(p.kind == "resize_figure" for p in patches)
    apply_patches(scene, patches)
    fig = [e for e in scene.panel("sec-main-method").elements if e.kind == "figure"][0]
    assert 0.15 <= fig.box_hint <= 0.7


def test_remove_figure_patch():
    scene = _scene()
    review = _review([PosterComment(issue="Broken figure", severity="error", target="sec-main-method",
                                    suggestion="", action="remove")])
    apply_patches(scene, patches_from_review(scene, review))
    assert all(e.kind != "figure" for e in scene.panel("sec-main-method").elements)


def test_patches_ignore_unknown_targets():
    scene = _scene()
    review = _review([PosterComment(issue="x", severity="info", target="sec-nonexistent", suggestion="", action="condense")])
    patches = patches_from_review(scene, review)
    assert patches == []


def test_font_patch_from_audit_hint():
    scene = _scene()
    # 模拟求解器把正文缩小过（0.85），审计提示字号过小
    scene.panel("sec-motivation").elements[0].font_scale = 0.85
    review = PosterReview(
        quality_score=5, needs_improvement=True, issues=[], summary="s",
        deterministic_checks={"checks": [{"name": "min_body_font", "passed": False}]},
    )
    patches = patches_from_review(scene, review)
    assert any(p.kind == "adjust_font" for p in patches)
    apply_patches(scene, patches)
    text_el = scene.panel("sec-motivation").elements[0]
    assert text_el.font_scale == 1.0  # 恢复到默认，不超 1.0


def test_font_patch_never_upscales_past_default():
    scene = _scene()
    scene.panel("sec-motivation").elements[0].font_scale = 1.0
    review = PosterReview(
        quality_score=5, needs_improvement=True, issues=[], summary="s",
        deterministic_checks={"checks": [{"name": "min_body_font", "passed": False}]},
    )
    apply_patches(scene, patches_from_review(scene, review))
    assert scene.panel("sec-motivation").elements[0].font_scale == 1.0
