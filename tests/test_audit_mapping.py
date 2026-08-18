"""Audit model mapping + capture helpers (non-browser units)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.agents.poster_harness import _merge_audit_issues
from src.schemas.poster_v2 import PosterReview
from src.schemas.review import DeterministicAudit, DeterministicCheck
from src.visual.capture import _downscale, _draw_grid_overlay, _side_by_side


def test_deterministic_audit_collects_hard_failures():
    audit = DeterministicAudit(available=True, reason="ok")
    audit.add(DeterministicCheck(name="canvas_size", passed=False, severity="error", detail="bad"))
    audit.add(DeterministicCheck(name="element_overlap", passed=True, severity="error"))
    audit.add(DeterministicCheck(name="blank_space", passed=False, severity="warning", detail="blank"))
    assert audit.hard_failures == ["canvas_size"]
    assert audit.warnings == ["blank_space"]
    assert audit.has_hard_failures()


def test_merge_audit_issues_adds_comments_and_hard_failures():
    audit = DeterministicAudit(available=True, reason="ok")
    audit.add(DeterministicCheck(
        name="text_clipping", passed=False, severity="error",
        detail="clipped", data={"clipped": ["sec-a", "sec-b"]},
    ))
    audit.add(DeterministicCheck(name="broken_images", passed=False, severity="error", detail="broken"))
    audit.add(DeterministicCheck(
        name="blank_space", passed=False, severity="warning", detail="blank", data={"blank": []},
    ))

    review = PosterReview(quality_score=8, needs_improvement=True, issues=[], summary="")
    _merge_audit_issues(review, audit)

    assert review.hard_failures == ["text_clipping", "broken_images"]
    targets = {c.target for c in review.issues}
    assert "sec-a" in targets and "sec-b" in targets
    assert any(c.action == "condense" for c in review.issues)
    assert any(c.action == "replace_figure" for c in review.issues)
    assert review.deterministic_checks["hard_failures"] == ["text_clipping", "broken_images"]


def test_merge_audit_issues_keeps_existing_targets():
    audit = DeterministicAudit(available=True, reason="ok")
    audit.add(DeterministicCheck(name="text_clipping", passed=False, severity="error",
                                 detail="x", data={"clipped": ["sec-a"]}))
    review = PosterReview(quality_score=8, needs_improvement=True, summary="",
                          issues=__import__("src.schemas.poster_v2", fromlist=["PosterComment"]).PosterComment
                          and [__import__("src.schemas.poster_v2", fromlist=["PosterComment"]).PosterComment(
                              issue="already flagged", severity="error", target="sec-a", action="condense")])
    _merge_audit_issues(review, audit)
    assert sum(1 for c in review.issues if c.target == "sec-a") == 1


def test_downscale_writes_smaller_copy(tmp_path):
    src = tmp_path / "big.png"
    Image.new("RGB", (4000, 2000), "white").save(src)
    dst = tmp_path / "small.png"
    result = _downscale(src, dst, max_width=2000)
    assert result is not None
    with Image.open(dst) as img:
        assert img.size == (2000, 1000)


def test_downscale_returns_source_when_small(tmp_path):
    src = tmp_path / "small.png"
    Image.new("RGB", (800, 400), "white").save(src)
    assert _downscale(src, tmp_path / "x.png", max_width=2000) == src


def test_grid_overlay_draws_section_boxes(tmp_path):
    src = tmp_path / "full.png"
    Image.new("RGB", (960, 720), "white").save(src)
    dst = tmp_path / "grid.png"
    bboxes = {"sec-a": [10, 20, 300, 200], "sec-b": [320, 20, 300, 200]}
    result = _draw_grid_overlay(src, dst, bboxes, canvas_width=960, canvas_height=720)
    assert result is not None
    with Image.open(dst) as img:
        assert img.size == (960, 720)


def test_side_by_side_creates_diff(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    Image.new("RGB", (800, 600), "red").save(a)
    Image.new("RGB", (800, 600), "blue").save(b)
    dst = tmp_path / "diff.png"
    assert _side_by_side(a, b, dst, total_width=1600) is not None
    with Image.open(dst) as img:
        assert img.width == 1608  # 800 + 8 + 800
