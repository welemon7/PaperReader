from pathlib import Path

from src.agents.poster_harness import (
    BlankRegionCandidate,
    SectionBlankReport,
    _analyze_blank_regions,
    _measure_png_blank_ratio,
    _should_supplement_report,
    _size_supplement_svg,
    _supplement_overlay_html,
)
from src.schemas.poster import PosterSection


def test_blank_ratio_and_geometry(tmp_path: Path):
    from PIL import Image, ImageDraw

    path = tmp_path / "section.png"
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((15, 68, 85, 72), fill=(0, 0, 0))
    image.save(path)
    assert 0.95 <= (_measure_png_blank_ratio(path) or 0) <= 1.0
    regions = _analyze_blank_regions(path)
    assert regions and max(regions, key=lambda x: x["area_pixels"])["area_pixels"] > 0


def test_supplement_threshold_is_35_percent():
    section = PosterSection(section_id="sec-motivation", type="motivation", title="Motivation")
    values = dict(section_id=section.section_id, section_type=section.type,
                  section_title=section.title, content_ratio=0.65, width=100, height=100,
                  text_words=20, figure_count=0, has_figures=False)
    assert not _should_supplement_report(SectionBlankReport(blank_ratio=0.349, **values), section)
    assert _should_supplement_report(SectionBlankReport(blank_ratio=0.35, **values), section)


def test_supporting_sections_are_skipped():
    values = dict(section_id="sec-project", section_type="project_link", section_title="Project",
                  blank_ratio=0.9, content_ratio=0.1, width=100, height=100,
                  text_words=1, figure_count=0, has_figures=False)
    report = SectionBlankReport(**values)
    for section_type in ("contributions", "highlights", "project_link"):
        section = PosterSection(section_id=f"sec-{section_type}", type=section_type, title=section_type)
        assert not _should_supplement_report(report, section)


def test_svg_supplement_uses_detected_region_geometry():
    candidate = BlankRegionCandidate(
        section_id="sec-motivation", section_type="motivation", section_title="Motivation",
        blank_ratio=0.4, content_ratio=0.6, width=100, height=100, text_words=20,
        figure_count=0, has_figures=False, local_context="context", nearby_context="",
        global_context="", blank_regions=[{"x": 0, "y": 60, "width": 100, "height": 40, "area_pixels": 4000}],
    )
    assert 'width="100"' in _size_supplement_svg('<svg viewBox="0 0 10 10"></svg>', candidate)
    assert 'left:0px;top:6px;width:64px;height:24px' in _supplement_overlay_html(
        "figures/sec-motivation_supplement.svg", candidate, "Motivation"
    )
