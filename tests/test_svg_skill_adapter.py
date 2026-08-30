from src.agents.svg_skill_adapter import (
    normalize_svg_dimensions,
    svg_generation_guidance,
    validate_svg_document,
)


def test_guidance_mentions_target_geometry_and_static_constraints():
    guidance = svg_generation_guidance(400, 200)
    assert "400x200" in guidance
    assert "aspect ratio" in guidance
    assert "no scripts" in guidance


def test_validate_svg_document_accepts_matching_standalone_svg():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200"><rect width="400" height="200" fill="white"/></svg>'
    assert validate_svg_document(svg, 400, 200) == (True, "ok")


def test_validate_svg_document_rejects_external_resources_and_bad_ratio():
    external = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200"><image href="https://example.com/x.png"/></svg>'
    assert not validate_svg_document(external, 400, 200)[0]
    wrong_ratio = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="10"/></svg>'
    assert not validate_svg_document(wrong_ratio, 400, 200)[0]
    no_namespace = '<svg viewBox="0 0 400 200"><rect width="400" height="200"/></svg>'
    assert not validate_svg_document(no_namespace, 400, 200)[0]


def test_normalize_svg_dimensions_keeps_internal_references_and_target_size():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="360" height="220" viewBox="0 0 360 220">
      <defs><marker id="arrow"><path d="M0 0L4 2L0 4Z"/></marker></defs>
      <path d="M10 10H100" marker-end="url(#arrow)"/>
    </svg>'''
    normalized = normalize_svg_dimensions(svg, 400, 200)
    assert 'width="400"' in normalized
    assert 'height="200"' in normalized
    assert 'viewBox="0 0 360 220"' in normalized
    assert 'preserveAspectRatio="xMidYMid meet"' in normalized
    assert validate_svg_document(normalized, 400, 200) == (True, "ok")
