"""Project adapter for the vendored Software Mansion SVG guidance.

The upstream skill targets React Native, while this project embeds SVGs in a
browser-rendered scientific poster.  This adapter keeps the upstream files
unchanged and translates the reusable guidance into prompt rules plus a small
deterministic validity gate for model-generated SVG documents.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_SKILL_DIR = _PROJECT_ROOT / ".codex" / "skills" / "svg"
_SVG_NS = "http://www.w3.org/2000/svg"
_MAX_ELEMENTS = 90


def _read_upstream_guidance() -> str:
    parts: list[str] = []
    for name in ("SKILL.md", "svg.md", "when-to-use.md"):
        path = _UPSTREAM_SKILL_DIR / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


_UPSTREAM_GUIDANCE = _read_upstream_guidance()


def svg_generation_guidance(width: int, height: int) -> str:
    """Return compact, poster-specific rules derived from the vendored skill."""
    aspect = width / height if height else 1.0
    return (
        "SVG quality rules adapted from the vendored Software Mansion SVG skill:\n"
        "- This is a static browser poster asset: use a simple, legible composition, "
        "not a React component tree or animation.\n"
        "- Return one standalone <svg> with the SVG namespace and an explicit viewBox "
        "whose aspect ratio is close to the target region. Keep width and height at "
        f"{width}x{height}; target aspect ratio is {aspect:.3f}.\n"
        "- Preserve geometry inside the viewBox with comfortable padding. Use vector "
        "shapes and text only when labels are essential; avoid clipping and overflow.\n"
        "- Keep the asset lightweight: at most 90 visible elements, no scripts, no "
        "external images/fonts/URLs, and avoid filters, masks, patterns, and large "
        "decorative backgrounds unless they communicate the paper result.\n"
        "- Use a restrained white background, 1-2 accent colors, strong contrast, "
        "and no more than 3 short labels. Match visual density to the blank region.\n"
        "- Do not invent unsupported metrics or claims. Return SVG XML only."
    )


def validate_svg_document(svg_text: str, width: int, height: int) -> tuple[bool, str]:
    """Validate a model response before it becomes a browser-loaded asset."""
    content = (svg_text or "").strip()
    if not content or len(content) > 120_000:
        return False, "empty or oversized SVG response"
    if re.search(
        r"<\s*(script|foreignObject)\b|(?:href|src)\s*=\s*['\"]https?://|url\s*\(",
        content,
        re.I,
    ):
        return False, "script, foreign object, or external resource is not allowed"
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        return False, f"invalid XML: {exc}"
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        return False, "root element is not svg"
    if root.tag != "{" + _SVG_NS + "}svg":
        return False, "SVG namespace is missing or incorrect"
    view_box = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(view_box) != 4:
        return False, "viewBox must contain four numbers"
    try:
        _, _, vb_width, vb_height = (float(value) for value in view_box)
    except ValueError:
        return False, "viewBox contains non-numeric values"
    if vb_width <= 0 or vb_height <= 0:
        return False, "viewBox dimensions must be positive"
    target_ratio = width / height if height else 1.0
    if abs((vb_width / vb_height) - target_ratio) / max(target_ratio, 0.01) > 0.35:
        return False, "viewBox aspect ratio does not fit the target region"
    if sum(1 for _ in root.iter()) > _MAX_ELEMENTS:
        return False, "SVG contains too many elements"
    return True, "ok"
