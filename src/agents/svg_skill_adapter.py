"""Load the project SVG skill and validate generated standalone SVG assets."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_SKILL_DIR = _PROJECT_ROOT / ".codex" / "skills" / "svg"
_SVG_NS = "http://www.w3.org/2000/svg"
_MAX_ELEMENTS = 90
ElementTree.register_namespace("", _SVG_NS)


def _read_svg_skill() -> str:
    """Read the single executable skill used by the SVG generation prompt."""
    path = _UPSTREAM_SKILL_DIR / "SKILL.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


_SVG_SKILL = _read_svg_skill()


def svg_generation_guidance(width: int, height: int) -> str:
    """Return the canonical SVG skill plus the current harness geometry."""
    aspect = width / height if height else 1.0
    skill = _SVG_SKILL or (
        "Generate one valid, self-contained browser SVG. Return SVG XML only."
    )
    return (
        f"{skill}\n\n"
        "RUNTIME HARNESS OVERRIDE (follow these values exactly):\n"
        f"- CONTENT is supplied by the caller above.\n"
        f"- WIDTH={width}px; HEIGHT={height}px; target region={width}x{height}; aspect ratio={aspect:.3f}.\n"
        f"- Root width and height must be exactly {width} and {height}.\n"
        f"- Use viewBox=\"0 0 {width} {height}\".\n"
        "- Use no scripts and no external resources.\n"
        "- The detected region is already the usable fill area; do not add an outer "
        "layout wrapper or design for a different canvas."
    )


def normalize_svg_dimensions(svg_text: str, width: int, height: int) -> str:
    """Normalize a valid model SVG to the harness region without changing its drawing."""
    content = (svg_text or "").strip()
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return content
    if root.tag != "{" + _SVG_NS + "}svg":
        return content
    root.set("width", str(width))
    root.set("height", str(height))
    view_box = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(view_box) == 4:
        root.set("viewBox", " ".join(view_box))
    else:
        root.set("viewBox", f"0 0 {width} {height}")
    # The detected region controls the available box, but must not distort the
    # model's internal geometry or text glyphs when its aspect ratio differs.
    root.set("preserveAspectRatio", "xMidYMid meet")
    return ElementTree.tostring(root, encoding="unicode")


def validate_svg_document(svg_text: str, width: int, height: int) -> tuple[bool, str]:
    """Validate a model response before it becomes a browser-loaded asset."""
    content = (svg_text or "").strip()
    if not content or len(content) > 120_000:
        return False, "empty or oversized SVG response"
    if re.search(
        r"<\s*(script|foreignObject)\b|(?:href|src)\s*=\s*['\"](?:https?:|//)|url\s*\(\s*['\"]?(?:https?:|//)",
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
    for name, expected in (("width", width), ("height", height)):
        value = root.attrib.get(name)
        if value is None:
            return False, f"root {name} is missing"
        try:
            if float(re.match(r"^\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*", value).group()) != expected:
                return False, f"root {name} does not match target region"
        except (AttributeError, ValueError):
            return False, f"root {name} is not numeric"
    view_box = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(view_box) != 4:
        return False, "viewBox must contain four numbers"
    try:
        vb_x, vb_y, vb_width, vb_height = (float(value) for value in view_box)
    except ValueError:
        return False, "viewBox contains non-numeric values"
    if vb_x != 0 or vb_y != 0:
        return False, "viewBox must begin at zero"
    if vb_width <= 0 or vb_height <= 0:
        return False, "viewBox dimensions must be positive"
    target_ratio = width / height if height else 1.0
    if abs((vb_width / vb_height) - target_ratio) / max(target_ratio, 0.01) > 0.35:
        return False, "viewBox aspect ratio does not fit the target region"
    if sum(1 for _ in root.iter()) > _MAX_ELEMENTS:
        return False, "SVG contains too many elements"
    return True, "ok"
