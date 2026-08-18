"""Poster theme presets (design tokens; layout comes from the solver).

Each theme carries a complete set of design tokens consumed by the HTML
templates: colors, type, radii, shadows, the section-title symbol and the
title-band gradient.  Blueprints may still override individual colors via
``color_scheme`` (see :func:`resolve_colors`).
"""

from __future__ import annotations

from typing import Any

# Baseline token set that every theme must provide.
_BASE_TOKENS = {
    # colors
    "primary": "#16324f",        # section headers / titles / links
    "accent": "#c9a84c",         # gold accent (rules, symbols, badges)
    "accent_text": "#a8842c",    # accent darkened for text on white
    "background": "#ffffff",
    "text": "#182433",
    "muted": "#4a5568",
    "border": "#cfd8e3",
    "panel_soft": "#f7f9fc",
    "highlight": "#8fb3d9",
    "header_text": "#ffffff",    # text on the section-title band
    # type
    "font_display": "'Inter', 'Segoe UI', sans-serif",
    "font_body": "'Inter', 'Segoe UI', sans-serif",
    # section title band
    "section_symbol": "\u25b8 ",  # "▸ " prefix drawn before the section title
    "band_top": "#16324f",
    "band_bottom": "#1f4a75",
    # surfaces
    "radius": "10px",
    "shadow": "0 6px 16px rgba(22, 50, 79, 0.10)",
}

THEMES: dict[str, dict[str, str]] = {
    # ---- Academic: deep navy + gold (classic conference poster) ----
    "academic": {
        **_BASE_TOKENS,
        "primary": "#16324f",
        "accent": "#c9a84c",
        "accent_text": "#a8842c",
        "background": "#ffffff",
        "text": "#182433",
        "muted": "#4a5568",
        "border": "#cfd8e3",
        "panel_soft": "#f7f9fc",
        "highlight": "#8fb3d9",
        "header_text": "#ffffff",
        "band_top": "#16324f",
        "band_bottom": "#1f4a75",
        "shadow": "0 6px 16px rgba(22, 50, 79, 0.10)",
    },
    # ---- Minimal: graphite + sky blue (clean, modern, low noise) ----
    "minimal": {
        **_BASE_TOKENS,
        "primary": "#1f2937",
        "accent": "#0ea5e9",
        "accent_text": "#0369a1",
        "background": "#ffffff",
        "text": "#111827",
        "muted": "#6b7280",
        "border": "#d1d5db",
        "panel_soft": "#f9fafb",
        "highlight": "#93c5fd",
        "header_text": "#ffffff",
        "band_top": "#111827",
        "band_bottom": "#1f2937",
        "shadow": "0 6px 16px rgba(17, 24, 39, 0.08)",
    },
    # ---- Dark: deep slate + sky accent + amber highlight ----
    "dark": {
        **_BASE_TOKENS,
        "primary": "#0ea5e9",
        "accent": "#f59e0b",
        "accent_text": "#fbbf24",
        "background": "#0f172a",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "border": "#334155",
        "panel_soft": "#1e293b",
        "highlight": "#38bdf8",
        "header_text": "#0f172a",
        "band_top": "#0284c7",
        "band_bottom": "#0369a1",
        "shadow": "0 6px 18px rgba(2, 6, 23, 0.45)",
    },
}

DEFAULT_THEME = "academic"


def resolve_colors(theme: str, base: dict[str, Any] | None = None) -> dict[str, str]:
    """Merge a theme preset with any explicit blueprint colors.

    Blueprint colors win over the preset for the shared keys; all other
    tokens (symbol, band gradient, radii, shadows, fonts) come from the
    preset so a bare blueprint still renders with a complete design system.
    """
    preset = dict(THEMES.get(theme or DEFAULT_THEME, THEMES[DEFAULT_THEME]))
    base = base or {}
    mapping = {
        "primary": "primary",
        "accent": "accent",
        "background": "background",
        "text": "text",
        "muted": "muted",
        "border": "border",
        "panel_soft": "section_header_bg",
        "highlight": "highlight",
        "header_text": "section_header_text",
    }
    for target, source_key in mapping.items():
        value = base.get(source_key)
        if value:
            preset[target] = str(value)
    preset.setdefault("header_text", "#ffffff")
    return preset
