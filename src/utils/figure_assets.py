from __future__ import annotations

import logging
import re
import subprocess
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

logger = logging.getLogger(__name__)

_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_VECTOR_SUFFIXES = {".svg"}
_IMAGE_SUFFIXES = _RASTER_SUFFIXES | _VECTOR_SUFFIXES
_CONTENT_BACKGROUND_THRESHOLD = 240


def crop_content_with_padding(
    image_path: str | Path,
    padding: int = 10,
) -> tuple[Image.Image, dict[str, list[int] | float]]:
    """Crop near-white margins and return the image with JSON-safe metadata."""
    if padding < 0:
        raise ValueError("padding must be non-negative")

    with Image.open(image_path) as source:
        image = source.copy()

    width, height = image.size
    grayscale = image.convert("L")
    content_mask = grayscale.point(
        lambda pixel: 255 if pixel < _CONTENT_BACKGROUND_THRESHOLD else 0
    )
    bbox = content_mask.getbbox()

    if bbox is None:
        cropped = image
    else:
        left, top, right, bottom = bbox
        cropped = image.crop(
            (
                max(0, left - padding),
                max(0, top - padding),
                min(width, right + padding),
                min(height, bottom + padding),
            )
        )

    cropped_width, cropped_height = cropped.size
    metadata: dict[str, list[int] | float] = {
        "original_size": [width, height],
        "cropped_size": [cropped_width, cropped_height],
        "aspect_ratio": cropped_width / cropped_height if cropped_height else 0.0,
    }
    return cropped, metadata


def _crop_image_file(image_path: Path, padding: int = 10) -> dict[str, list[int] | float] | None:
    """Crop an image in place, retaining the original file if processing fails."""
    if image_path.suffix.lower() in _VECTOR_SUFFIXES:
        return None

    temporary_path = image_path.with_name(f".{image_path.stem}.crop{image_path.suffix}")
    try:
        with Image.open(image_path) as source:
            image_format = source.format
        cropped, metadata = crop_content_with_padding(image_path, padding=padding)
        cropped.save(temporary_path, format=image_format)
        temporary_path.replace(image_path)
        return metadata
    except Exception as exc:
        logger.warning("Failed to crop image asset %s: %s", image_path, exc)
        return None
    finally:
        temporary_path.unlink(missing_ok=True)


def sanitize_asset_name(name: str, fallback: str = "asset") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name or fallback).strip("_")
    return cleaned or fallback


def resolve_figure_source(local_path: str | None, source_dir: str | Path | None) -> Path | None:
    if not local_path:
        return None

    source_dir_path = Path(source_dir) if source_dir else None
    path = Path(local_path)

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)
        if source_dir_path is not None:
            candidates.append(source_dir_path / path)
            candidates.append(source_dir_path / path.name)
            if not path.suffix:
                for suffix in _IMAGE_SUFFIXES | {".pdf"}:
                    candidates.append(source_dir_path / f"{path.name}{suffix}")
    if not path.suffix:
            for suffix in _IMAGE_SUFFIXES | {".pdf"}:
                candidates.append(path.with_suffix(suffix))

    if source_dir_path is not None:
        candidates.append(source_dir_path / "tex_src" / path)
        candidates.append(source_dir_path / "tex_src" / path.name)

    # Check exact match first, then common suffixes next to the source tree.
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if source_dir_path is not None and not path.is_absolute() and not path.suffix:
        for base in (source_dir_path, source_dir_path / "tex_src"):
            if not base.exists():
                continue
            for suffix in _IMAGE_SUFFIXES | {".pdf"}:
                candidate = base / f"{path.name}{suffix}"
                if candidate.exists():
                    return candidate.resolve()
            for match in base.rglob(f"{path.name}.*"):
                if match.suffix.lower() in _IMAGE_SUFFIXES or match.suffix.lower() == ".pdf":
                    return match.resolve()

    return None


def copy_or_rasterize_asset(src: Path, out_dir: Path, target_name: str | None = None) -> Path | None:
    src = src.resolve()
    if not src.exists():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_asset_name(target_name or src.stem, src.stem)
    suffix = src.suffix.lower()

    if suffix == ".pdf":
        target = out_dir / f"{safe_name}.png"
        if target.exists() and target.stat().st_mtime >= src.stat().st_mtime:
            return target
        try:
            converter = _find_pdftocairo()
            if converter:
                output_base = target.with_suffix("")
                cmd = [converter, "-png", "-singlefile", "-r", "220", str(src), str(output_base)]
                subprocess.run(cmd, check=True, capture_output=True)
                if target.exists():
                    _crop_image_file(target)
                    return target
                generated = output_base.with_suffix(".png")
                if generated.exists():
                    if generated != target:
                        shutil.copy2(generated, target)
                    _crop_image_file(target)
                    return target
        except Exception as exc:
            logger.warning("Failed to rasterize PDF figure with pdftocairo %s: %s", src, exc)

        try:
            import fitz

            with fitz.open(str(src)) as doc:
                page = doc[0]
                pix = page.get_pixmap(dpi=220, alpha=False)
                pix.save(str(target))
            _crop_image_file(target)
            return target
        except Exception as exc:
            logger.warning("Failed to rasterize PDF figure %s: %s", src, exc)
        converter = _find_pdftoppm()
        if not converter:
            return None
        try:
            output_base = target.with_suffix("")
            cmd = [converter, "-png", "-singlefile", str(src), str(output_base)]
            subprocess.run(cmd, check=True, capture_output=True)
            if target.exists():
                _crop_image_file(target)
                return target
            generated = output_base.with_suffix(".png")
            if generated.exists():
                if generated != target:
                    shutil.copy2(generated, target)
                _crop_image_file(target)
                return target
        except Exception as exc:
            logger.warning("Failed to rasterize PDF figure with pdftoppm %s: %s", src, exc)
            return None

    if suffix in _IMAGE_SUFFIXES:
        target = out_dir / f"{safe_name}{suffix}"
        if src.resolve() == target.resolve() if target.exists() else src == target:
            return target
        if src != target:
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    shutil.copyfile(src, target)
                    shutil.copystat(src, target, follow_symlinks=True)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(0.2 * (attempt + 1))
                    else:
                        logger.warning("Failed to copy figure asset %s: %s", src, exc)
            if last_exc is not None:
                return src
        _crop_image_file(target)
        return target

    return src


def save_svg_asset(
    svg_or_symbol: str,
    out_dir: Path,
    target_name: str,
    *,
    width: int = 320,
    height: int = 320,
    view_box: str = "0 0 320 320",
) -> Path:
    """Persist a generated vector asset as SVG.

    Accepts either a complete SVG document or a symbol/string payload. Plain
    symbols are wrapped into a minimal SVG so the asset can be reused directly
    from HTML later.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_asset_name(target_name, "vector_asset")
    target = out_dir / f"{safe_name}.svg"

    content = (svg_or_symbol or "").strip()
    if not content:
        raise ValueError("svg_or_symbol must not be empty")

    # Accept both the default SVG namespace and ElementTree's prefixed form
    # (for example ``<ns0:svg ...>``). Otherwise a valid serialized SVG can be
    # mistaken for a plain symbol and embedded as escaped text.
    if not re.search(r"<(?:(?:[A-Za-z_][\w.-]*):)?svg\b", content, flags=re.IGNORECASE):
        symbol = html_escape(content)
        content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{view_box}">'
            '<rect width="100%" height="100%" fill="white"/>'
            f'<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
            f'font-family="Inter, Segoe UI, sans-serif" font-size="120" fill="#16324f">{symbol}</text>'
            "</svg>"
        )

    target.write_text(content, encoding="utf-8")
    return target


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def create_fallback_preview(src: Path, out_dir: Path, target_name: str) -> Path | None:
    """Create a browser-friendly preview when the original asset cannot be used directly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = out_dir / f"{sanitize_asset_name(target_name)}.png"
    if preview.exists():
        return preview
    try:
        if src.suffix.lower() == ".pdf":
            return copy_or_rasterize_asset(src, out_dir, target_name)
        if src.suffix.lower() in _IMAGE_SUFFIXES:
            return copy_or_rasterize_asset(src, out_dir, target_name)
    except Exception:
        pass
    return None


def collect_assets(paths: Iterable[Path]) -> list[Path]:
    collected: list[Path] = []
    for path in paths:
        if path.exists():
            collected.append(path)
    return collected


def _find_pdftoppm() -> str | None:
    for name in ("pdftoppm", "pdftoppm.cmd", "pdftoppm.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _find_pdftocairo() -> str | None:
    for name in ("pdftocairo", "pdftocairo.exe", "pdftocairo.cmd"):
        found = shutil.which(name)
        if found:
            return found
    return None
