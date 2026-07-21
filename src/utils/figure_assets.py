from __future__ import annotations

import logging
import re
import subprocess
import shutil
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


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
                    return target
                generated = output_base.with_suffix(".png")
                if generated.exists():
                    if generated != target:
                        shutil.copy2(generated, target)
                    return target
        except Exception as exc:
            logger.warning("Failed to rasterize PDF figure with pdftocairo %s: %s", src, exc)

        try:
            import fitz

            with fitz.open(str(src)) as doc:
                page = doc[0]
                pix = page.get_pixmap(dpi=220, alpha=False)
                pix.save(str(target))
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
                return target
            generated = output_base.with_suffix(".png")
            if generated.exists():
                if generated != target:
                    shutil.copy2(generated, target)
                return target
        except Exception as exc:
            logger.warning("Failed to rasterize PDF figure with pdftoppm %s: %s", src, exc)
            return None

    if suffix in _IMAGE_SUFFIXES:
        target = out_dir / f"{safe_name}{suffix}"
        if src != target:
            try:
                shutil.copy2(src, target)
            except Exception as exc:
                logger.warning("Failed to copy figure asset %s: %s", src, exc)
                return src
        return target

    return src


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
