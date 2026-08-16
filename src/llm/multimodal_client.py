from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".pdf": "application/pdf",
}


def multimodal_analyze(
        system_prompt: str,
        image_paths: list[str],
        user_text: str = "",
) -> Optional[dict[str, Any]]:
    """Send images + text to multimodal LLM using unified configuration.

    Uses the unified LLM configuration from settings (llm_api_key, llm_base_url, llm_model).
    """
    images = [(str(p), "poster") for p in image_paths]
    return multimodal_analyze_labeled(system_prompt, images, user_text)


def multimodal_analyze_labeled(
        system_prompt: str,
        images: list[tuple[str, str]],
        user_text: str = "",
        model: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Send labeled images + text to a multimodal LLM.

    Args:
        system_prompt: System prompt for the review task.
        images: List of (image_path, label) pairs. Labels help the VLM
            reference specific regions (e.g. "section: Motivation").
        user_text: Additional user instructions.
        model: Optional model override (defaults to settings.llm_model).

    Returns:
        Parsed JSON dict from the VLM, or None on failure / empty images.
    """
    # 直接使用统一的 LLM 配置
    api_key = settings.llm_api_key
    base_url = settings.llm_base_url.rstrip("/")
    target_model = model or settings.llm_model

    if not api_key or api_key in ("", "sk-your-key-here"):
        logger.warning("LLM API key not configured")
        return None

    content: list[dict] = [
        {"type": "text", "text": system_prompt + "\n\n" + user_text}
    ]
    # Keep this canonical order explicit.  The old review caller accidentally
    # passed (label, path), which silently caused every image to be skipped and
    # made the "visual" loop fall back to text-only optimisation.
    for idx, (img_path, label) in enumerate(images, start=1):
        p = Path(img_path)
        if not p.exists():
            logger.warning("Image not found: %s", img_path)
            continue
        try:
            mime_type = _IMAGE_MIME_TYPES.get(p.suffix.lower(), "image/png")
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            if label:
                content.append({"type": "text", "text": f"--- Image {idx} ({label}) ---"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            })
        except Exception as e:
            logger.warning("Failed to encode %s: %s", img_path, e)

    if len(content) <= 1:
        logger.warning("No valid images to send")
        return None

    body = {
        "model": target_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 8192,
    }

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        logger.info("Multimodal analysis: %d chars received", len(text))
        return json.loads(text)
    except Exception as e:
        logger.exception("Multimodal API call failed: %s", e)
        return None


def downscale_image(image_path: Path, max_width: int = 1400) -> Optional[Path]:
    """Downscale a captured poster PNG to keep VLM token usage bounded.

    Returns the path of a downscaled copy (written next to the original as
    ``<stem>_small.png``), or the original path when already small enough.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not installed; skipping downscale")
        return None

    image_path = Path(image_path)
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width <= max_width:
                return image_path
            ratio = max_width / float(width)
            new_size = (max_width, int(height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            target = image_path.with_name(f"{image_path.stem}_small.png")
            img.save(target, "PNG")
        return target
    except Exception as e:
        logger.warning("Downscale failed for %s: %s", image_path, e)
        return None


def capture_poster(html_path: Path, png_path: Path, width: int = 1200, height: int = 1697) -> Optional[Path]:
    """Capture poster HTML as PNG image using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("Playwright not installed. pip install playwright && playwright install chromium")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.goto(_as_file_uri(html_path), wait_until="load")
            page.wait_for_timeout(1500)
            page.evaluate("document.fonts && document.fonts.ready")
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        logger.info("Poster captured: %s (%d KB)", png_path, png_path.stat().st_size // 1024)
        return png_path
    except Exception as e:
        logger.warning("Poster capture failed: %s. Run: playwright install chromium", e)
        return None


def capture_poster_full_and_sections(
        html_path: Path,
        png_path: Path,
        section_selectors: dict[str, str],
        width: int = 1200,
        height: int = 1697,
) -> dict[str, Path]:
    """Capture the full poster plus per-section element screenshots in one browser session.

    Args:
        html_path: Rendered poster HTML file.
        png_path: Where to save the full-page PNG.
        section_selectors: Mapping of section name -> CSS selector (e.g. "#sec-motivation").
        width / height: Viewport size for the full-page capture.

    Returns:
        Dict mapping section name -> cropped PNG path (only successful captures).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("Playwright not installed. pip install playwright && playwright install chromium")
        return {}

    crops: dict[str, Path] = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.goto(_as_file_uri(html_path), wait_until="load")
            page.wait_for_timeout(1500)
            page.evaluate("document.fonts && document.fonts.ready")
            page.screenshot(path=str(png_path), full_page=True)
            for name, selector in section_selectors.items():
                try:
                    locator = page.locator(selector)
                    if locator.count() == 0:
                        logger.warning("Section selector not found: %s", selector)
                        continue
                    crop_path = png_path.with_name(f"{png_path.stem}_{name}.png")
                    locator.first.screenshot(path=str(crop_path))
                    crops[name] = crop_path
                except Exception as e:
                    logger.warning("Section screenshot failed for %s: %s", name, e)
            browser.close()
        logger.info("Poster captured: %s + %d crops", png_path, len(crops))
        return crops
    except Exception as e:
        logger.warning("Poster capture failed: %s. Run: playwright install chromium", e)
        return {}


def _as_file_uri(path: Path) -> str:
    resolved = path.resolve()
    return resolved.as_uri()
