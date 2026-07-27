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
    # 直接使用统一的 LLM 配置
    api_key = settings.llm_api_key
    base_url = settings.llm_base_url.rstrip("/")
    model = settings.llm_model

    if not api_key or api_key in ("", "sk-your-key-here"):
        logger.warning("LLM API key not configured")
        return None

    content: list[dict] = [
        {"type": "text", "text": system_prompt + "\n\n" + user_text}
    ]
    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            logger.warning("Image not found: %s", img_path)
            continue
        try:
            mime_type = _IMAGE_MIME_TYPES.get(p.suffix.lower(), "image/png")
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
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
        "model": model,
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
            page.goto(_as_file_uri(html_path))
            page.wait_for_timeout(5000)
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        logger.info("Poster captured: %s (%d KB)", png_path, png_path.stat().st_size // 1024)
        return png_path
    except Exception as e:
        logger.warning("Poster capture failed: %s. Run: playwright install chromium", e)
        return None


def _as_file_uri(path: Path) -> str:
    resolved = path.resolve()
    return resolved.as_uri()