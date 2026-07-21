from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


def multimodal_analyze(
    system_prompt: str,
    image_paths: list[str],
    user_text: str = "",
) -> Optional[dict[str, Any]]:
    """Send images + text to Gemini multimodal (vision) API."""
    api_key = settings.gemini_api_key
    if not api_key or api_key in ("", "sk-your-key-here"):
        logger.warning("Gemini API key not configured for multimodal")
        return None

    base_url = settings.gemini_base_url.rstrip("/")
    model = settings.gemini_model

    content: list[dict] = [
        {"type": "text", "text": system_prompt + "\n\n" + user_text}
    ]
    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            logger.warning("Image not found: %s", img_path)
            continue
        try:
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
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
            page.goto(html_path.as_uri())
            page.wait_for_timeout(5000)
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        logger.info("Poster captured: %s (%d KB)", png_path, png_path.stat().st_size // 1024)
        return png_path
    except Exception as e:
        logger.warning("Poster capture failed: %s. Run: playwright install chromium", e)
        return None