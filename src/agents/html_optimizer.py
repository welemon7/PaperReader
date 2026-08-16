from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from src.llm.client import LLMClient
from src.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert HTML/CSS optimizer specializing in academic poster design.
Your task is to improve the given HTML based on the user's optimization instructions.

Rules:
1. Keep the core content and structure intact
2. Improve visual hierarchy and readability
3. Optimize for a 48 × 27 inch landscape academic poster
4. Maintain all existing figures, formulas, and data
5. Preserve the original semantic meaning
6. Output ONLY valid HTML, no markdown or explanations
7. Keep all MathJax formulas and figure references intact
8. Improve whitespace, typography, and layout balance
9. Ensure responsive design principles where applicable
10. Maintain the color scheme unless specifically instructed to change it

Return the complete optimized HTML document."""


def optimize_html_with_llm(
    html_path: Path | str,
    prompt_path: Path | str,
    output_path: Optional[Path | str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Optimize an HTML poster using LLM with a custom prompt.

    Args:
        html_path: Path to the HTML file to optimize
        prompt_path: Path to the optimization instructions (LLM-up.txt)
        output_path: Path to save the optimized HTML (optional)
        model: LLM model to use (defaults to settings.llm_model)

    Returns:
        Optimized HTML content as string
    """
    # 读取输入文件
    html_path = Path(html_path)
    prompt_path = Path(prompt_path)

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    # 读取内容
    html_content = html_path.read_text(encoding='utf-8')
    user_prompt = prompt_path.read_text(encoding='utf-8')
    allowed_figures = _collect_allowed_figure_assets(html_path.parent / "figures")

    logger.info(f"Loaded HTML: {len(html_content)} chars")
    logger.info(f"Loaded prompt: {len(user_prompt)} chars")
    if allowed_figures:
        user_prompt = (
            f"{user_prompt}\n\nAllowed figure assets (use only these local files for <img> tags):\n"
            + "\n".join(f"- figures/{name}" for name in sorted(allowed_figures))
        )

    # 构建完整的用户提示
    full_user_prompt = _build_user_prompt(html_content, user_prompt)

    # 调用 LLM
    client = LLMClient(model=model)

    try:
        logger.info("Sending optimization request to LLM...")
        response = client.chat(
            system=_SYSTEM_PROMPT,
            user=full_user_prompt,
        )
        # 清理响应（移除可能的 markdown 包裹）
        optimized_html = _clean_response(response)
        optimized_html = _normalize_figure_sources(optimized_html, allowed_figures)

        if not optimized_html or "<html" not in optimized_html.lower() or "</html>" not in optimized_html.lower():
            raise RuntimeError("LLM response did not contain a complete HTML document")

        logger.info(f"Received optimized HTML: {len(optimized_html)} chars")

        # 保存结果
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(optimized_html, encoding='utf-8')
            logger.info(f"Optimized HTML saved to: {output_path}")

        return optimized_html

    except Exception as e:
        logger.exception("LLM optimization failed")
        raise RuntimeError(f"Optimization failed: {e}")


def _build_user_prompt(html_content: str, user_instructions: str) -> str:
    """构建完整的用户提示词"""
    return f"""Here is the HTML poster to optimize:
    
    ```html
    {html_content}
    ```
    
    Optimization instructions from the user:
    {user_instructions}
    
    Please optimize the HTML according to the instructions. Return ONLY the complete HTML document, no explanations."""


def _collect_allowed_figure_assets(figures_dir: Path) -> set[str]:
    if not figures_dir.exists():
        return set()
    return {path.name for path in figures_dir.iterdir() if path.is_file()}


def _normalize_figure_sources(html: str, allowed_figures: set[str]) -> str:
    if not html or not allowed_figures:
        return html

    img_pattern = re.compile(
        r"<img(?P<before>[^>]*?)\bsrc=(?P<quote>[\"'])(?P<src>.*?)(?P=quote)(?P<after>[^>]*)>",
        re.IGNORECASE | re.DOTALL,
    )

    def _rewrite(match: re.Match[str]) -> str:
        before = match.group("before")
        quote = match.group("quote")
        src = match.group("src").strip()
        after = match.group("after")

        normalized = _allowed_figure_src(src, allowed_figures)
        if normalized:
            return f"<img{before} src={quote}{normalized}{quote}{after}>"
        return '<div class="figure-placeholder">Figure unavailable</div>'

    return img_pattern.sub(_rewrite, html)


def _allowed_figure_src(src: str, allowed_figures: set[str]) -> str:
    if not src or src.startswith("data:"):
        return ""
    parsed = urlsplit(src)
    candidate = Path(parsed.path).name
    if candidate in allowed_figures:
        return f"figures/{candidate}"
    if src.replace("\\", "/").startswith("figures/") and candidate in allowed_figures:
        return f"figures/{candidate}"
    return ""


def _clean_response(response: str) -> str:
    """清理 LLM 响应，移除可能的 markdown 包裹"""
    text = response.strip()

    # 移除 ```html 和 ``` 包裹
    if text.startswith("```"):
        lines = text.split('\n')
        if lines and lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines)

    # 移除任何非 HTML 内容（如果 LLM 添加了额外文本）
    if '<!DOCTYPE html>' not in text and '<html' not in text.lower():
        # 尝试提取 HTML 内容
        html_pattern = r'(<!DOCTYPE html>.*?</html>)'
        match = re.search(html_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
        else:
            # 尝试更宽松的匹配
            html_pattern = r'(<html.*?</html>)'
            match = re.search(html_pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)

    return text.strip()


def batch_optimize_html(
    html_path: Path | str,
    prompts_dir: Path | str,
    output_dir: Path | str,
    model: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    使用多个提示词文件批量优化 HTML

    Args:
        html_path: 原始 HTML 文件路径
        prompts_dir: 包含多个提示词文件的目录
        output_dir: 输出目录
        model: LLM 模型

    Returns:
        优化结果列表
    """
    html_path = Path(html_path)
    prompts_dir = Path(prompts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 遍历所有 .txt 提示词文件
    for prompt_file in sorted(prompts_dir.glob("*.txt")):
        logger.info(f"Processing prompt: {prompt_file.name}")

        # 生成输出文件名
        output_file = output_dir / f"optimized_{prompt_file.stem}.html"

        try:
            optimized = optimize_html_with_llm(
                html_path=html_path,
                prompt_path=prompt_file,
                output_path=output_file,
                model=model,
            )

            results.append({
                "prompt_file": str(prompt_file),
                "output_file": str(output_file),
                "success": True,
                "size": len(optimized),
            })

        except Exception as e:
            logger.error(f"Failed to process {prompt_file.name}: {e}")
            results.append({
                "prompt_file": str(prompt_file),
                "output_file": str(output_file),
                "success": False,
                "error": str(e),
            })

    return results


def optimize_with_feedback(
    html_path: Path | str,
    prompt_path: Path | str,
    feedback_path: Optional[Path | str] = None,
    iterations: int = 1,
    output_dir: Optional[Path | str] = None,
) -> list[dict[str, Any]]:
    """
    迭代优化 HTML，每轮可以加入新的反馈

    Args:
        html_path: 初始 HTML 文件
        prompt_path: 初始提示词
        feedback_path: 反馈提示词（每轮迭代使用）
        iterations: 迭代次数
        output_dir: 输出目录

    Returns:
        每轮优化结果
    """
    html_path = Path(html_path)
    prompt_path = Path(prompt_path)
    output_dir = Path(output_dir) if output_dir else html_path.parent / "optimized"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    current_html = html_path
    temp_prompt_path = None

    for i in range(iterations):
        logger.info(f"=== Iteration {i + 1}/{iterations} ===")

        # 构建当前轮的提示词
        if feedback_path and i > 0:
            # 使用反馈提示词（叠加）
            feedback = Path(feedback_path).read_text(encoding='utf-8')
            base_prompt = prompt_path.read_text(encoding='utf-8')
            user_prompt = f"{base_prompt}\n\nAdditional feedback from iteration {i}:\n{feedback}"
            temp_prompt_path = output_dir / f"temp_prompt_{i}.txt"
            temp_prompt_path.write_text(user_prompt, encoding='utf-8')
            current_prompt = temp_prompt_path
        else:
            current_prompt = prompt_path

        # 优化
        output_file = output_dir / f"iter_{i + 1:02d}.html"

        try:
            optimized = optimize_html_with_llm(
                html_path=current_html,
                prompt_path=current_prompt,
                output_path=output_file,
            )

            results.append({
                "iteration": i + 1,
                "output_file": str(output_file),
                "success": True,
                "size": len(optimized),
            })

            # 为下一轮准备
            current_html = output_file

        except Exception as e:
            logger.error(f"Iteration {i + 1} failed: {e}")
            results.append({
                "iteration": i + 1,
                "success": False,
                "error": str(e),
            })
            break

        # 清理临时文件
        if temp_prompt_path and temp_prompt_path.exists():
            temp_prompt_path.unlink()
            temp_prompt_path = None

    return results


def optimize_html_string(
    html_content: str,
    prompt: str,
    model: Optional[str] = None,
) -> str:
    """
    直接优化 HTML 字符串（不读写文件）

    Args:
        html_content: HTML 内容字符串
        prompt: 优化提示词
        model: LLM 模型

    Returns:
        优化后的 HTML 内容
    """
    client = LLMClient(model=model)
    full_prompt = _build_user_prompt(html_content, prompt)

    try:
        logger.info("Sending optimization request to LLM...")
        response = client.chat(
            system=_SYSTEM_PROMPT,
            user=full_prompt,
        )
        optimized_html = _clean_response(response)
        logger.info(f"Optimized HTML: {len(html_content)} -> {len(optimized_html)} chars")
        return optimized_html
    except Exception as e:
        logger.exception("LLM optimization failed")
        raise RuntimeError(f"Optimization failed: {e}")
