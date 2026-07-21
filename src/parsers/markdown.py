from __future__ import annotations

import logging
import re

from .latex_parser import ParseResult

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """Convert LaTeX to Markdown for downstream use."""

    def convert(self, result: ParseResult) -> str:
        """Full paper → Markdown string."""
        parts: list[str] = []

        # Title
        if result.title:
            parts.append(f"# {self._clean_text(result.title)}\n")

        # Authors
        if result.authors:
            names = [a["name"] for a in result.authors]
            parts.append(f"**{'; '.join(names)}**\n\n---\n")

        # Abstract
        if result.abstract:
            parts.append("## Abstract\n")
            parts.append(self._convert_section_body(result.abstract) + "\n")

        # Sections
        self._section_counter = 0
        for sec in result.section_bodies:
            parts.append(self._section_to_md(sec, result.figures))

        return "\n".join(parts)

    def _section_to_md(
        self, sec, all_figures: list = None
    ) -> str:
        """Single section body → Markdown heading + body."""
        heading = "#" * min(sec.level + 1, 6)  # +1 because Abstract is ##
        parts = [f"\n{heading} {self._clean_text(sec.title)}\n"]

        body = sec.raw_latex

        # Convert math environments: \[...\] → $$...$$
        body = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", body, flags=re.DOTALL)
        body = re.sub(
            r"\\begin\{(equation|equation\*|align|align\*|gather|gather\*)\}(.*?)\\end\{\1\}",
            r"$$\2$$",
            body,
            flags=re.DOTALL,
        )
        # Inline math: \(...\) → $...$
        body = re.sub(r"\\\((.*?)\\\)", r"$\1$", body, flags=re.DOTALL)

        # Remove figure environments but leave caption as italic
        body = re.sub(
            r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
            self._figure_placeholder,
            body,
            flags=re.DOTALL,
        )

        # Remove table environments
        body = re.sub(
            r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
            "",
            body,
            flags=re.DOTALL,
        )

        # Simple formatting
        body = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", body)
        body = re.sub(r"\\textit\{([^}]*)\}", r"*\1*", body)
        body = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", body)
        body = re.sub(r"\\(?:text)?tt\{([^}]*)\}", r"`\1`", body)

        # Remove remaining LaTeX commands
        body = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", "", body)
        body = re.sub(r"\s+", " ", body).strip()

        # Handle sections through recursive subsections
        if hasattr(sec, "subsections") and sec.subsections:
            sub_md = "\n".join(
                self._section_to_md(sub, all_figures) for sub in sec.subsections
            )
            body += "\n" + sub_md

        parts.append(body + "\n")
        return "\n".join(parts)

    @staticmethod
    def _figure_placeholder(m: re.Match) -> str:
        """Replace figure environment with an image placeholder in Markdown."""
        content = m.group(0)
        cap_m = re.search(r"\\caption\s*\{([^}]*)\}", content, re.DOTALL)
        caption = cap_m.group(1).strip() if cap_m else ""

        # Extract graphic path
        g_m = re.search(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]*)\}", content)
        img_path = g_m.group(1).strip() if g_m else "TODO"

        cap_text = f": {caption}" if caption else ""
        return f"\n![{img_path}]({img_path}){cap_text}\n"

    @staticmethod
    def _convert_section_body(latex: str) -> str:
        """Convert a short piece of LaTeX (abstract body etc.) to Markdown."""
        body = latex
        body = re.sub(r"\\\((.*?)\\\)", r"$\1$", body, flags=re.DOTALL)
        body = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", body, flags=re.DOTALL)
        body = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", body)
        body = re.sub(r"\\textit\{([^}]*)\}", r"*\1*", body)
        body = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", "", body)
        body = re.sub(r"\s+", " ", body).strip()
        return body

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove LaTeX commands from plain text fields."""
        text = re.sub(r"\{|\}", "", text)
        text = re.sub(r"\\(?:protect|label|ref|cite)\{.*?\}", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
