from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ParseResult:
    """Container for intermediate parsing result."""

    def __init__(self) -> None:
        self.merged_latex: str = ""
        self.title: str = ""
        self.authors: list[dict] = []
        self.abstract: str = ""
        self.section_bodies: list[SectionBody] = []  # top-level sections
        self.all_section_bodies: list[SectionBody] = []  # flattened
        self.formulas: list = []  # filled by extractor
        self.figures: list = []   # filled by extractor


class SectionBody:
    """Raw content of a single (sub)section."""

    def __init__(
        self,
        section_id: str,
        title: str,
        level: int,
        raw_latex: str,
        parent_id: Optional[str] = None,
    ) -> None:
        self.section_id = section_id
        self.title = title
        self.level = level
        self.raw_latex = raw_latex
        self.parent_id = parent_id
        self.subsections: list[SectionBody] = []

    def __repr__(self) -> str:
        return f"SectionBody({self.section_id}, lv={self.level}, '{self.title}')"


class LatexParser:
    """Parse LaTeX source into structured sections.

    Uses a hybrid approach:
    - regex for structural section parsing (robust across paper templates)
    - TexSoup for targeted environment extraction
    """

    def __init__(self, source_dir: Path) -> None:
        self.source_dir = source_dir

    def parse(self, main_tex: Path) -> ParseResult:
        """Parse the main .tex file and return structured result."""
        merged = self._merge_latex(main_tex)
        merged = self._preprocess(merged)

        result = ParseResult()
        result.merged_latex = merged

        result.title = self._extract_title(merged)
        result.authors = self._extract_authors(merged)
        result.abstract = self._extract_abstract(merged)
        result.section_bodies = self._extract_sections(merged)

        # Build flattened index
        LatexParser._flatten(result.section_bodies, result.all_section_bodies)

        return result

    # ---- merging ----

    def _merge_latex(self, main_tex: Path) -> str:
        """Read the main .tex and inline \\input / \\include files."""
        content = main_tex.read_text(encoding="utf-8", errors="replace")
        return self._resolve_inputs(content, main_tex.parent)

    def _resolve_inputs(self, content: str, base_dir: Path) -> str:
        """Replace \\input{file} and \\include{file} with file contents."""
        pattern = re.compile(r"\\(?:input|include)\{([^}]+)\}")

        def _replacer(m: re.Match) -> str:
            name = m.group(1)
            # Try .tex extension if not present
            tex_path = base_dir / (name if name.endswith(".tex") else name + ".tex")
            if not tex_path.exists():
                # Try subdirectories
                for p in sorted(base_dir.rglob(f"**/{name}.tex")):
                    tex_path = p
                    break
                if not tex_path.exists():
                    logger.warning("\\input file not found: %s", name)
                    return ""
            try:
                sub = tex_path.read_text(encoding="utf-8", errors="replace")
                # Recursively resolve inputs in the included file
                return self._resolve_inputs(sub, tex_path.parent)
            except Exception:
                logger.exception("Error reading included file %s", tex_path)
                return ""

        return pattern.sub(_replacer, content)

    # ---- preprocessing helpers ----

    @staticmethod
    def _preprocess(latex: str) -> str:
        """Remove comments and clean whitespace for easier parsing."""
        # Remove line comments (but not \\% or inside verbatim)
        lines = latex.split("\n")
        cleaned = []
        in_verbatim = False
        for line in lines:
            if re.search(r"\\begin\{verbatim\}", line):
                in_verbatim = True
            if in_verbatim:
                cleaned.append(line)
                if re.search(r"\\end\{verbatim\}", line):
                    in_verbatim = False
                continue
            # Remove comment chars at start or after some whitespace
            stripped = re.sub(r"(?<!\\)%.*", "", line)
            cleaned.append(stripped)
        merged = "\n".join(cleaned)
        # Collapse multiple blank lines
        merged = re.sub(r"\n{3,}", "\n\n", merged)
        return merged

    # ---- extraction helpers ----

    @staticmethod
    def _extract_title(latex: str) -> str:
        """Extract \\title{...} content."""
        m = re.search(r"\\title\s*\{([^}]*)\}", latex, re.DOTALL)
        if m:
            return m.group(1).strip().replace("\n", " ")
        return ""

    @staticmethod
    def _extract_authors(latex: str) -> list[dict]:
        """Extract \\author{...} with minimal parsing."""
        m = re.search(r"\\author\s*\{([^}]*)\}", latex, re.DOTALL)
        if not m:
            return []
        author_text = m.group(1)
        # Split by \\and
        names = re.split(r"\\and", author_text)
        authors = []
        for name in names:
            name = name.strip()
            # Remove \thanks, \footnote etc.
            name = re.sub(r"\\(?:thanks|footnote)\{.*?\}", "", name, flags=re.DOTALL)
            name = re.sub(r"\{|\}", "", name)
            name = name.strip()
            if name:
                authors.append({"name": name, "affiliation": None})
        return authors

    @staticmethod
    def _extract_abstract(latex: str) -> str:
        """Extract \\begin{abstract}...\\end{abstract} content."""
        m = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}", latex, re.DOTALL
        )
        if m:
            return m.group(1).strip()
        return ""

    # ---- section extraction ----

    @staticmethod
    def _extract_sections(latex: str) -> list[SectionBody]:
        """Split merged LaTeX into a flat ordered list of (sub)sections.

        Handles:
          \\section{...}
          \\subsection{...}
          \\subsubsection{...}
        Skips: \\section*{...} (unnumbered, e.g. References, Appendix)
        """
        # Regex: capture leading cruft before first section, then each section
        pattern = re.compile(
            r'\\(section|subsection|subsubsection)\s*(?:\*)?\{(?P<title>[^}]+)\}',
            re.DOTALL | re.IGNORECASE,
        )
        sections: list[SectionBody] = []
        section_counter = 0
        sub_counter = 0
        subsub_counter = 0
        current_section: Optional[SectionBody] = None
        current_subsection: Optional[SectionBody] = None

        # Find all sectioning commands
        pos = 0
        while True:
            m = re.search(
                r"\\(?:(?:sub)*(?:section|paragraph|subparagraph)"
                r"(?:\*)?)\s*(?:\[[^\]]*\])?\s*(?={)",
                latex[pos:],
            )
            if not m:
                # Remaining text goes to current
                tail = latex[pos:] if current_section else ""
                if current_subsection and tail.strip():
                    current_subsection.raw_latex += tail
                elif current_section and tail.strip():
                    current_section.raw_latex += tail
                break

            cmd_start = pos + m.start()
            # The command name
            cmd_text = m.group(0).strip()

            # Determine level from command name
            if cmd_text.startswith("\\subsubsection"):
                level = 3
            elif cmd_text.startswith("\\subsection"):
                level = 2
            elif cmd_text.startswith("\\section") or cmd_text.startswith("\\section*"):
                level = 1
            else:
                level = 4  # paragraph-like, skip for now
                pos = cmd_start + len(m.group(0))
                # Still need to find matching closing brace
                continue

            # Content before this section heading belongs to the previous section
            before_text = latex[pos:cmd_start]
            if current_subsection and before_text.strip():
                current_subsection.raw_latex += before_text
            elif current_section and before_text.strip():
                current_section.raw_latex += before_text

            # Extract the title from the braces
            brace_start = cmd_start + len(m.group(0))
            title, end_pos = LatexParser._read_braces(latex, brace_start)
            title = title.strip()
            if not title:
                pos = end_pos
                continue

            # Build SectionBody
            section_counter += 1
            sec_id = f"sec-{section_counter:03d}"
            sec_body = SectionBody(
                section_id=sec_id,
                title=title,
                level=level,
                raw_latex="",
                parent_id=current_section.section_id if level > 1 else None,
            )

            if level == 1:
                sections.append(sec_body)
                current_section = sec_body
                current_subsection = None
            elif level == 2 and current_section:
                current_section.subsections.append(sec_body)
                current_subsection = sec_body
            elif level == 3 and current_subsection:
                current_subsection.subsections.append(sec_body)

            pos = end_pos

        # ---- flatten tree to flat list preserving order ----
        flat: list[SectionBody] = []
        LatexParser._flatten(sections, flat)
        return sections

    @staticmethod
    def _flatten(
        sections: list[SectionBody], out: list[SectionBody]
    ) -> None:
        for s in sections:
            out.append(s)
            if s.subsections:
                # Convert nested SectionBody objects recursively
                for sub in s.subsections:
                    out.append(sub)
                    if sub.subsections:
                        for subsub in sub.subsections:
                            out.append(subsub)

    # ---- helper ----

    @staticmethod
    def _read_braces(text: str, start: int) -> tuple[str, int]:
        """Read a balanced brace group starting at ``start`` (which points to '{').
        Returns (content, index_of_closing_brace+1).
        """
        if start >= len(text) or text[start] != "{":
            return "", start
        depth = 1
        i = start + 1
        buf: list[str] = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth > 0:
                    buf.append(ch)
            else:
                buf.append(ch)
            i += 1
        return "".join(buf), i


# Helper to nest flat SectionBody list into tree
def build_section_tree(flat_sections: list) -> list:
    """Convert flat section list to nested tree based on level/parent_id.
    Each item is a dict with at least: section_id, level, parent_id, subsections=[], ...
    """
    lookup = {}
    for sec in flat_sections:
        sec_dict = {k: v for k, v in sec.__dict__.items()}
        sec_dict["subsections"] = []
        lookup[sec_dict["section_id"]] = sec_dict

    roots = []
    for sec_dict in lookup.values():
        pid = sec_dict.get("parent_id")
        if pid and pid in lookup:
            lookup[pid]["subsections"].append(sec_dict)
        else:
            roots.append(sec_dict)
    return roots