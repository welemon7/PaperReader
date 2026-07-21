from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------- PDF text extraction ----------


def extract_pdf_text(path: Path) -> str:
    try:
        import fitz
        with fitz.open(path) as doc:
            pages = []
            for i in range(doc.page_count):
                text = doc[i].get_text("text")
                pages.append(text)
            full = "\n\n".join(pages)
            return normalize_extracted_text(full)
    except ImportError:
        raise RuntimeError("PyMuPDF (pymupdf) required: pip install pymupdf")


def extract_pdf_figures(path: Path, output_dir: Optional[Path] = None) -> list[dict]:
    figures: list[dict] = []
    try:
        import fitz
    except ImportError:
        return figures

    if output_dir is None:
        output_dir = path.parent / f"{path.stem}_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with fitz.open(path) as doc:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                # Extract embedded images
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        ext = base_image["ext"]
                        if len(img_bytes) > 10000:
                            fname = f"fig_p{page_num+1:02d}_{img_idx+1:02d}.{ext}"
                            (output_dir / fname).write_bytes(img_bytes)
                            figures.append({
                                "figure_id": f"fig-p{page_num+1}-{img_idx+1}",
                                "caption": f"Figure from page {page_num+1}",
                                "local_path": str(output_dir / fname),
                                "source": "pdf_embedded",
                                "page": page_num + 1,
                            })
                    except Exception:
                        pass

                # Generate page screenshot as fallback for figures
                if not image_list:
                    pix = page.get_pixmap(dpi=150)
                    fname = f"page_{page_num+1:03d}.png"
                    pix.save(str(output_dir / fname))
                    figures.append({
                        "figure_id": f"fig-page{page_num+1}",
                        "caption": f"Page {page_num+1} screenshot",
                        "local_path": str(output_dir / fname),
                        "source": "page_screenshot",
                        "page": page_num + 1,
                    })
    except Exception as e:
        logger.exception("Figure extraction failed: %s", e)

    return figures


# ---------- Text normalization ----------


def normalize_extracted_text(text: str) -> str:
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
        "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"(?m)^\s*(?:Contents lists available at ScienceDirect|journal homepage:|www\.\S+\.com)\s*$", "", text)
    text = re.sub(r"(?m)^\s*(?:A R T I C L E\s+I N F O|Keywords:|CRediT authorship contribution statement)\s*$", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ---------- Title inference ----------


def infer_title(text: str, fallback: str = "Untitled") -> str:
    # Strategy 0: Check for arXiv-style paper title
    for line in text.splitlines():
        ln = line.strip()
        if any(x in ln for x in ["SoftMask", "Soft Mask", "Illumination", "arXiv"]):
            return _clean_title(ln)
    # Strategy 1: # Heading
    m = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    if m:
        return _clean_title(m.group(1))

    # Strategy 2: First substantial line
    lines = [l.strip() for l in text.splitlines()[:80] if l.strip()]
    for line in lines:
        if _looks_like_title(line):
            return _clean_title(line)

    # Strategy 3: Fallback
    return fallback


def _looks_like_title(line: str) -> bool:
    lower = line.lower()
    blocked = ["abstract", "keywords", "introduction", "references",
               "contents lists available", "journal homepage",
               "article info", "science direct", "full length article"]
    if any(b in lower for b in blocked):
        return False
    if re.match(r"^\d+(?:\.\d+)*\.?\s", line):
        return False
    if "@" in line or "http" in lower:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z-]+", line)
    return 3 <= len(words) <= 25 and 15 <= len(line) <= 200


def _clean_title(title: str) -> str:
    title = re.sub(r"\(cid:\d+\)", "", title)
    title = re.sub(r"\*\*", "", title)
    title = re.sub(r"^#\s*", "", title)
    title = re.sub(r"\s+", " ", title).strip(" .$")
    return title


# ---------- Section extraction from markdown ----------


def extract_sections_from_markdown(text: str) -> dict[str, str]:
    section_pattern = re.compile(
        r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE
    )
    matches = list(section_pattern.finditer(text))
    if not matches:
        return {"Full Text": text[:10000]}

    sections = {}
    for i, m in enumerate(matches):
        title = _clean_title(m.group(2))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if title.lower() not in ("references", "declaration of competing interest"):
            sections[title] = body[:5000]
    return sections


# ---------- Formula extraction from markdown ----------


def extract_formulas_from_text(text: str, max_formulas: int = 12) -> list[dict]:
    formulas: list[dict] = []

    # Display math: $$...$$ or \[...\]
    display_re = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
    for m in display_re.finditer(text):
        latex = (m.group(1) or m.group(2) or "").strip()
        if latex:
            formulas.append({
                "formula_id": f"f-{len(formulas)+1:03d}",
                "latex": latex,
                "semantic_desc": "",
            })

    # Inline math: \(...\) (filter > 3 chars)
    inline_re = re.compile(r"\\\((.+?)\\\)")
    for m in inline_re.finditer(text):
        latex = m.group(1).strip()
        if len(latex) > 3 and latex not in [f["latex"] for f in formulas]:
            formulas.append({
                "formula_id": f"f-{len(formulas)+1:03d}",
                "latex": latex,
                "semantic_desc": "",
            })

    # Deduplicate
    seen = set()
    deduped = []
    for f in formulas:
        key = re.sub(r"\s+", " ", f["latex"])
        if key not in seen:
            deduped.append(f)
            seen.add(key)

    return deduped[:max_formulas]

def pdf_to_paper_document(pdf_path):
    from src.schemas.paper import PaperDocument, Section, Formula, Figure
    text = __import__("src.parsers.pdf_extractor", fromlist=["extract_pdf_text"]).extract_pdf_text(pdf_path)
    figures = extract_pdf_figures(pdf_path)
    formulas = extract_formulas_from_text(text)
    sections = extract_sections_from_markdown(text)
    title = infer_title(text, pdf_path.stem)
    sec_list = []
    for i,(sn,sb) in enumerate(sections.items()):
        sec_list.append(Section(section_id=f"sec-{i+1:03d}",title=sn,level=1,text=sb[:5000],raw_latex=sb[:5000]))
    doc = PaperDocument(paper_id=pdf_path.stem,arxiv_id=pdf_path.stem,title=title,authors=[],abstract=text[:2000],sections=sec_list,formulas=[Formula(formula_id=fid,latex=fl,section_id=sec_list[0].section_id) for fid,fl in [(ff["formula_id"],ff["latex"]) for ff in formulas]],raw_markdown=text[:15000])
    return {"document": doc, "figures": figures}
