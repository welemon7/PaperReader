from __future__ import annotations

import logging
import re
import shutil
import tarfile
import gzip
import io
from pathlib import Path
from typing import Optional

import httpx
import arxiv

from src.config import settings

logger = logging.getLogger(__name__)


class ArxivDownloader:
    def __init__(self) -> None:
        self.client = arxiv.Client(page_size=1, delay_seconds=3, num_retries=3)
        self.cache_dir = Path(settings.arxiv_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, arxiv_id: str) -> tuple[Path, Path, str]:
        base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        target_dir = self.cache_dir / base_id

        if target_dir.exists():
            main_tex = self._find_main_tex(target_dir)
            if main_tex:
                return target_dir, main_tex, "latex"

        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading LaTeX source for arXiv %s ...", arxiv_id)

        # Query arxiv API for metadata
        search = arxiv.Search(id_list=[arxiv_id])
        paper = None
        for result in self.client.results(search):
            paper = result
            break
        if paper is None:
            raise RuntimeError(f"Paper {arxiv_id} not found on arXiv")

        # Save metadata
        (target_dir / "arxiv_meta.txt").write_text(
            f"Title: {paper.title}\nAuthors: {', '.join(a.name for a in paper.authors)}\nAbstract: {paper.summary}\n",
            encoding="utf-8",
        )

        # Download LaTeX source from e-print endpoint
        tex_content = self._download_eprint(arxiv_id, target_dir)
        if tex_content:
            main_tex_path = target_dir / "main.tex"
            main_tex_path.write_text(tex_content, encoding="utf-8")
            logger.info("LaTeX source downloaded for %s", arxiv_id)
            return target_dir, main_tex_path, "latex"

        # Fallback: download PDF and extract text
        logger.warning("LaTeX source not available for %s, falling back to PDF", arxiv_id)
        pdf_text = self._download_pdf_text(arxiv_id, target_dir)
        md_path = target_dir / "paper.md"
        md_path.write_text(pdf_text, encoding="utf-8")
        logger.info("PDF text extracted for %s (fallback mode)", arxiv_id)
        return target_dir, md_path, "pdf"

    def _download_eprint(self, arxiv_id: str, target_dir: Path) -> Optional[str]:
        try:
            url = f"https://arxiv.org/e-print/{arxiv_id}"
            resp = httpx.get(url, follow_redirects=True, timeout=120)
            resp.raise_for_status()
            data = resp.content

            if data[:100].strip().startswith(b"<!DOCTYPE") or data[:100].strip().startswith(b"<html"):
                logger.warning("e-print endpoint returned HTML (no LaTeX source)")
                return None

            # Try gzip decompression if needed
            try:
                data = gzip.decompress(data)
            except (gzip.BadGzipFile, OSError):
                pass

            # Check if it's a tar archive
            try:
                tar_path = target_dir / "source.tar"
                tar_path.write_bytes(data)
                extract_dir = target_dir / "tex_src"
                extract_dir.mkdir(exist_ok=True)
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=str(extract_dir), filter="data")

                # Find and concatenate all .tex files
                tex_files = sorted(extract_dir.rglob("*.tex"))
                if tex_files:
                    parts = []
                    for tf in tex_files[:30]:
                        try:
                            parts.append(tf.read_text(encoding="utf-8", errors="replace"))
                        except Exception:
                            pass
                    return "\n\n".join(parts)

                return None
            except (tarfile.TarError, Exception):
                pass

            # If not a tar, the data might be a single .tex file (gzip decompressed)
            return data.decode("utf-8", errors="replace")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning("arXiv e-print endpoint returned 403 (source not available)")
            else:
                logger.warning("arXiv e-print download failed: %s", e)
            return None
        except Exception as e:
            logger.exception("e-print download failed")
            return None

    def _download_pdf_text(self, arxiv_id: str, target_dir: Path) -> str:
        try:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            resp = httpx.get(url, follow_redirects=True, timeout=120)
            resp.raise_for_status()
            pdf_path = target_dir / "paper.pdf"
            pdf_path.write_bytes(resp.content)

            try:
                import fitz
                with fitz.open(pdf_path) as doc:
                    pages = [doc[i].get_text("text") for i in range(doc.page_count)]
                    return "\n\n".join(pages)
            except ImportError:
                pass

            return f"[PDF downloaded to {pdf_path} but text extraction unavailable]"
        except Exception as e:
            raise RuntimeError(f"Failed to download PDF for {arxiv_id}: {e}") from e

    @staticmethod
    def _find_main_tex(source_dir: Path) -> Optional[Path]:
        for name in ["main.tex", "ms.tex", "paper.tex", "article.tex"]:
            p = source_dir / "tex_src" / name
            if p.exists():
                return p
            p = source_dir / name
            if p.exists():
                return p

        # Fallback: file containing documentclass
        tex_src = source_dir / "tex_src"
        if tex_src.exists():
            for tex in sorted(tex_src.rglob("*.tex")):
                content = tex.read_text(encoding="utf-8", errors="replace")
                if "\\documentclass" in content:
                    return tex

        return None

    def cleanup(self, source_dir: Path) -> None:
        if source_dir.exists():
            shutil.rmtree(source_dir)

    @staticmethod
    def extract_arxiv_id(raw: str) -> Optional[str]:
        patterns = [
            r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
            r"ar[xX]iv:(\d{4}\.\d{4,5}(?:v\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, raw)
            if m:
                return m.group(1).rstrip("/")
        return None