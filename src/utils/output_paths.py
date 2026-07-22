from __future__ import annotations

import re
from pathlib import Path


def sanitize_output_name(name: str, fallback: str = "paper") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name or fallback).strip("_")
    return cleaned or fallback


def resolve_paper_output_dir(base_dir: str | Path, arxiv_id: str) -> Path:
    """Return the per-paper output directory under the shared output root."""
    base_path = Path(base_dir)
    paper_name = sanitize_output_name(arxiv_id)
    if base_path.name == paper_name:
        paper_dir = base_path
    else:
        paper_dir = base_path / paper_name
    paper_dir.mkdir(parents=True, exist_ok=True)
    return paper_dir
