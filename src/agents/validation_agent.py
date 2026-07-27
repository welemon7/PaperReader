from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, TypedDict

from src.llm.client import LLMClient, LLMError
from src.config import settings
from src.schemas.poster import PosterBlueprint
from src.schemas.validation import PosterValidation, ValidationIssue
from src.storage.sqlite import PaperDatabase

logger = logging.getLogger(__name__)

ValidationState = TypedDict("ValidationState", {
    "arxiv_id": str,
    "blueprint_path": Optional[str],
    "blueprint": Optional[PosterBlueprint],
    "paper_markdown": Optional[str],
    "llm_response": Optional[dict[str, Any]],
    "validation": Optional[PosterValidation],
    "error": Optional[str],
    "validation_prompt": Optional[str],
})


# ---------- Prompt ----------

_SYSTEM_PROMPT = (
    "You are a research poster validation expert. "
    "Compare a research paper with its poster representation and identify issues. "
    "Evaluate on four dimensions: coverage, accuracy, clarity, completeness. "
    "For each issue, specify severity (error/warning/info), "
    "category (missing_contribution/incorrect_method/missing_experiment/formatting/clarity/completeness), "
    "description, location (poster section), and suggestion."
    "\n\nOutput JSON: "
    '{"scores": {"coverage": int, "accuracy": int, "clarity": int, "completeness": int}, '
    '"issues": [{"severity": str, "category": str, "description": str, "location": str, "suggestion": str}], '
    '"summary": str}'
)


def _build_validation_prompt(paper_md: str, bp: PosterBlueprint) -> str:
    parts = ["## Paper Content\n", paper_md[:8000]]  # truncate for token limit
    parts.append("\n\n## Poster Blueprint\n")
    for sec in bp.sections:
        parts.append(f"\n### {sec.title} (row {sec.row}, col {sec.column}, span {sec.col_span})")
        parts.append(sec.content_md[:500])  # per-section truncation
    parts.append("\n\nValidate the poster against the paper.")
    return "\n".join(parts)


def call_llm_node(state: ValidationState) -> dict:
    """Backward-compatible helper used by tests and older callers."""
    prompt = state.get("validation_prompt", "")
    if not prompt:
        return {"error": "No prompt"}
    try:
        if not LLMClient.is_configured():
            return {"error": "LLM API key not configured"}
        client = LLMClient()
        result = client.chat_json(system=_SYSTEM_PROMPT, user=prompt)
        return {"llm_response": result}
    except LLMError as e:
        return {"error": f"LLM call failed: {e}"}


def validate_poster(arxiv_id: str, blueprint_path: str) -> PosterValidation:
    state: ValidationState = {
        "arxiv_id": arxiv_id,
        "blueprint_path": blueprint_path,
        "blueprint": None,
        "paper_markdown": None,
        "llm_response": None,
        "validation": None,
        "error": None,
        "validation_prompt": None,
    }
    try:
        db = PaperDatabase()
        doc = db.get_paper_by_arxiv(arxiv_id)
        db.close()
        if not doc:
            raise RuntimeError(f"Paper {arxiv_id} not found. Run parse first.")
        bp = PosterBlueprint.model_validate_json(Path(blueprint_path).read_text(encoding="utf-8"))
        state["paper_markdown"] = doc.raw_markdown or doc.title
        state["blueprint"] = bp
        state["validation_prompt"] = _build_validation_prompt(state["paper_markdown"] or "", bp)

        if not LLMClient.is_configured():
            raise RuntimeError("LLM API key not configured")
        client = LLMClient()
        llm_resp = client.chat_json(system=_SYSTEM_PROMPT, user=state["validation_prompt"] or "")
        state["llm_response"] = llm_resp
        issues_data = llm_resp.get("issues", [])
        issues = [ValidationIssue(**i) for i in issues_data]
        scores = llm_resp.get("scores", {})
        validation = PosterValidation(
            paper_id=bp.paper_id,
            arxiv_id=arxiv_id,
            scores=scores,
            issues=issues,
            summary=llm_resp.get("summary", ""),
        )
        state["validation"] = validation
        return validation
    except Exception as e:
        raise RuntimeError(f"Validation failed: {e}") from e
