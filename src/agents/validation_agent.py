from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.llm.client import LLMClient, LLMError
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


# ---------- Nodes ----------


def load_paper_node(state: ValidationState) -> dict:
    arxiv_id = state.get("arxiv_id", "")
    if not arxiv_id:
        return {"error": "No arxiv_id"}
    try:
        db = PaperDatabase()
        doc = db.get_paper_by_arxiv(arxiv_id)
        db.close()
        if not doc:
            return {"error": f"Paper {arxiv_id} not found. Run parse first."}
        return {"paper_markdown": doc.raw_markdown or doc.title}
    except Exception as e:
        return {"error": f"Load paper failed: {e}"}


def load_blueprint_node(state: ValidationState) -> dict:
    bp_path = state.get("blueprint_path")
    if not bp_path:
        return {"error": "No blueprint path"}
    try:
        bp = PosterBlueprint.model_validate_json(Path(bp_path).read_text(encoding="utf-8"))
        return {"blueprint": bp}
    except Exception as e:
        return {"error": f"Load blueprint failed: {e}"}


def build_prompt_node(state: ValidationState) -> dict:
    md = state.get("paper_markdown")
    bp = state.get("blueprint")
    if not md or not bp:
        return {"error": "Missing paper or blueprint"}
    prompt = _build_validation_prompt(md, bp)
    return {"validation_prompt": prompt}


def call_llm_node(state: ValidationState) -> dict:
    prompt = state.get("validation_prompt", "")
    if not prompt:
        return {"error": "No prompt"}
    if not LLMClient.is_configured():
        return {"error": "LLM API key not configured"}
    try:
        client = LLMClient()
        result = client.chat_json(system=_SYSTEM_PROMPT, user=prompt)
        return {"llm_response": result}
    except LLMError as e:
        return {"error": f"LLM call failed: {e}"}


def parse_validation_node(state: ValidationState) -> dict:
    llm_resp = state.get("llm_response")
    arxiv_id = state.get("arxiv_id", "")
    blueprint = state.get("blueprint")
    if not llm_resp or not blueprint:
        return {"error": "Missing LLM response or blueprint"}
    try:
        issues_data = llm_resp.get("issues", [])
        issues = [ValidationIssue(**i) for i in issues_data]
        scores = llm_resp.get("scores", {})
        validation = PosterValidation(
            paper_id=blueprint.paper_id,
            arxiv_id=arxiv_id,
            scores=scores,
            issues=issues,
            summary=llm_resp.get("summary", ""),
        )
        return {"validation": validation}
    except Exception as e:
        return {"error": f"Parse failed: {e}"}


# ---------- Graph ----------


def build_validation_graph():
    workflow = StateGraph(ValidationState)
    workflow.add_node("load_paper", load_paper_node)
    workflow.add_node("load_blueprint", load_blueprint_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("call_llm", call_llm_node)
    workflow.add_node("parse", parse_validation_node)
    workflow.set_entry_point("load_paper")
    workflow.add_edge("load_paper", "load_blueprint")
    workflow.add_edge("load_blueprint", "build_prompt")
    workflow.add_edge("build_prompt", "call_llm")
    workflow.add_edge("call_llm", "parse")
    workflow.add_edge("parse", END)
    return workflow.compile()


_compiled_graph = None


def validate_poster(arxiv_id: str, blueprint_path: str) -> PosterValidation:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_validation_graph()
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
    result = _compiled_graph.invoke(state)
    if result.get("error"):
        raise RuntimeError(f"Validation failed: {result['error']}")
    v = result.get("validation")
    if not v:
        raise RuntimeError("No validation result")
    return v