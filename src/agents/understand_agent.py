from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from typing import TypedDict

from src.llm.client import LLMClient, LLMError
from src.schemas.analysis import (
    Contribution,
    ExperimentSummary,
    KeyFigure,
    KeyFormula,
    PaperAnalysis,
)
from src.schemas.paper import PaperDocument
from src.storage.sqlite import PaperDatabase

logger = logging.getLogger(__name__)

class UnderstandState(TypedDict):
    arxiv_id: str
    paper_document: Optional[PaperDocument]
    analysis_prompt: Optional[str]
    llm_response: Optional[dict[str, Any]]
    paper_analysis: Optional[PaperAnalysis]
    error: Optional[str]

def load_paper_node(state: UnderstandState) -> dict:
    arxiv_id = state.get("arxiv_id", "")
    if not arxiv_id:
        return {"error": "No arxiv_id provided"}
    try:
        db = PaperDatabase()
        doc = db.get_paper_by_arxiv(arxiv_id)
        db.close()
        if doc is None:
            return {"error": f"Paper {arxiv_id} not found. Run parse first."}
        logger.info("Loaded paper: %s", doc.title)
        return {"paper_document": doc}
    except Exception as e:
        logger.exception("Failed to load paper")
        return {"error": f"Load failed: {e}"}

def build_prompt_node(state: UnderstandState) -> dict:
    doc = state.get("paper_document")
    if not doc:
        return {"error": "No paper document"}
    try:
        prompt = _build_analysis_prompt(doc)
        return {"analysis_prompt": prompt}
    except Exception as e:
        return {"error": f"Prompt build failed: {e}"}

def call_llm_node(state: UnderstandState) -> dict:
    prompt = state.get("analysis_prompt")
    if not prompt:
        return {"error": "No analysis prompt"}
    if not LLMClient.is_configured():
        return {"error": "LLM API key not configured. Set OPENAI_API_KEY in .env"}
    try:
        client = LLMClient()
        result = client.chat_json(system=_SYSTEM_PROMPT, user=prompt)
        logger.info("LLM response received (%d keys)", len(result))
        return {"llm_response": result}
    except LLMError as e:
        return {"error": f"LLM call failed: {e}"}
    except Exception as e:
        return {"error": f"Unexpected LLM error: {e}"}

def validate_node(state: UnderstandState) -> dict:
    doc = state.get("paper_document")
    llm_resp = state.get("llm_response")
    if not doc or not llm_resp:
        return {"error": "Missing paper document or LLM response"}
    try:
        analysis = _parse_analysis(doc, llm_resp)
        return {"paper_analysis": analysis}
    except Exception as e:
        logger.exception("Validation failed: %s", e)
        return {"error": f"Validation failed: {e}"}

def store_analysis_node(state: UnderstandState) -> dict:
    analysis = state.get("paper_analysis")
    if not analysis:
        return {"error": "No analysis to store"}
    try:
        db = PaperDatabase()
        db.save_analysis(analysis)
        db.close()
        logger.info("Analysis stored for %s", analysis.arxiv_id)
        return {}
    except Exception as e:
        return {"error": f"Store failed: {e}"}

_SYSTEM_PROMPT = (
    "You are a research paper analysis expert. "
    "Your task is to analyze a computer science paper and produce "
    "a structured analysis suitable for creating a research poster."
    "\n\n"
    "Analyze the paper thoroughly and output a JSON object "
    "with the following fields:"
    "\n- title_zh: Chinese translation of the paper title"
    "\n- problem_statement: The core problem this paper solves (1-2 sentences)"
    "\n- contributions: List of contributions, each with:"
    "\n  - text: contribution description"
    "\n  - category: method/theory/system/dataset/application/other"
    "\n- method_overview: High-level method description (2-4 sentences)"
    "\n- key_formulas: List of most important formulas (max 5), each with:"
    "\n  - formula_id: the formula ID from the paper"
    "\n  - latex: the LaTeX source"
    "\n  - semantic_desc: plain-language meaning"
    "\n- key_figures: List of most important figures (max 4), each with:"
    "\n  - figure_id: the figure ID"
    "\n  - caption: the figure caption"
    "\n  - role: what this figure illustrates"
    "\n- experiments: Object with datasets, metrics, main_results, takeaways"
    "\n- conclusion: Summary of the paper conclusion"
    "\n- full_analysis_md: Complete markdown analysis of the paper"
    "\n\nBe precise and concise."
)

def _build_analysis_prompt(doc: PaperDocument) -> str:
    parts = []
    parts.append(f"# {doc.title}")
    if doc.authors:
        names = [a.name for a in doc.authors]
        parts.append(f"Authors: {'; '.join(names)}")
    parts.append("\n## Abstract")
    parts.append(doc.abstract or "(no abstract)")
    for sec in doc.sections:
        level = sec.level + 2
        parts.append(f"\n{'#' * level} {sec.title}")
        parts.append(sec.raw_latex or "(empty)")
    if doc.formulas:
        parts.append("\n## Formula Index")
        for f in doc.formulas:
            label = f.label or "(no label)"
            parts.append(f"- [{f.formula_id}] {label}: `{f.latex[:120]}`")
    if doc.figures:
        parts.append("\n## Figure Index")
        for fig in doc.figures:
            label = fig.label or "(no label)"
            caption = fig.caption or "(no caption)"
            parts.append(f"- [{fig.figure_id}] {label}: {caption}")
    if doc.references:
        parts.append("\n## References")
        for ref in doc.references[:20]:
            parts.append(f"- {ref.title} ({ref.year or 'n.d.'})")
    return "\n".join(parts)



def _safe_get(obj, key, default):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def _safe_parse_list(items, model_cls):
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if isinstance(item, dict):
            try:
                result.append(model_cls(**item))
            except Exception:
                pass
    return result
def _parse_analysis(doc: PaperDocument, llm_resp: dict) -> PaperAnalysis:
    return PaperAnalysis(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        title_zh=llm_resp.get("title_zh", ""),
        problem_statement=llm_resp.get("problem_statement", ""),
        contributions=_safe_parse_list(llm_resp.get("contributions", []), Contribution),
        method_overview=llm_resp.get("method_overview", ""),
        key_formulas=_safe_parse_list(llm_resp.get("key_formulas", []), KeyFormula),
        key_figures=_safe_parse_list(llm_resp.get("key_figures", []), KeyFigure),
        experiments=(
            ExperimentSummary(**llm_resp["experiments"])
            if isinstance(llm_resp.get("experiments"), dict)
            else None
        ),
        conclusion=llm_resp.get("conclusion", ""),
        full_analysis_md=llm_resp.get("full_analysis_md", ""),
    )

def build_understand_graph():
    workflow = StateGraph(UnderstandState)
    workflow.add_node("load_paper", load_paper_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("call_llm", call_llm_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("store", store_analysis_node)
    workflow.set_entry_point("load_paper")
    workflow.add_edge("load_paper", "build_prompt")
    workflow.add_edge("build_prompt", "call_llm")
    workflow.add_edge("call_llm", "validate")
    workflow.add_edge("validate", "store")
    workflow.add_edge("store", END)
    return workflow.compile()

_compiled_graph = None

def run_understand_paper(arxiv_id: str) -> PaperAnalysis:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_understand_graph()
    initial_state: UnderstandState = {
        "arxiv_id": arxiv_id,
        "paper_document": None,
        "analysis_prompt": None,
        "llm_response": None,
        "paper_analysis": None,
        "error": None,
    }
    result = _compiled_graph.invoke(initial_state)
    if result.get("error"):
        raise RuntimeError(f"Understanding failed: {result['error']}")
    analysis = result.get("paper_analysis")
    if not analysis:
        raise RuntimeError("No analysis produced")
    return analysis
