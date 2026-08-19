from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from typing import TypedDict

try:
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    END = "__end__"
    StateGraph = None

from src.llm.client import LLMClient, LLMError
from src.schemas.analysis import (
    Contribution,
    ExperimentSummary,
    KeyFigure,
    KeyFormula,
    SelectedTable,
    FinalTable,
    PaperAnalysis,
)
from src.schemas.paper import PaperDocument
from src.storage.sqlite import PaperDatabase

logger = logging.getLogger(__name__)

MAX_FINAL_TABLE_ROWS = 8
MAX_FINAL_TABLE_COLUMNS = 6
MAX_FINAL_CELL_LENGTH = 36

_CODE_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|huggingface\.co)/[^\s\)\]\}<>\"']+",
    re.IGNORECASE,
)

_URL_PATTERN = re.compile(r"https?://[^\s\)\]\}<>\"']+", re.IGNORECASE)


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:")


def _extract_code_url(doc: PaperDocument) -> str:
    """Best-effort deterministic code-link extraction from paper text."""
    text_parts = [doc.abstract or "", doc.raw_markdown or ""]
    for sec in doc.sections:
        text_parts.append(sec.raw_latex or "")
        text_parts.append(sec.text or "")
    text = "\n".join(text_parts)

    for match in _CODE_URL_PATTERN.findall(text):
        return _clean_url(match)

    # Fall back to any URL mentioned near code/project keywords.
    lowered = text.lower()
    for keyword in ("code", "project", "github", "gitlab", "bitbucket", "repository", "repo"):
        idx = lowered.find(keyword)
        if idx < 0:
            continue
        window = text[max(0, idx - 220): idx + 420]
        urls = [_clean_url(u) for u in _URL_PATTERN.findall(window)]
        for url in urls:
            if any(host in url.lower() for host in
                   ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org", "huggingface.co")):
                return url
        if urls:
            return urls[0]
    return ""


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
        return {"error": "LLM API key not configured. Set LLM_API_KEY in .env"}
    try:
        client = LLMClient()
        result = client.chat_json(system=_SYSTEM_PROMPT, user=prompt)
        # ✅ 修复 LLM 响应格式问题
        result = _fix_llm_response(result)
        if not isinstance(result, dict) or not result:
            return {"error": "LLM returned an empty/invalid response"}
        try:
            result["final_tables"] = _run_final_table_review(client, state.get("paper_document"), result)
        except Exception as exc:
            logger.warning("Final table review skipped; tables will not be rendered: %s", exc)
            result["final_tables"] = []
        logger.info("LLM response received (%d keys)", len(result))
        return {"llm_response": result}
    except LLMError as e:
        return {"error": f"LLM call failed: {e}"}
    except Exception as e:
        return {"error": f"Unexpected LLM error: {e}"}


def _run_final_table_review(client: LLMClient, doc: PaperDocument | None, first_response: dict) -> list[dict]:
    """Ask the LLM to compress candidate method rows into the final display table."""
    if not doc or not doc.tables:
        return []
    candidates = []
    table_map = {table.table_id: table for table in doc.tables}
    for item in _safe_parse_list(first_response.get("selected_tables", []), SelectedTable):
        table = table_map.get(item.table_id)
        if not table:
            continue
        rows = [
            {"row_index": index, "group": (table.row_groups[index] if index < len(table.row_groups) else ""), "cells": table.rows[index]}
            for index in item.row_indices[:MAX_FINAL_TABLE_ROWS]
            if 0 <= index < len(table.rows)
        ]
        if rows:
            candidates.append({
                "table_id": table.table_id,
                "caption": table.caption,
                "headers": table.headers,
                "row_groups": table.row_groups,
                "rows": rows,
            })
    if not candidates:
        return []
    prompt = (
        "Create the final compact but semantically complete HTML table data from the candidate rows below. "
        "Keep only the paper's own method results. Drop baseline rows, comparison rows, and any method not proposed by this paper. "
        "Never invent, average, or rewrite any value. Return JSON only with final_tables. "
        f"Use at most {MAX_FINAL_TABLE_ROWS} rows, {MAX_FINAL_TABLE_COLUMNS} displayed columns, and {MAX_FINAL_CELL_LENGTH} characters per cell. "
        "Preserve proposed-method variants and settings, including Ours-Small/Ours-Large and dataset/setting rows. "
        "Always preserve identity columns such as Dataset, Setting, Resolution, Method, Variant, Params. "
        "Only compress numeric metric columns: put their source column indices into one column_groups item, and display their values joined by ' / '. "
        "For every compressed group, the displayed header must name the metric group and list the source metrics in order. "
        "Do not compress identity/text columns. Each final table must include table_id, row_indices, column_groups, headers, rows, and optional datasets, metrics, row_groups, caption, notes. "
        "The row/column values must be copied from candidates; headers may only clarify source headers.\n\n"
        + json.dumps(candidates, ensure_ascii=False)
    )
    response = client.chat_json(
        system="You are a strict table editor. Output valid JSON only. Preserve source values exactly.",
        user=prompt,
    )
    return _validate_final_tables(doc, candidates, response.get("final_tables", []))


def _validate_final_tables(doc: PaperDocument, candidates: list[dict], raw_tables) -> list[dict]:
    candidate_map = {item["table_id"]: item for item in candidates}
    final_tables = []
    for raw in raw_tables if isinstance(raw_tables, list) else []:
        if not isinstance(raw, dict):
            continue
        source = candidate_map.get(raw.get("table_id"))
        if not source:
            continue
        row_indices = [int(i) for i in raw.get("row_indices", []) if str(i).lstrip("-").isdigit()]
        column_indices = [int(i) for i in raw.get("column_indices", []) if str(i).lstrip("-").isdigit()]
        raw_groups = raw.get("column_groups", [])
        row_indices = [i for i in row_indices if any(row["row_index"] == i for row in source["rows"])][:MAX_FINAL_TABLE_ROWS]
        column_indices = [i for i in column_indices if 0 <= i < len(source["headers"])]
        groups: list[list[int]] = []
        for group in raw_groups if isinstance(raw_groups, list) else []:
            if not isinstance(group, list):
                continue
            valid = [int(i) for i in group if str(i).lstrip("-").isdigit() and 0 <= int(i) < len(source["headers"])]
            if valid and valid not in groups:
                groups.append(valid)
        if not groups:
            groups = [[index] for index in column_indices]
        # Preserve identity columns when an over-aggressive LLM omits them.
        identity_tokens = ("setting", "dataset", "resolution", "method", "variant", "param", "model")
        for index, header in enumerate(source["headers"]):
            if any(token in header.lower() for token in identity_tokens) and [index] not in groups:
                groups.insert(0, [index])
        groups = groups[:MAX_FINAL_TABLE_COLUMNS]
        if not row_indices or not groups:
            continue
        column_indices = [index for group in groups for index in group]
        headers = []
        for group in groups:
            names = [source["headers"][i] for i in group]
            headers.append(names[0] if len(names) == 1 else " / ".join(names))
        source_rows = {row["row_index"]: row["cells"] for row in source["rows"]}
        rows = [[" / ".join(source_rows[index][i] for i in group)[:MAX_FINAL_CELL_LENGTH] for group in groups] for index in row_indices]
        datasets = _safe_string_list(raw.get("datasets", [])) or _infer_table_datasets(headers, rows, source.get("row_groups", []))
        metrics = _safe_string_list(raw.get("metrics", [])) or _infer_table_metrics(headers)
        row_groups = _safe_string_list(raw.get("row_groups", []))
        if not row_groups:
            source_groups = source.get("row_groups", []) or []
            row_groups = [source_groups[index] for index in row_indices if 0 <= index < len(source_groups) and source_groups[index]]
        final_tables.append({
            "table_id": source["table_id"],
            "caption": source.get("caption", ""),
            "headers": headers,
            "rows": rows,
            "row_indices": row_indices,
            "column_indices": column_indices,
            "column_groups": groups,
            "datasets": datasets,
            "metrics": metrics,
            "row_groups": row_groups,
            "notes": _safe_string(raw.get("notes", "")),
        })
    return final_tables[:3]


def _safe_string(value) -> str:
    return str(value).strip() if value is not None else ""


def _safe_string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _infer_table_datasets(headers: list[str], rows: list[list[str]], row_groups: list[str]) -> list[str]:
    candidates: list[str] = []
    for idx, header in enumerate(headers):
        lowered = header.lower()
        if any(token in lowered for token in ("dataset", "data", "benchmark", "split", "setting")):
            for row in rows:
                if idx < len(row) and row[idx]:
                    candidates.append(row[idx])
    if not candidates:
        for group in row_groups:
            if group:
                candidates.append(group)
    return list(dict.fromkeys(candidates))[:MAX_FINAL_TABLE_ROWS]


def _infer_table_metrics(headers: list[str]) -> list[str]:
    metrics = []
    for header in headers:
        lowered = header.lower()
        if any(token in lowered for token in ("psnr", "ssim", "acc", "accuracy", "f1", "iou", "dice", "mse", "mae", "score", "metric")):
            metrics.append(header)
    return list(dict.fromkeys(metrics))[:MAX_FINAL_TABLE_COLUMNS]


def validate_node(state: UnderstandState) -> dict:
    doc = state.get("paper_document")
    llm_resp = state.get("llm_response")
    if not doc or not llm_resp:
        # 保留上游节点设置的真实错误，避免用笼统信息掩盖根因
        return {"error": state.get("error") or "Missing paper document or LLM response"}
    try:
        analysis = _parse_analysis(doc, llm_resp)
        return {"paper_analysis": analysis}
    except Exception as e:
        logger.exception("Validation failed: %s", e)
        return {"error": f"Validation failed: {e}"}


def store_analysis_node(state: UnderstandState) -> dict:
    analysis = state.get("paper_analysis")
    if not analysis:
        # Preserve the original failure reason from validation or LLM calls.
        return {"error": state.get("error") or "No analysis to store"}
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
    "\n- title_zh: Chinese translation of the paper title for internal reference only"
    "\n- problem_statement: The core problem this paper solves, in English, using wording close to the paper (1-2 sentences)"
    "\n- contributions: List of contributions (3-5 items), each with:"
    "\n  - text: contribution description, at most 15 words, one short claim"
    "\n  - category: method/theory/system/dataset/application/other"
    "\n- method_overview: High-level method description, in English and faithful to the paper (at most 2 sentences)"
    "\n- key_formulas: List of most important formulas (max 5), each with:"
    "\n  - formula_id: the formula ID from the paper"
    "\n  - latex: the LaTeX source"
    "\n  - semantic_desc: plain-language meaning, at most 12 words"
    "\n- key_figures: List of most important figures (max 4), each with:"
    "\n  - figure_id: the figure ID or label"
    "\n  - caption: the figure caption"
    "\n  - role: what this figure illustrates (overview/architecture/result)"
    "\n- selected_tables: List of tables useful for explaining ONLY the proposed method (never baselines/other methods), max 3. Each item has table_id, role, and row_indices (zero-based indices into data rows). Select ALL relevant rows for the paper method, including every method variant (e.g. Ours-Small/Ours-Large) and every important setting/dataset; do not keep only the best row."
    "\n- experiments: Object with datasets (list of strings), metrics (list of strings), main_results (ONE short sentence), takeaways (at most 3 short items)"
    "\n- conclusion: Summary of the paper conclusion (at most 2 sentences)"
    "\n- code_url: Project code repository URL extracted from the paper body (e.g. GitHub link). Leave empty string if not found."
    "\n- full_analysis_md: Complete markdown analysis of the paper"
    "\n\nAll narrative fields that will appear in the poster must be in English only. Do not mix Chinese into problem_statement, contributions, method_overview, key_figures, experiments, or conclusion. Be precise, concise, and faithful to the source text. Poster copy must stay short: contributions <= 15 words each, method_overview <= 2 sentences, main_results 1 sentence."
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
    if doc.tables:
        parts.append("\n## Table Index (select IDs only; values must remain source-grounded)")
        for table in doc.tables:
            parts.append(json.dumps({
                "table_id": table.table_id,
                "caption": table.caption,
                "label": table.label,
                "headers": table.headers,
                "rows": table.rows,
                "row_groups": table.row_groups,
            }, ensure_ascii=False))
    if doc.references:
        parts.append("\n## References")
        for ref in doc.references[:20]:
            parts.append(f"- {ref.title} ({ref.year or 'n.d.'})")
    return "\n".join(parts)


# ✅ 新增: 修复 LLM 响应格式
def _fix_llm_response(response: dict) -> dict:
    """修复 LLM 响应中的常见格式问题"""
    if not isinstance(response, dict):
        return response

    # 复制一份避免修改原始数据
    fixed = response.copy()

    # 修复 experiments 字段
    if "experiments" in fixed and isinstance(fixed["experiments"], dict):
        exp = fixed["experiments"].copy()

        # main_results: 列表 → 字符串
        if "main_results" in exp:
            if isinstance(exp["main_results"], list):
                # 将列表元素连接成字符串
                exp["main_results"] = " ".join(
                    str(item) for item in exp["main_results"] if item
                )
            elif exp["main_results"] is None:
                exp["main_results"] = ""

        # takeaways: 字符串 → 列表
        if "takeaways" in exp:
            if isinstance(exp["takeaways"], str):
                exp["takeaways"] = [exp["takeaways"]] if exp["takeaways"] else []
            elif exp["takeaways"] is None:
                exp["takeaways"] = []

        # datasets: 字符串 → 列表
        if "datasets" in exp:
            if isinstance(exp["datasets"], str):
                exp["datasets"] = [exp["datasets"]] if exp["datasets"] else []
            elif exp["datasets"] is None:
                exp["datasets"] = []

        # metrics: 字符串 → 列表
        if "metrics" in exp:
            if isinstance(exp["metrics"], str):
                exp["metrics"] = [exp["metrics"]] if exp["metrics"] else []
            elif exp["metrics"] is None:
                exp["metrics"] = []

        fixed["experiments"] = exp

    return fixed


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
    code_url = llm_resp.get("code_url", "") or _extract_code_url(doc)
    valid_table_ids = {table.table_id for table in getattr(doc, "tables", [])}
    table_by_id = {table.table_id: table for table in getattr(doc, "tables", [])}
    selected_tables = []
    for item in _safe_parse_list(llm_resp.get("selected_tables", []), SelectedTable):
        table = table_by_id.get(item.table_id)
        if not table:
            continue
        item.row_indices = sorted({index for index in item.row_indices if 0 <= index < len(table.rows)})
        if item.row_indices:
            selected_tables.append(item)
    selected_tables = selected_tables[:3]
    final_tables = _safe_parse_list(llm_resp.get("final_tables", []), FinalTable)

    # 处理 experiments，确保格式正确
    experiments = None
    if isinstance(llm_resp.get("experiments"), dict):
        exp_data = llm_resp["experiments"].copy()

        # 确保 main_results 是字符串
        if "main_results" in exp_data:
            if isinstance(exp_data["main_results"], list):
                exp_data["main_results"] = " ".join(
                    str(item) for item in exp_data["main_results"] if item
                )
            elif exp_data["main_results"] is None:
                exp_data["main_results"] = ""

        # 确保 takeaways 是列表
        if "takeaways" in exp_data:
            if isinstance(exp_data["takeaways"], str):
                exp_data["takeaways"] = [exp_data["takeaways"]] if exp_data["takeaways"] else []
            elif exp_data["takeaways"] is None:
                exp_data["takeaways"] = []

        # 确保 datasets 是列表
        if "datasets" in exp_data:
            if isinstance(exp_data["datasets"], str):
                exp_data["datasets"] = [exp_data["datasets"]] if exp_data["datasets"] else []
            elif exp_data["datasets"] is None:
                exp_data["datasets"] = []

        # 确保 metrics 是列表
        if "metrics" in exp_data:
            if isinstance(exp_data["metrics"], str):
                exp_data["metrics"] = [exp_data["metrics"]] if exp_data["metrics"] else []
            elif exp_data["metrics"] is None:
                exp_data["metrics"] = []

        try:
            experiments = ExperimentSummary(**exp_data)
        except Exception as e:
            logger.warning(f"Failed to parse experiments: {e}, data: {exp_data}")
            experiments = None

    return PaperAnalysis(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        title_zh=llm_resp.get("title_zh", ""),
        problem_statement=llm_resp.get("problem_statement", ""),
        contributions=_safe_parse_list(llm_resp.get("contributions", []), Contribution),
        method_overview=llm_resp.get("method_overview", ""),
        key_formulas=_safe_parse_list(llm_resp.get("key_formulas", []), KeyFormula),
        key_figures=_safe_parse_list(llm_resp.get("key_figures", []), KeyFigure),
        selected_tables=selected_tables,
        final_tables=final_tables,
        experiments=experiments,
        conclusion=llm_resp.get("conclusion", ""),
        code_url=code_url,
        full_analysis_md=llm_resp.get("full_analysis_md", ""),
    )


def build_understand_graph():
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed")
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
        try:
            _compiled_graph = build_understand_graph()
        except RuntimeError:
            _compiled_graph = None

    if _compiled_graph is None:
        state: UnderstandState = {
            "arxiv_id": arxiv_id,
            "paper_document": None,
            "analysis_prompt": None,
            "llm_response": None,
            "paper_analysis": None,
            "error": None,
        }
        for node in (load_paper_node, build_prompt_node, call_llm_node, validate_node, store_analysis_node):
            state.update(node(state))
            if state.get("error"):
                raise RuntimeError(state["error"])
        return state["paper_analysis"]
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
