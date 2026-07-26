from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional
import re
from src.config import settings
from src.llm.client import LLMClient, LLMError
from src.llm.multimodal_client import capture_poster, multimodal_analyze
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import FigurePlacement, FormulaDisplay, PosterBlueprint, PosterSection
from src.schemas.poster_v2 import LayoutConstraints, LayoutNode, LayoutTree, PosterComment, PosterQAEval, PosterReview, EvaluationQuestion
from src.renderers.html_renderer import HtmlPosterRenderer
from src.storage.sqlite import PaperDatabase
from src.utils.output_paths import resolve_paper_output_dir
from src.agents.poster_planner import (
    _augment_key_formulas,
    _default_colors,
    _drop_top_summary_sections,
    _format_authors,
    _tighten_layout,
    generate_blueprint,
    normalize_analysis_for_poster,
)

logger = logging.getLogger(__name__)

_TREE_SYSTEM_PROMPT = (
    "You are a layout planner for scientific posters. "
    "Produce a hierarchical layout tree that decides what must appear, where figures go, "
    "the reading order, and the space budget of each region. "
    "Return JSON with required_items, reading_path, layout_notes, and nodes. "
    "Each node must contain node_id, node_type, title, content_md, figure_ids, child_ids, reading_order, space_ratio, constraints, and notes."
)

_COMMENT_SYSTEM_PROMPT = (
    "You are a strict scientific poster reviewer. Inspect the rendered poster image. "
    "Identify text overflow, tiny figures, dense columns, weak hierarchy, and missing information. "
    "Return json with quality_score, needs_improvement, layout_feedback, summary, and issues. "
    "For each issue, severity must be one of error, warning, or info."
)

_QA_SYSTEM_PROMPT = (
    "You are a scientific poster evaluator. Create paper-understanding questions from the paper, "
    "then answer them using only the poster content. Return JSON with questions, poster_answers, "
    "correct_count, total_count, accuracy, coverage, recall, visual_score, qa_score, and summary."
)


def _poster_vision_provider() -> str:
    provider = (settings.poster_vision_provider or "agnes").lower()
    return provider if provider in {"agnes", "gemini", "openai"} else "agnes"


def _normalize_severity(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"error", "warning", "info"}:
        return text
    if text in {"high", "critical", "severe", "major"}:
        return "error"
    if text in {"medium", "moderate", "normal"}:
        return "warning"
    if text in {"low", "minor", "suggestion", "note"}:
        return "info"
    return "warning"


def _normalize_quality_score(value: object) -> int:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0
    if score <= 0:
        return 0
    if score <= 10:
        return int(round(score))
    if score <= 100:
        return int(round(score / 10.0))
    return 10


def _layout_node_from_section(sec: PosterSection, reading_order: int) -> LayoutNode:
    return LayoutNode(
        node_id=sec.section_id,
        node_type="text" if sec.type != "title" else "title",
        title=sec.title,
        content_md=sec.content_md,
        child_ids=[],
        figure_ids=[],
        reading_order=reading_order,
        space_ratio=min(1.0, max(0.08, (len(sec.content_md or "") / 1400.0) + 0.1)),
        section_col_span=sec.col_span,
        section_row_span=sec.row_span,
        figure_width_ratio=0.9,
        constraints=LayoutConstraints(
            min_ratio=0.08,
            max_ratio=0.95,
            priority=2 if sec.type in {"main_method", "experiments"} else 1,
        ),
        notes=sec.type,
    )


def _layout_nodes_from_paper_section(sec, reading_order: int = 0) -> list[LayoutNode]:
    nodes = [LayoutNode(
        node_id=sec.section_id,
        node_type="title" if getattr(sec, "level", 1) == 1 else "text",
        title=getattr(sec, "title", ""),
        content_md=(getattr(sec, "text", "") or getattr(sec, "raw_latex", "") or ""),
        figure_ids=[getattr(fig, "figure_id", "") for fig in getattr(sec, "figures", []) if getattr(fig, "figure_id", "")],
        child_ids=[getattr(sub, "section_id", "") for sub in getattr(sec, "subsections", []) if getattr(sub, "section_id", "")],
        reading_order=reading_order,
        space_ratio=min(1.0, max(0.08, (len((getattr(sec, "text", "") or "")) / 1400.0) + 0.08)),
        section_col_span=1,
        section_row_span=1,
        figure_width_ratio=0.9,
        constraints=LayoutConstraints(min_ratio=0.08, max_ratio=0.95, priority=max(1, 4 - getattr(sec, "level", 1))),
        notes=f"paper-section:{getattr(sec, 'level', 1)}",
    )]
    next_order = reading_order + 1
    for sub in getattr(sec, "subsections", []) or []:
        sub_nodes = _layout_nodes_from_paper_section(sub, next_order)
        nodes.extend(sub_nodes)
        next_order += len(sub_nodes)
    return nodes


def build_layout_tree(doc: PaperDocument, analysis: PaperAnalysis, use_gpt5: bool = True) -> LayoutTree:
    analysis = normalize_analysis_for_poster(analysis.model_copy(deep=True))
    _augment_key_formulas(doc, analysis)

    required_items = [doc.title]
    if analysis.problem_statement:
        required_items.append(analysis.problem_statement)
    required_items.extend(c.text for c in analysis.contributions[:4] if c.text)
    if analysis.method_overview:
        required_items.append(analysis.method_overview)
    if analysis.experiments and analysis.experiments.main_results:
        required_items.append(analysis.experiments.main_results)

    if use_gpt5 and LLMClient.is_configured():
        try:
            client = LLMClient(api_key=getattr(settings, "planner_api_key", None) or None, base_url=getattr(settings, "planner_base_url", None) or None, model=getattr(settings, "planner_model", None) or None)
            prompt = _build_tree_prompt(doc, analysis)
            response = client.chat_json(system=_TREE_SYSTEM_PROMPT, user=prompt)
            return _parse_tree_response(doc, analysis, response)
        except Exception as e:
            logger.warning("Tree planning failed, falling back to deterministic tree: %s", e)

    return _fallback_layout_tree(doc, analysis, required_items)


def _build_tree_prompt(doc: PaperDocument, analysis: PaperAnalysis) -> str:
    parts = [f"Paper: {doc.title}"]
    parts.append(f"Problem: {analysis.problem_statement}")
    parts.append("Contributions:")
    for contrib in analysis.contributions:
        parts.append(f"- {contrib.text}")
    parts.append(f"Method: {analysis.method_overview}")
    if analysis.experiments:
        parts.append(f"Results: {analysis.experiments.main_results}")
        if analysis.experiments.takeaways:
            parts.append("Takeaways:")
            for item in analysis.experiments.takeaways:
                parts.append(f"- {item}")
    parts.append("Figures:")
    for fig in doc.figures[:12]:
        parts.append(f"- [{fig.figure_id}] {fig.caption}")
    return "\n".join(parts)


def _parse_tree_response(doc: PaperDocument, analysis: PaperAnalysis, response: dict[str, Any]) -> LayoutTree:
    nodes = []
    for idx, node_data in enumerate(response.get("nodes", [])):
        if not isinstance(node_data, dict):
            continue
        nodes.append(LayoutNode(
            node_id=str(node_data.get("node_id", f"node-{idx}")),
            node_type=node_data.get("node_type", "text"),
            title=str(node_data.get("title", "")),
            content_md=str(node_data.get("content_md", "")),
            figure_ids=[str(x) for x in node_data.get("figure_ids", []) if x],
            child_ids=[str(x) for x in node_data.get("child_ids", []) if x],
            reading_order=int(node_data.get("reading_order", idx)),
            space_ratio=float(node_data.get("space_ratio", 0.2)),
            constraints=_parse_layout_constraints(node_data.get("constraints", {}), priority=max(1, 4 - idx)),
            notes=str(node_data.get("notes", "")),
        ))
    if not nodes:
        return _fallback_layout_tree(doc, analysis)
    return LayoutTree(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        title=doc.title,
        required_items=[str(x) for x in response.get("required_items", []) if x],
        nodes=nodes,
        root_id=str(response.get("root_id", "root")),
        reading_path=[str(x) for x in response.get("reading_path", []) if x],
        layout_notes=[str(x) for x in response.get("layout_notes", []) if x],
    )


def _parse_layout_constraints(value: object, priority: int = 0) -> LayoutConstraints:
    if isinstance(value, LayoutConstraints):
        return value
    if isinstance(value, dict):
        return LayoutConstraints.model_validate({
            "min_ratio": value.get("min_ratio", 0.08),
            "max_ratio": value.get("max_ratio", 0.95),
            "priority": value.get("priority", priority),
        })
    if isinstance(value, list):
        note_text = "; ".join(str(x) for x in value if str(x).strip())
        return LayoutConstraints(min_ratio=0.08, max_ratio=0.95, priority=priority)
    return LayoutConstraints(min_ratio=0.08, max_ratio=0.95, priority=priority)


def _fallback_layout_tree(doc: PaperDocument, analysis: PaperAnalysis, required_items: Optional[list[str]] = None) -> LayoutTree:
    nodes = []
    order = 0
    for sec in doc.sections:
        paper_nodes = _layout_nodes_from_paper_section(sec, order)
        nodes.extend(paper_nodes)
        order += len(paper_nodes)

    sections = generate_blueprint(doc, analysis).sections
    sections = _drop_top_summary_sections(sections)
    for sec in sections:
        nodes.append(_layout_node_from_section(sec, order))
        order += 1
    return LayoutTree(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        title=doc.title,
        required_items=required_items or [doc.title],
        nodes=nodes,
        root_id="root",
        reading_path=[node.node_id for node in nodes],
        layout_notes=["Fallback deterministic layout tree."],
    )


def layout_tree_to_blueprint(tree: LayoutTree, doc: PaperDocument, analysis: PaperAnalysis) -> PosterBlueprint:
    base = generate_blueprint(doc, analysis)
    node_map = {node.node_id: node for node in tree.nodes}

    sections: list[PosterSection] = []
    for sec in base.sections:
        node = node_map.get(sec.section_id)
        if node:
            sec.content_md = node.content_md or sec.content_md
            sec.title = node.title or sec.title
            sec.row = min(3, max(0, node.reading_order))
            sec.col_span = node.section_col_span if node.section_col_span else (2 if node.space_ratio >= 0.35 else sec.col_span)
            sec.row_span = node.section_row_span if node.section_row_span else sec.row_span
            if sec.type == "title":
                sec.col_span = 3
        sections.append(sec)

    if not sections:
        sections = base.sections

    figure_placements = list(base.figure_placements)
    formula_displays = list(base.formula_displays)

    for node in tree.nodes:
        if node.node_type == "figure":
            for figure_id in node.figure_ids:
                if figure_id not in {fp.figure_id for fp in figure_placements}:
                    figure_placements.append(FigurePlacement(
                        figure_id=figure_id,
                        section_id=node.node_id,
                        width_ratio=max(0.35, min(1.0, node.figure_width_ratio)),
                        caption=node.title,
                    ))
        if node.node_type == "formula" and node.content_md:
            formula_displays.append(FormulaDisplay(
                formula_id=node.node_id,
                section_id=node.node_id,
                latex=node.content_md,
                semantic_desc=node.notes,
            ))

    _tighten_layout(sections, figure_placements)
    return PosterBlueprint(
        paper_id=doc.paper_id,
        poster_title=doc.title,
        authors_str=_format_authors(doc.authors),
        code_url=analysis.code_url,
        width_px=base.width_px,
        height_px=base.height_px,
        width_mm=base.width_mm,
        height_mm=base.height_mm,
        sections=sections,
        figure_placements=figure_placements[:4],
        formula_displays=formula_displays[:5],
        color_scheme=base.color_scheme or _default_colors(),
    )


def _apply_comment_feedback(tree: LayoutTree, blueprint: PosterBlueprint, review: PosterReview) -> tuple[LayoutTree, PosterBlueprint]:
    node_map = {node.node_id: node for node in tree.nodes}
    section_map = {sec.section_id: sec for sec in blueprint.sections}
    for item in review.issues:
        target = (item.target or "").strip()
        if target and target in node_map:
            node = node_map[target]
            if item.action in {"resize", "reflow"}:
                node.space_ratio = min(1.0, node.space_ratio + 0.12)
                node.section_col_span = min(3, max(node.section_col_span, 2))
                node.constraints.min_ratio = min(node.constraints.min_ratio + 0.04, 0.9)
            if item.action == "rewrite":
                if node.content_md:
                    node.content_md = _trim_dense_text(node.content_md)
                else:
                    node.content_md = item.suggestion or node.content_md
            if item.action == "replace_figure" and node.figure_ids:
                node.space_ratio = min(1.0, node.space_ratio + 0.1)
                node.figure_width_ratio = min(1.0, max(node.figure_width_ratio, 0.92))
        if target and target in section_map:
            sec = section_map[target]
            if item.action in {"resize", "reflow"}:
                sec.col_span = min(3, max(sec.col_span, 2))
                sec.row_span = min(3, max(sec.row_span, 1))
            if item.action == "rewrite" and item.suggestion:
                sec.content_md = item.suggestion
            if item.action == "replace_figure" and sec.type == "experiments":
                sec.col_span = min(3, sec.col_span + 1)
    if any("text" in (item.issue or "").lower() or "density" in (item.issue or "").lower() for item in review.issues):
        for sec in blueprint.sections:
            if sec.type in {"main_method", "experiments", "contributions", "highlights"}:
                sec.col_span = min(3, max(sec.col_span, 2))
    if any("figure" in (item.issue or "").lower() for item in review.issues):
        for fp in blueprint.figure_placements:
            fp.width_ratio = min(1.0, max(fp.width_ratio, 0.88))
    return tree, blueprint


def _trim_dense_text(text: str, max_words: int = 90) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip() + ""


def render_layout_tree(doc: PaperDocument, analysis: PaperAnalysis, tree: LayoutTree, output_dir: Path) -> tuple[PosterBlueprint, Path]:
    blueprint = layout_tree_to_blueprint(tree, doc, analysis)
    renderer = HtmlPosterRenderer()
    html_path = output_dir / "poster.html"
    renderer.render_to_file(blueprint, doc, html_path)
    return blueprint, html_path


def review_rendered_poster(html_path: Path, output_dir: Path, provider: str | None = None) -> PosterReview:
    png_path = output_dir / "poster.png"
    capture_poster(html_path, png_path)
    provider = provider or _poster_vision_provider()
    review = multimodal_analyze(
        system_prompt=_COMMENT_SYSTEM_PROMPT,
        image_paths=[str(png_path)] if png_path.exists() else [],
        user_text="Review the rendered poster for text overload, tiny figures, density imbalance, and reading order.",
        provider=provider,
    )
    if not review:
        return PosterReview(summary="No vision review available.")
    issues = []
    for item in review.get("issues", []):
        if isinstance(item, dict):
            issues.append(PosterComment(
                issue=str(item.get("description", "")),
                severity=_normalize_severity(item.get("severity", "warning")),
                target=str(item.get("target", "")),
                suggestion=str(item.get("suggestion", "")),
                action=str(item.get("action", "rewrite")) if item.get("action") else "rewrite",
            ))
    return PosterReview(
        quality_score=_normalize_quality_score(review.get("quality_score", 0)),
        needs_improvement=bool(review.get("needs_improvement", True)),
        issues=issues,
        summary=str(review.get("summary", "")),
        layout_feedback=[str(x) for x in review.get("layout_feedback", []) if x],
    )


def generate_paperquiz_questions(doc: PaperDocument, analysis: PaperAnalysis, count: int = 6) -> list[EvaluationQuestion]:
    questions: list[EvaluationQuestion] = []
    if analysis.problem_statement:
        questions.append(EvaluationQuestion(question_id="q-problem", question="What problem does this method solve?", answer=analysis.problem_statement, evidence=[analysis.problem_statement], category="problem"))
    if analysis.method_overview:
        questions.append(EvaluationQuestion(question_id="q-method", question="How does the method work at a high level?", answer=analysis.method_overview, evidence=[analysis.method_overview], category="method"))
    for idx, contrib in enumerate(analysis.contributions[:2], start=1):
        questions.append(EvaluationQuestion(question_id=f"q-contrib-{idx}", question=f"What is contribution {idx}?", answer=contrib.text, evidence=[contrib.text], category="contribution"))
    if analysis.experiments and analysis.experiments.main_results:
        questions.append(EvaluationQuestion(question_id="q-result", question="What are the main results?", answer=analysis.experiments.main_results, evidence=[analysis.experiments.main_results], category="results"))
    if analysis.experiments and analysis.experiments.takeaways:
        questions.append(EvaluationQuestion(question_id="q-takeaway", question="What is an important takeaway from the experiments?", answer=analysis.experiments.takeaways[0], evidence=[analysis.experiments.takeaways[0]], category="results"))
    return questions[:count]


def evaluate_poster_qa(doc: PaperDocument, analysis: PaperAnalysis, poster_text: str, visual_score: int = 0) -> PosterQAEval:
    questions = generate_paperquiz_questions(doc, analysis)
    client = LLMClient(api_key=settings.openai_api_key or settings.planner_api_key, base_url=settings.planner_base_url or settings.llm_base_url, model=settings.qa_model)
    poster_answers: list[str] = []
    correct_count = 0
    for q in questions:
        prompt = (
            f"Poster content summary:\n{_summarize_poster_text(poster_text, max_chars=9000)}"
            f"\n\nQuestion: {q.question}\n"
            "Return json with answer, short_reason, and confidence."
        )
        try:
            resp = client.chat_json(system=_QA_SYSTEM_PROMPT, user=prompt)
            answer = str(resp.get("answer", "")).strip() or str(resp.get("poster_answer", "")).strip()
        except Exception as e:
            logger.warning("QA eval failed for %s: %s", q.question_id, e)
            answer = ""
        poster_answers.append(answer)
        if q.answer and answer and _answer_overlap(q.answer, answer):
            correct_count += 1
    total = len(questions)
    accuracy = (correct_count / total) if total else 0.0
    return PosterQAEval(
        paper_id=doc.paper_id,
        arxiv_id=doc.arxiv_id,
        questions=questions,
        poster_answers=poster_answers,
        correct_count=correct_count,
        total_count=total,
        accuracy=accuracy,
        coverage=accuracy,
        recall=accuracy,
        visual_score=visual_score,
        qa_score=int(round(accuracy * 10)),
        summary=f"Answered {correct_count}/{total} questions from poster content.",
    )


def _answer_overlap(reference: str, candidate: str) -> bool:
    ref = _normalize_text_for_eval(reference)
    cand = _normalize_text_for_eval(candidate)
    if not ref or not cand:
        return False
    if ref in cand or cand in ref:
        return True
    ref_tokens = {tok for tok in ref.split() if len(tok) > 2}
    cand_tokens = {tok for tok in cand.split() if len(tok) > 2}
    return len(ref_tokens & cand_tokens) >= max(1, min(3, len(ref_tokens) // 3))


def _normalize_text_for_eval(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s.%+-]", " ", (text or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _summarize_poster_text(poster_text: str, max_chars: int = 9000) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", poster_text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def run_poster_v2(arxiv_id: str, output_dir: Path | str = Path("output"), use_gpt5: bool = True) -> dict[str, Any]:
    out = resolve_paper_output_dir(output_dir, arxiv_id)
    db = PaperDatabase()
    doc = db.get_paper_by_arxiv(arxiv_id)
    analysis = db.get_analysis_by_arxiv(arxiv_id)
    db.close()
    if not doc or not analysis:
        raise RuntimeError(f"Paper or analysis not found for {arxiv_id}")

    tree = build_layout_tree(doc, analysis, use_gpt5=use_gpt5)
    blueprint, html_path = render_layout_tree(doc, analysis, tree, out)
    review = review_rendered_poster(html_path, out, provider=_poster_vision_provider())

    if review.needs_improvement and review.issues:
        tree, blueprint = _apply_comment_feedback(tree, blueprint, review)
        (out / "layout_tree.json").write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        (out / "blueprint_v2.json").write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")
        blueprint, html_path = render_layout_tree(doc, analysis, tree, out)
        review = review_rendered_poster(html_path, out, provider=_poster_vision_provider())

    poster_text = html_path.read_text(encoding="utf-8")
    qa_eval = evaluate_poster_qa(doc, analysis, poster_text, visual_score=review.quality_score)

    (out / "layout_tree.json").write_text(tree.model_dump_json(indent=2), encoding="utf-8")
    (out / "blueprint_v2.json").write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")
    (out / "poster_review.json").write_text(review.model_dump_json(indent=2), encoding="utf-8")
    (out / "poster_qa_eval.json").write_text(qa_eval.model_dump_json(indent=2), encoding="utf-8")

    return {
        "layout_tree": tree,
        "blueprint": blueprint,
        "html_path": html_path,
        "review": review,
        "qa_eval": qa_eval,
    }
