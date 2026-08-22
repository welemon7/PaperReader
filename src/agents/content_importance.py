from __future__ import annotations

import re

from src.schemas.analysis import (
    ContentImportance,
    ImportanceItem,
    MethodComponent,
    PaperAnalysis,
)
from src.schemas.paper import PaperDocument


def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ""
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]


class ContentImportanceAnalyzer:
    """Rank claims before layout so every poster section is not equally loud.

    This first version is deterministic: it uses already validated analysis and
    source-object presence, so a second network call cannot make the pipeline
    fail. The schema leaves room for an LLM ranker later without changing the
    planner contract.
    """

    def analyze(self, analysis: PaperAnalysis, doc: PaperDocument | None = None) -> ContentImportance:
        exp = analysis.experiments
        result_text = _sentence(exp.main_results if exp else "")
        if not result_text:
            result_text = _sentence(analysis.conclusion)
        innovation_text = _sentence(analysis.method_overview)
        if analysis.contributions:
            innovation_text = _sentence(analysis.contributions[0].text) or innovation_text

        components = [
            MethodComponent(
                name=f"Contribution {index + 1}",
                text=_sentence(item.text),
                importance=max(0.55, 0.9 - index * 0.1),
                role="method_component",
            )
            for index, item in enumerate(analysis.contributions[:3])
            if _sentence(item.text)
        ]

        supporting: list[ImportanceItem] = []
        if analysis.problem_statement:
            supporting.append(ImportanceItem(
                text=_sentence(analysis.problem_statement), importance=0.45, role="motivation"
            ))
        if analysis.code_url:
            supporting.append(ImportanceItem(
                text=analysis.code_url, importance=0.25, role="project"
            ))
        if doc and doc.tables:
            supporting.append(ImportanceItem(
                text=f"{len(doc.tables)} source table(s) available for evidence",
                importance=0.70,
                role="benchmark_evidence",
            ))

        return ContentImportance(
            main_message=ImportanceItem(
                text=result_text, importance=1.0, role="hero_result"
            ),
            core_innovation=ImportanceItem(
                text=innovation_text, importance=0.95, role="hero_method"
            ),
            method_components=components,
            supporting_information=supporting,
        )


def analyze_content_importance(analysis: PaperAnalysis, doc: PaperDocument | None = None) -> ContentImportance:
    return ContentImportanceAnalyzer().analyze(analysis, doc)
