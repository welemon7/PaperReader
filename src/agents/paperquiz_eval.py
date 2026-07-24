from __future__ import annotations

from pathlib import Path

from src.agents.poster_v2 import evaluate_poster_qa, generate_paperquiz_questions
from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster_v2 import PosterQAEval


def run_paperquiz_eval(doc: PaperDocument, analysis: PaperAnalysis, poster_html_path: Path, visual_score: int = 0) -> PosterQAEval:
    poster_text = poster_html_path.read_text(encoding="utf-8")
    return evaluate_poster_qa(doc, analysis, poster_text, visual_score=visual_score)

