from __future__ import annotations

import re

from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import PosterStoryBeat, PosterStoryPlan


def _short(text: str, words: int = 28) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ''
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    tokens = sentence.split()
    return ' '.join(tokens[:words]).rstrip(' ,;:') + ('.' if len(tokens) > 0 and not sentence.endswith(('.', '!', '?')) else '')


def _result_text(analysis: PaperAnalysis) -> str:
    experiments = analysis.experiments
    if experiments and experiments.main_results:
        return _short(experiments.main_results, 24)
    if experiments and experiments.takeaways:
        return _short(experiments.takeaways[0], 24)
    return _short(analysis.conclusion, 24)


class PosterStoryPlanner:
    """Turn paper analysis into a five-beat poster narrative.

    The planner deliberately asks a sequence of reader questions instead of
    mirroring paper section headings. It is deterministic for now, which keeps
    story planning reproducible and makes it safe to run before rendering.
    """

    def plan(self, doc: PaperDocument, analysis: PaperAnalysis) -> PosterStoryPlan:
        result = _result_text(analysis)
        idea = ''
        importance = getattr(analysis, 'content_importance', None)
        if importance and importance.core_innovation.text:
            idea = _short(importance.core_innovation.text, 24)
        idea = idea or _short(analysis.method_overview, 24)
        contribution_text = _short(analysis.contributions[0].text, 22) if analysis.contributions else idea
        why = _short(analysis.problem_statement, 28)
        how = contribution_text or _short(analysis.method_overview, 28)
        if not how:
            how = 'A compact pipeline connects the proposed idea to the target task.'
        if not result:
            result = 'The proposed method preserves performance on the reported benchmarks.'
        hook = result or _short(analysis.conclusion, 24)

        beats = [
            PosterStoryBeat(
                beat_id='story-hook', type='hook',
                question='What should the viewer remember first?', text=hook,
                importance=1.0, source_ids=['experiments.main_results'], target_section_id='sec-main-method',
            ),
            PosterStoryBeat(
                beat_id='story-why', type='why',
                question='Why is this problem important?', text=why,
                importance=0.72, source_ids=['problem_statement'], target_section_id='sec-motivation',
            ),
            PosterStoryBeat(
                beat_id='story-idea', type='idea',
                question='What is the key idea?', text=idea,
                importance=0.95, source_ids=['content_importance.core_innovation', 'method_overview'], target_section_id='sec-key-idea',
            ),
            PosterStoryBeat(
                beat_id='story-how', type='how',
                question='How does the method work?', text=how,
                importance=0.88, source_ids=['contributions', 'method_overview'], target_section_id='sec-method-overview',
            ),
            PosterStoryBeat(
                beat_id='story-evidence', type='evidence',
                question='What evidence proves it?', text=result,
                importance=1.0, source_ids=['experiments', 'figures', 'tables'], target_section_id='sec-main-method',
            ),
        ]
        thesis = hook
        return PosterStoryPlan(
            thesis=thesis,
            beats=beats,
            reading_path=[beat.beat_id for beat in beats],
        )


def plan_poster_story(doc: PaperDocument, analysis: PaperAnalysis) -> PosterStoryPlan:
    return PosterStoryPlanner().plan(doc, analysis)
