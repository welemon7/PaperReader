from __future__ import annotations
import sys, pytest
from src.schemas.analysis import PaperAnalysis, Contribution, ExperimentSummary, KeyFormula, KeyFigure
from src.schemas.paper import PaperDocument, Author, Section, Formula, Figure, Reference
from src.schemas.poster import PosterBlueprint
from src.agents.poster_planner import generate_blueprint, _build_row1, _build_row2, _build_row3, _place_figures, _place_formulas

def _make_analysis():
    return PaperAnalysis(
        paper_id='test-999', arxiv_id='9999.99999',
        title_zh='\u6d4b\u8bd5\u8bba\u6587',
        problem_statement='A challenging problem.',
        contributions=[
            Contribution(text='First contribution', category='method'),
            Contribution(text='Second contribution', category='system'),
        ],
        method_overview='We propose a novel approach.',
        key_formulas=[
            KeyFormula(formula_id='f-001', latex='E=mc^2', semantic_desc='Energy'),
        ],
        key_figures=[
            KeyFigure(figure_id='fig-001', caption='Results', role='result'),
            KeyFigure(figure_id='fig-002', caption='Architecture', role='architecture'),
        ],
        experiments=ExperimentSummary(
            datasets=['Dataset-A'], metrics=['Accuracy'],
            main_results='State-of-the-art', takeaways=['Works well'],
        ),
        conclusion='We conclude.',
        full_analysis_md='# Test',
    )

def _make_doc():
    return PaperDocument(
        paper_id='test-999', arxiv_id='9999.99999', title='Test Paper',
        authors=[Author(name='Alice', affiliation='MIT')], abstract='.',
        sections=[Section(section_id='s1', title='Intro', level=1, text='.', raw_latex='.')],
        raw_markdown='.',
    )

class TestPosterBlueprint:
    def test_generate_returns_blueprint(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        assert isinstance(bp, PosterBlueprint)
        assert bp.width_px == 1200 and bp.height_px == 1697
        assert len(bp.sections) >= 8

    def test_title_section(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        titles = [s.section_id for s in bp.sections]
        assert 'sec-title' in titles
        assert 'sec-motivation' in titles
        assert 'sec-main-method' in titles
        assert 'sec-experiments' in titles
        assert 'sec-contributions' in titles
        assert 'sec-highlights' in titles

    def test_figure_placement(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        assert len(bp.figure_placements) == 2
        placements = {p.figure_id: p.section_id for p in bp.figure_placements}
        assert placements['fig-001'] == 'sec-experiments'
        assert placements['fig-002'] == 'sec-main-method'

    def test_formula_display(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        assert len(bp.formula_displays) == 1
        assert bp.formula_displays[0].section_id == 'sec-main-method'

    def test_row_columns(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        for s in bp.sections:
            assert 1 <= s.column <= 3
            assert 1 <= s.col_span <= 3
            assert 0 <= s.row <= 3

    def test_color_scheme(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        assert 'primary' in bp.color_scheme
        assert 'accent' in bp.color_scheme
