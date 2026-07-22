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
        figures=[
            Figure(figure_id='fig-101', caption='Framework overview', section_id='s1', local_path='framework.pdf'),
            Figure(figure_id='fig-102', caption='Results on benchmarks', section_id='s1', local_path='results.pdf'),
        ],
    )

class TestPosterBlueprint:
    def test_generate_returns_blueprint(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        assert isinstance(bp, PosterBlueprint)
        assert bp.width_px == 1200 and bp.height_px == 1697
        assert len(bp.sections) >= 6

    def test_title_section(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        titles = [s.section_id for s in bp.sections]
        assert 'sec-title' in titles
        assert 'sec-main-method' in titles
        assert 'sec-experiments' in titles
        assert 'sec-contributions' in titles
        assert 'sec-highlights' in titles

    def test_top_summary_sections_are_removed(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        titles = {s.section_id for s in bp.sections}
        assert 'sec-motivation' not in titles
        assert 'sec-method-overview' not in titles
        assert 'sec-key-idea' not in titles

    def test_figure_placement(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        placements = {p.figure_id: p.section_id for p in bp.figure_placements}
        assert placements['fig-001'] == 'sec-experiments'
        assert placements['fig-002'] == 'sec-main-method'
        assert placements['fig-101'] == 'sec-main-method'
        assert placements['fig-102'] == 'sec-experiments'

    def test_doc_figures_can_feed_placements(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        placements = {p.figure_id: p.section_id for p in bp.figure_placements}
        assert placements['fig-101'] == 'sec-main-method'
        assert placements['fig-102'] == 'sec-experiments'
        assert len(bp.figure_placements) == 4

    def test_formula_display(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        assert len(bp.formula_displays) == 1
        assert bp.formula_displays[0].section_id == 'sec-main-method'

    def test_formula_display_drops_broken_latex(self):
        analysis = _make_analysis()
        analysis.key_formulas.append(
            KeyFormula(
                formula_id='f-bad',
                latex=r'\\mathbf{d}_0=\\mathcal{D}_0(\\mathbf{z}^\\mathbf{y}_0),\\\\ \\Tilde{\\mathbf{d}}_i=\\mathbf{d}_i+\\mathrm{Conv}(\\mathrm{RRDB}(',
                semantic_desc='Broken formula fragment',
            )
        )
        bp = generate_blueprint(_make_doc(), analysis)
        latex_values = [f.latex for f in bp.formula_displays]
        assert all('\\Tilde' not in latex for latex in latex_values)
        assert all('RRDB(' not in latex for latex in latex_values)
        assert all('\\cite' not in latex and '\\ref' not in latex for latex in latex_values)

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

    def test_highlights_use_contributions_when_no_takeaways(self):
        analysis = _make_analysis()
        analysis.experiments.takeaways = []
        bp = generate_blueprint(_make_doc(), analysis)
        highlights = next(s for s in bp.sections if s.section_id == 'sec-highlights')
        assert 'First contribution' in highlights.content_md

    def test_author_cleaning(self):
        doc = _make_doc()
        doc.authors[0].name = r'Gongfan Fang\textsuperscript{1}'
        bp = generate_blueprint(doc, _make_analysis())
        assert '\\textsuperscript' not in bp.authors_str
        assert 'Gongfan Fang' in bp.authors_str

    def test_author_cleaning_from_parser_style_input(self):
        doc = _make_doc()
        doc.authors[0].name = r'Gongfan Fang\thanks{Equal contribution}'
        bp = generate_blueprint(doc, _make_analysis())
        assert 'thanks' not in bp.authors_str.lower()

    def test_figure_limit_is_bounded(self):
        analysis = _make_analysis()
        analysis.key_figures.extend([
            KeyFigure(figure_id='fig-003', caption='Intro figure', role='overview'),
            KeyFigure(figure_id='fig-004', caption='Another result', role='result'),
            KeyFigure(figure_id='fig-005', caption='Yet another result', role='comparison'),
        ])
        bp = generate_blueprint(_make_doc(), analysis)
        assert len(bp.figure_placements) <= 4

    def test_dense_layout_tightens_method_section(self):
        analysis = _make_analysis()
        analysis.method_overview = ' '.join(['Dense method text'] * 120)
        bp = generate_blueprint(_make_doc(), analysis)
        method = next(s for s in bp.sections if s.section_id == 'sec-main-method')
        assert method.col_span >= 2

    def test_method_hero_gets_largest_width(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        method_figs = [p for p in bp.figure_placements if p.section_id == 'sec-main-method']
        assert method_figs
        assert max(p.width_ratio for p in method_figs) >= 0.9
