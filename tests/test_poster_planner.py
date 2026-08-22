from __future__ import annotations
import sys, pytest
from unittest.mock import patch
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


def _make_analysis_with_doc_formulas():
    analysis = _make_analysis()
    analysis.key_formulas = [
        KeyFormula(formula_id='f-001', latex='E=mc^2', semantic_desc='Energy'),
    ]
    return analysis

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
        assert bp.width_px == 1920 and bp.height_px == 1080
        assert len(bp.sections) >= 5

    def test_title_section(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        titles = [s.section_id for s in bp.sections]
        assert 'sec-title' in titles
        assert 'sec-motivation' in titles
        assert 'sec-method-overview' in titles
        assert 'sec-key-idea' in titles
        assert 'sec-main-method' in titles
        assert 'sec-contributions' in titles
        assert 'sec-highlights' in titles
        assert 'sec-project' in titles

    def test_compact_layout_positions(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        lookup = {s.section_id: s for s in bp.sections}
        assert lookup['sec-motivation'].row == 1 and lookup['sec-motivation'].column == 1
        assert lookup['sec-method-overview'].row == 1 and lookup['sec-method-overview'].column == 2
        assert lookup['sec-key-idea'].row == 1 and lookup['sec-key-idea'].column == 3
        assert lookup['sec-main-method'].row == 2 and lookup['sec-main-method'].column == 1
        assert lookup['sec-main-method'].col_span == 3
        assert lookup['sec-contributions'].row == 3 and lookup['sec-contributions'].column == 1
        assert lookup['sec-highlights'].row == 3 and lookup['sec-highlights'].column == 2
        assert lookup['sec-project'].row == 3 and lookup['sec-project'].column == 3
        assert lookup['sec-motivation'].row_span == 1
        assert lookup['sec-main-method'].row_span == 1

    def test_motivation_is_short(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        motiv = next(s for s in bp.sections if s.section_id == 'sec-motivation')
        assert len(motiv.content_md.split()) <= 80
        assert '……' not in motiv.content_md

    def test_highlights_are_content_based(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        method = next(s for s in bp.sections if s.section_id == 'sec-main-method')
        overview = next(s for s in bp.sections if s.section_id == 'sec-method-overview')
        motiv = next(s for s in bp.sections if s.section_id == 'sec-motivation')
        highlights = next(s for s in bp.sections if s.section_id == 'sec-highlights')
        assert 'Core Results' in method.title
        assert 'State-of-the-art' in method.content_md
        assert 'First contribution' in highlights.content_md
        # 当前架构为静态渲染：核心内容不带 LLM 注入的 span 标记
        assert '<span class="poster-highlight' not in method.content_md
        assert '<span class="poster-highlight' not in overview.content_md
        assert '<span class="poster-highlight' not in motiv.content_md

    def test_figure_placement(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        placements = {p.figure_id: p.section_id for p in bp.figure_placements}
        assert placements['fig-001'] == 'sec-main-method'
        assert placements['fig-002'] == 'sec-method-overview'
        # 图分配策略：hero 1 张（overview）、core 2 张（结果）、示意 1 张（key_idea）
        assert placements['fig-101'] == 'sec-main-method'
        assert placements['fig-102'] == 'sec-key-idea'

    def test_doc_figures_can_feed_placements(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        placements = {p.figure_id: p.section_id for p in bp.figure_placements}
        assert placements['fig-101'] == 'sec-main-method'
        assert placements['fig-102'] == 'sec-key-idea'
        assert len(bp.figure_placements) == 4

    def test_core_section_gets_two_figures_when_available(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        core_figs = [p for p in bp.figure_placements if p.section_id == 'sec-main-method']
        assert len(core_figs) == 2  # core 左右列均有图，避免空白
        overview_figs = [p for p in bp.figure_placements if p.section_id == 'sec-method-overview']
        assert len(overview_figs) == 1  # hero 只放一张

    def test_formula_display(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        assert len(bp.formula_displays) == 1
        assert bp.formula_displays[0].section_id == 'sec-key-idea'

    def test_formula_titles_are_sentence_cased(self):
        analysis = _make_analysis()
        analysis.key_formulas = [
            KeyFormula(
                formula_id='f-001',
                latex='E=mc^2',
                semantic_desc='DECOMPOSES OBSERVED IMAGES INTOREFLECTANCE AND ILLUMINATION',
            ),
        ]
        bp = generate_blueprint(_make_doc(), analysis)
        assert bp.sections[3].content_md.count('formula-box') == 1
        assert '<div class="formula-label">Decomposes Observed Images into Reflectance and Illumination</div>' in bp.sections[3].content_md

    def test_motivation_restores_original_prose_and_key_idea_receives_formulas(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        sections = {section.section_id: section for section in bp.sections}
        assert 'A challenging problem.' in sections['sec-motivation'].content_md
        assert 'The core advantage is that We propose a novel approach.' in sections['sec-motivation'].content_md
        assert 'First contribution' not in sections['sec-motivation'].content_md
        assert 'formula-box' in sections['sec-key-idea'].content_md
        assert 'E=mc^2' in sections['sec-key-idea'].content_md
        assert '<div class="formula-label">Energy</div>' in sections['sec-key-idea'].content_md
        assert 'Formula 1' not in sections['sec-key-idea'].content_md
        assert 'formula-box' not in sections['sec-method-overview'].content_md

    def test_key_idea_limits_formula_boxes_to_four(self):
        analysis = _make_analysis()
        analysis.key_formulas = [
            KeyFormula(formula_id=f'f-{index}', latex=f'x_{index}=y_{index}', semantic_desc=f'Formula meaning {index}')
            for index in range(5)
        ]
        bp = generate_blueprint(_make_doc(), analysis)
        key_idea = next(section for section in bp.sections if section.section_id == 'sec-key-idea')
        assert key_idea.content_md.count('class="formula-box"') == 4
        assert len(bp.formula_displays) == 4
        assert 'Formula meaning 4' not in key_idea.content_md

    def test_formula_display_backfills_from_document_when_underfilled(self):
        doc = _make_doc()
        doc.formulas = [
            Formula(formula_id='f-001', latex='E=mc^2', section_id='s1'),
            Formula(formula_id='f-002', latex='a+b=c', section_id='s1'),
            Formula(formula_id='f-003', latex='x=y', section_id='s1'),
        ]
        analysis = _make_analysis_with_doc_formulas()
        analysis.key_formulas = [KeyFormula(formula_id='f-001', latex='E=mc^2', semantic_desc='Energy')]
        bp = generate_blueprint(doc, analysis)
        assert len(bp.formula_displays) >= 2
        latex_values = [f.latex for f in bp.formula_displays]
        assert 'E=mc^2' in latex_values
        assert any(latex in latex_values for latex in ['a+b=c', 'x=y'])

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

    def test_results_omit_takeaways_from_core_table(self):
        bp = generate_blueprint(_make_doc(), _make_analysis())
        core = next(s for s in bp.sections if s.section_id == 'sec-main-method')
        assert 'State-of-the-art' in core.content_md
        assert 'Takeaways' not in core.content_md
        assert 'Works well' not in core.content_md

    def test_core_results_prefer_final_table_over_source_table(self):
        doc = _make_doc()
        doc.raw_markdown = r'''
        \begin{table}
        \begin{tabular}{lcc}
        Method & PSNR & SSIM \\
        Baseline & 30.1 & 0.90 \\
        \end{tabular}
        \end{table}
        '''
        analysis = _make_analysis()
        analysis.final_tables = [
            {
                'table_id': 'table-001',
                'caption': 'Comparison on benchmarks',
                'datasets': ['Dataset-A'],
                'metrics': ['PSNR', 'SSIM'],
                'row_groups': ['Dataset-A'],
                'headers': ['Dataset', 'Method', 'PSNR', 'SSIM'],
                'rows': [['Dataset-A', 'Ours', '32.4', '0.94']],
                'row_indices': [0],
                'column_indices': [0, 1, 2, 3],
                'column_groups': [[0], [1], [2], [3]],
                'notes': 'paper method only',
            }
        ]
        core = next(section for section in generate_blueprint(doc, analysis).sections if section.section_id == 'sec-main-method')
        assert 'core-metrics-table' in core.content_md
        assert 'Comparison on benchmarks' in core.content_md
        assert 'Dataset-A' in core.content_md
        assert 'Ours' in core.content_md
        assert 'Baseline' not in core.content_md

    def test_core_results_use_real_numeric_tex_table_without_item_details_label(self):
        doc = _make_doc()
        doc.raw_markdown = r'''
        \begin{table}
        \caption{Comparison on the benchmark.}
        \begin{tabular}{lccc}
        \toprule
        Method & Params & FLOPs & Accuracy \\
        Baseline & 1.01B & 0.52T & 73.0\% \\
        \textbf{Ours} & \textbf{0.62B} & \textbf{0.32T} & \textbf{73.1\%} \\
        \bottomrule
        \end{tabular}
        \end{table}
        '''
        core = next(section for section in generate_blueprint(doc, _make_analysis()).sections if section.section_id == 'sec-main-method')
        assert 'core-metrics-table' in core.content_md
        assert 'Method' in core.content_md and '0.62B' in core.content_md and '73.1%' in core.content_md
        assert 'Item Details' not in core.content_md

    def test_core_results_omit_table_when_source_has_no_numeric_tex_table(self):
        doc = _make_doc()
        doc.raw_markdown = r'\begin{table}\begin{tabular}{ll}Name & Type \\ A & B \\ \end{tabular}\end{table}'
        core = next(section for section in generate_blueprint(doc, _make_analysis()).sections if section.section_id == 'sec-main-method')
        assert '[[CORE_TABLE]]' not in core.content_md

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

    def test_planner_respects_word_budget(self):
        from src.agents.content_policy import TOTAL_WORD_BUDGET, count_words, section_budget
        analysis = _make_analysis()
        analysis.problem_statement = ' '.join(['word'] * 300)
        analysis.method_overview = ' '.join(['word'] * 300)
        analysis.contributions = [Contribution(text=' '.join(['word'] * 80), category='method') for _ in range(5)]
        analysis.experiments.main_results = ' '.join(['word'] * 200)
        analysis.experiments.takeaways = [' '.join(['word'] * 60)] * 5
        bp = generate_blueprint(_make_doc(), analysis)
        total = 0
        for sec in bp.sections:
            if sec.type == 'title':
                continue
            words = count_words(sec.content_md)
            total += words
            tolerance = 25 if sec.type == 'main_method' else 10  # 结果表/公式容差
            assert words <= section_budget(sec.type) + tolerance, f"{sec.section_id}: {words} words"
        assert total <= TOTAL_WORD_BUDGET + 40

