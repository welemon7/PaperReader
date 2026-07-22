from __future__ import annotations
import sys, json, pytest
from unittest.mock import MagicMock, patch
from src.agents.understand_agent import (
    load_paper_node, build_prompt_node, validate_node,
    store_analysis_node, call_llm_node,
    _build_analysis_prompt, _parse_analysis,
)
from src.schemas.paper import PaperDocument, Author, Section, Formula, Figure, Reference
from src.schemas.analysis import PaperAnalysis

def _make_paper():
    return PaperDocument(
        paper_id='test-999', arxiv_id='9999.99999', title='Test Paper',
        authors=[Author(name='Alice')],
        abstract='Test abstract.',
        sections=[Section(section_id='sec-001', title='Intro', level=1, text='Some text.', raw_latex='Some text.')],
        formulas=[Formula(formula_id='f-001', latex='E=mc^2', section_id='sec-001')],
        figures=[Figure(figure_id='fig-001', caption='Example.', section_id='sec-001')],
        references=[Reference(ref_id='ref-001', bibkey='test2024', title='Ref', authors='A', year=2024)],
        raw_markdown='# Test\n',
    )

class TestBuildPrompt:
    def test_contains_title(self):
        p = _build_analysis_prompt(_make_paper())
        assert 'Test Paper' in p and 'Alice' in p and 'E=mc^2' in p

class TestParseAnalysis:
    def test_minimal(self):
        d = _make_paper()
        r = {
            'title_zh': '\u6d4b\u8bd5\u8bba\u6587',
            'problem_statement': 'A problem.',
            'contributions': [{'text': 'C1', 'category': 'method'}],
            'method_overview': 'Method.',
            'key_formulas': [{'formula_id': 'f-001', 'latex': 'E=mc^2', 'semantic_desc': 'Energy'}],
            'key_figures': [{'figure_id': 'fig-001', 'caption': 'Ex.', 'role': 'result'}],
            'experiments': {'datasets': ['d1'], 'metrics': ['acc'], 'main_results': '99%', 'takeaways': ['ok']},
            'conclusion': 'Done.', 'full_analysis_md': '# A\n',
        }
        a = _parse_analysis(d, r)
        assert isinstance(a, PaperAnalysis)
        assert a.title_zh == '\u6d4b\u8bd5\u8bba\u6587'
        assert len(a.contributions) == 1
        assert a.contributions[0].category == 'method'
        assert a.experiments and len(a.experiments.datasets) == 1

class TestAgentNodes:
    def test_load_empty(self):
        assert 'error' in load_paper_node({'arxiv_id': ''})
    def test_build_no_doc(self):
        assert 'error' in build_prompt_node({'paper_document': None})
    def test_call_no_key(self):
        with patch('src.agents.understand_agent.LLMClient.is_configured', return_value=False):
            r = call_llm_node({'analysis_prompt': 'test'})
            assert 'error' in r and 'API' in r['error']
    def test_validate_no_doc(self):
        assert 'error' in validate_node({'paper_document': None, 'llm_response': None})

    def test_store_preserves_upstream_error(self):
        result = store_analysis_node({'paper_analysis': None, 'error': 'Missing paper document or LLM response'})
        assert result['error'] == 'Missing paper document or LLM response'
