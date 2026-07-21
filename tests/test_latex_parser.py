from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
   sys.path.insert(0, str(_root))

from src.parsers.extractor import ComponentExtractor
from src.parsers.latex_parser import LatexParser, ParseResult, SectionBody
from src.parsers.markdown import MarkdownConverter


def _make_minimal_latex(with_refs: bool = False) -> str:
   parts = [
       r'\documentclass{article}',
       r'\title{A Minimal Test Paper}',
       r'\author{Alice Chen\and Bob Wang}',
       r'\begin{document}',
       r'\maketitle',
       r'\begin{abstract}',
       r'This is the abstract. It contains an inline formula \( E = mc^2 \).',
       r'\end{abstract}',
       r'\section{Introduction}',
       r'This is the introduction section. It has a display equation:',
       r'\[',
       r'  \nabla \cdot \mathbf{D} = \rho',
       r'\]',
       r'and an equation environment:',
       r'\begin{equation}',
       r'  \oint \mathbf{E} \cdot d\mathbf{l} = -\frac{d\Phi_B}{dt}',
       r'  \label{eq:faraday}',
       r'\end{equation}',
       r'\subsection{Background}',
       r'Some background text with a figure:',
       r'\begin{figure}',
       r'  \centering',
       r'  \includegraphics[width=0.5\textwidth]{example.png}',
       r'  \caption{An example figure.}',
       r'  \label{fig:example}',
       r'\end{figure}',
       r'\subsection{Motivation}',
       r'More text.',
       r'\section{Method}',
       r'Our method uses:',
       r'\begin{align}',
       r'  f(x) &= x^2 \\',
       r'  g(x) &= \sin(x)',
       r'\end{align}',
       r'\section{Conclusion}',
       r'We conclude.',
   ]
   if with_refs:
       parts.extend([
           r'\begin{thebibliography}{99}',
           r'\bibitem{ref1} J. Smith, A Great Paper, 2020.',
           r'\bibitem{ref2} K. Lee et al., Another Paper, NeurIPS 2023.',
           r'\end{thebibliography}',
       ])
   parts.append(r'\end{document}')
   return '\n'.join(parts)


class TestLatexParser:
   def test_extract_title(self):
       latex = _make_minimal_latex()
       assert LatexParser._extract_title(latex) == 'A Minimal Test Paper'

   def test_extract_authors(self):
       latex = _make_minimal_latex()
       authors = LatexParser._extract_authors(latex)
       assert len(authors) == 2
       assert authors[0]['name'] == 'Alice Chen'
       assert authors[1]['name'] == 'Bob Wang'

   def test_extract_abstract(self):
       latex = _make_minimal_latex()
       abstract = LatexParser._extract_abstract(latex)
       assert 'This is the abstract' in abstract

   def test_extract_sections(self):
       latex = _make_minimal_latex()
       sections = LatexParser._extract_sections(latex)
       titles = [s.title for s in sections if s.level == 1]
       assert 'Introduction' in titles
       assert 'Method' in titles
       assert 'Conclusion' in titles

   def test_read_braces(self):
       text = 'prefix{hello world}suffix'
       content, end = LatexParser._read_braces(text, 6)
       assert content == 'hello world'

   def test_nested_braces(self):
       text = '{outer {inner} end}'
       content, end = LatexParser._read_braces(text, 0)
       assert content == 'outer {inner} end'


class TestComponentExtractor:
   def test_extract_formulas(self):
       latex = _make_minimal_latex()
       result = ParseResult()
       result.merged_latex = latex
       extractor = ComponentExtractor()
       comps = extractor.extract_all(result)
       formulas = comps['formulas']
       assert len(formulas) >= 3
       assert any('nabla' in f['latex'] for f in formulas)

   def test_extract_figures(self):
       latex = _make_minimal_latex()
       result = ParseResult()
       result.merged_latex = latex
       extractor = ComponentExtractor()
       comps = extractor.extract_all(result)
       figures = comps['figures']
       assert len(figures) >= 1
       assert figures[0]['caption'] == 'An example figure.'

   def test_extract_references(self):
       latex = _make_minimal_latex(with_refs=True)
       result = ParseResult()
       result.merged_latex = latex
       extractor = ComponentExtractor()
       comps = extractor.extract_all(result)
       refs = comps['references']
       assert len(refs) >= 2


class TestMarkdownConverter:
   def test_convert(self):
       latex = _make_minimal_latex(with_refs=True)
       result = ParseResult()
       result.merged_latex = latex
       result.title = 'A Minimal Test Paper'
       result.authors = [{'name': 'Alice Chen', 'affiliation': None}, {'name': 'Bob Wang', 'affiliation': None}]
       result.abstract = LatexParser._extract_abstract(latex)
       result.section_bodies = LatexParser._extract_sections(latex)
       result.all_section_bodies = []
       for sb in result.section_bodies:
           result.all_section_bodies.append(sb)
       converter = MarkdownConverter()
       md = converter.convert(result)
       assert '# A Minimal Test Paper' in md
       assert '## Abstract' in md
       assert '## Introduction' in md


if __name__ == '__main__':
   pytest.main([__file__, '-v'])
