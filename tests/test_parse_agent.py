from __future__ import annotations

"""Integration test for the full parse pipeline using mock data."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
   sys.path.insert(0, str(_root))

from src.agents.parse_agent import build_node, download_node
from src.parsers.latex_parser import ParseResult, SectionBody
from src.schemas.paper import PaperDocument


class TestParseAgent:

   def test_build_node_with_synthetic_data(self):
       result = ParseResult()
       result.title = "Test Paper"
       result.authors = [{"name": "Alice", "affiliation": "MIT"}]
       result.abstract = "A test abstract."
       sec1 = SectionBody("sec-001", "Intro", 1, "Some intro text.", None)
       sec2 = SectionBody("sec-002", "Method", 1, "Our method.", None)
       result.section_bodies = [sec1, sec2]
       result.all_section_bodies = [sec1, sec2]
       result.formulas = []
       result.figures = []

       state = {
           "arxiv_id": "9999.99999",
           "source_dir": "/tmp/test",
           "main_tex": "/tmp/test/main.tex",
           "parse_result": result,
           "components": {
               "formulas": [
                   {
                       "formula_id": "f-001", "latex": "E=mc^2",
                       "semantic_desc": "", "section_id": "sec-001", "label": None,
                   }
               ],
               "figures": [],
               "references": [
                   {
                       "ref_id": "ref-001", "bibkey": "test2024",
                       "title": "A Test Reference", "authors": "Alice",
                       "journal": None, "year": 2024, "doi": None,
                   }
               ],
           },
           "paper_document": None,
           "error": None,
       }

       updated = build_node(state)
       doc = updated.get("paper_document")
       assert doc is not None
       assert isinstance(doc, PaperDocument)
       assert doc.arxiv_id == "9999.99999"
       assert doc.title == "Test Paper"
       assert len(doc.sections) == 2
       assert len(doc.formulas) == 1
       assert len(doc.references) == 1

   @patch("src.agents.parse_agent.ArxivDownloader")
   def test_download_node_no_id(self, mock_downloader):
       state = {"arxiv_id": "", "error": None}
       result = download_node(state)
       assert "error" in result


if __name__ == "__main__":
   pytest.main([__file__, "-v"])
