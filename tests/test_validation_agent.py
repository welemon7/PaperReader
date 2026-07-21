from __future__ import annotations
import sys, json, pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.schemas.validation import PosterValidation, ValidationIssue
from src.agents.validation_agent import _build_validation_prompt
from src.schemas.poster import PosterBlueprint, PosterSection
from src.agents.optimizer_agent import _apply_structured_quality_gate
from src.schemas.analysis import PaperAnalysis, Contribution
from src.schemas.paper import PaperDocument

def _make_bp():
    return PosterBlueprint(
        paper_id="test-999", poster_title="Test",
        sections=[
            PosterSection(section_id="s1", type="motivation", title="Motivation",
                        content_md="We solve problem X.", column=1, col_span=1, row=1),
            PosterSection(section_id="s2", type="experiments", title="Experiments",
                        content_md="Our method gets 99%.", column=1, col_span=1, row=2),
        ],
        color_scheme={"primary": "#1a5276"},
    )

class TestValidation:
    def test_build_prompt_contains_title(self):
        bp = _make_bp()
        prompt = _build_validation_prompt("# Paper Title\n\nAbstract.", bp)
        assert "Paper Title" in prompt
        assert "Motivation" in prompt
        assert "We solve problem X." in prompt
        assert "Experiments" in prompt

    def test_build_prompt_truncation(self):
        bp = _make_bp()
        long_md = "# Long\n" + ("x" * 10000)
        prompt = _build_validation_prompt(long_md, bp)
        assert len(prompt) > 1000
        assert "Poster Blueprint" in prompt

    def test_parse_llm_response(self):
        llm_resp = {
            "scores": {"coverage": 8, "accuracy": 9, "clarity": 7, "completeness": 8},
            "issues": [
                {"severity": "warning", "category": "missing_experiment",
                 "description": "Missing dataset details", "location": "Experiments",
                 "suggestion": "Add dataset names"}
            ],
            "summary": "Good poster, minor issues.",
        }
        issues = [ValidationIssue(**i) for i in llm_resp["issues"]]
        scores = llm_resp["scores"]
        v = PosterValidation(paper_id="test-999", arxiv_id="9999.99999", scores=scores, issues=issues, summary=llm_resp["summary"])
        assert v.scores["coverage"] == 8
        assert len(v.issues) == 1
        assert v.issues[0].severity == "warning"
        assert v.issues[0].category == "missing_experiment"

    @patch("src.agents.validation_agent.settings.gemini_api_key", "")
    @patch("src.agents.validation_agent.LLMClient.is_configured", return_value=False)
    def test_llm_not_configured(self, mock_conf):
        from src.agents.validation_agent import call_llm_node
        result = call_llm_node({"validation_prompt": "test"})
        assert "error" in result
        assert "API key" in result["error"]

    def test_validation_from_scratch(self):
        from src.schemas.validation import PosterValidation
        v = PosterValidation(paper_id="t1", arxiv_id="9999.99999")
        assert v.scores["coverage"] == 0
        assert v.issues == []

    def test_quality_gate_drops_unmatched_numbers_and_formulas(self):
        bp = _make_bp()
        doc = PaperDocument(
            paper_id="test-999",
            arxiv_id="9999.99999",
            title="Test",
            raw_markdown="We reach 91%.",
        )
        analysis = PaperAnalysis(
            paper_id="test-999",
            arxiv_id="9999.99999",
            title_zh="",
            problem_statement="",
            contributions=[Contribution(text="Improve accuracy to 91%", category="method")],
            method_overview="We optimize E=mc^2.",
            key_formulas=[],
            key_figures=[],
            experiments=None,
            conclusion="",
            full_analysis_md="",
        )
        review = {
            "suggestions": {
                "s1": {"content_md": "We reach 92% and use $a=b$"},
                "s2": {"content_md": "Keep this exact 91%"},
            }
        }

        gated = _apply_structured_quality_gate(bp, doc, analysis, review)
        assert "s1" not in gated["suggestions"]
        assert gated["suggestions"]["s2"]["content_md"] == "Keep this exact 91%"
