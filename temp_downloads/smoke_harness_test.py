"""End-to-end smoke test: real rendering + mocked capture + mocked VLM (sandbox-safe).

Playwright browser launch is blocked inside the DSH sandbox (named-pipe EPERM), so this
script fakes the capture step with a valid 1x1 PNG while exercising the real renderer,
the real feedback-application logic and the gated loop on real paper data.

Run with the project venv:  .venv\\Scripts\\python.exe temp_downloads/smoke_harness_test.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.schemas.analysis import PaperAnalysis
from src.schemas.paper import PaperDocument
from src.schemas.poster import PosterBlueprint
from src.schemas.poster_harness import HarnessConfig
from src.renderers.html_renderer import HtmlPosterRenderer
from src.agents.poster_harness import run_poster_harness, review_rendered_poster

OUT = _ROOT / "temp_downloads" / "smoke_harness"
OUT.mkdir(parents=True, exist_ok=True)
# 清理上次运行残留，保证轮次计数正确
import shutil
if (OUT / "harness").exists():
    shutil.rmtree(OUT / "harness")
for stale in OUT.glob("harness_report.json"):
    stale.unlink()

# valid 1x1 transparent PNG
_PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

doc = PaperDocument.model_validate(json.loads((_ROOT / "example/softmask/paper.json").read_text(encoding="utf-8")))
blueprint = PosterBlueprint.model_validate(json.loads((_ROOT / "example/softmask/blueprint.json").read_text(encoding="utf-8")))

analysis = PaperAnalysis(
    paper_id=doc.paper_id,
    arxiv_id=doc.arxiv_id,
    problem_statement=doc.abstract or "Soft mask problem",
    contributions=[],
    method_overview="Soft mask generation with detail injection.",
    experiments=None,
)

draft = OUT / "poster_draft.html"
HtmlPosterRenderer().render_to_file(blueprint, doc, draft)
print(f"[smoke] draft rendered: {draft} ({draft.stat().st_size} bytes)")


def _fake_capture(html_path, png_path, section_selectors, width=1200, height=1697):
    png_path.write_bytes(base64.b64decode(_PNG_B64))
    crops = {}
    for name, selector in section_selectors.items():
        crop = png_path.with_name(f"{png_path.stem}_{name}.png")
        crop.write_bytes(base64.b64decode(_PNG_B64))
        crops[name] = crop
    return crops


def _fake_vlm(system_prompt, images, user_text="", model=None):
    # Determine round number from existing review files.
    round_no = len(list((OUT / "harness").rglob("review.json"))) + 1
    if round_no == 1:
        return {
            "quality_score": 6,
            "dimension_scores": {"layout": 6, "typography": 5, "figures": 7, "color": 7, "coverage": 6, "overflow": 5},
            "needs_improvement": True,
            "summary": "Motivation text is dense.",
            "issues": [
                {"description": "Motivation text is too dense and overflows",
                 "severity": "warning", "target": "sec-motivation",
                 "suggestion": "Shorten to key insight", "action": "resize"},
                {"description": "Bullet points in contributions are verbose",
                 "severity": "info", "target": "sec-contributions",
                 "suggestion": "Trim to core claims", "action": "rewrite"},
            ],
        }
    return {
        "quality_score": 8,
        "dimension_scores": {"layout": 8, "typography": 8, "figures": 8, "color": 8, "coverage": 8, "overflow": 8},
        "needs_improvement": False,
        "summary": "Layout is balanced now.",
        "issues": [],
    }


with patch("src.agents.poster_harness.capture_poster_full_and_sections", side_effect=_fake_capture), \
     patch("src.agents.poster_harness.multimodal_analyze_labeled", side_effect=_fake_vlm):
    result = run_poster_harness(
        doc, analysis, blueprint, draft, OUT,
        config=HarnessConfig(threshold=8, max_rounds=3, zoom_crops=True, enable_qa_eval=False),
    )

print(f"[smoke] passed={result.passed} stop={result.stop_reason} scores={[r.quality_score for r in result.rounds]}")
print(f"[smoke] best_round={result.best_round_no} best_score={result.best_score}")
print(f"[smoke] final_html={Path(result.final_html).exists()} final_png={Path(result.final_png).exists()}")

assert result.passed and result.stop_reason == "passed"
assert [r.quality_score for r in result.rounds] == [6, 8]
assert Path(result.final_png).exists()
assert (OUT / "harness_report.json").exists()
assert (OUT / "harness" / "round_1" / "review.json").exists()
assert (OUT / "harness" / "round_1" / "poster.html").exists()
html1 = (OUT / "harness" / "round_1" / "poster.html").read_text(encoding="utf-8")
html2 = (OUT / "harness" / "round_2" / "poster.html").read_text(encoding="utf-8")
assert "harness-css-patch" not in html1, "round1 should not have patches yet"
assert "harness-css-patch" in html2, "CSS patch not injected after feedback round"
print("[smoke] OK round-2 CSS patch applied:", "harness-css-patch" in html2)

# --- Degradation path: capture unavailable -> graceful fallback ---
OUT2 = _ROOT / "temp_downloads" / "smoke_harness_fallback"
OUT2.mkdir(parents=True, exist_ok=True)
draft2 = OUT2 / "poster_draft.html"
HtmlPosterRenderer().render_to_file(blueprint, doc, draft2)

with patch("src.agents.poster_harness.capture_poster_full_and_sections", return_value={}), \
     patch("src.agents.poster_harness.multimodal_analyze_labeled", return_value=None):
    res2 = run_poster_harness(
        doc, analysis, blueprint, draft2, OUT2,
        config=HarnessConfig(threshold=8, max_rounds=3, enable_qa_eval=False),
        fallback_optimizer=lambda old, new: new.write_bytes(old.read_bytes()),
    )
print(f"[smoke] fallback passed={res2.passed} stop={res2.stop_reason} fallback={res2.fallback}")
assert res2.fallback and res2.stop_reason == "vision_unavailable"
assert Path(res2.final_html).exists()
print("[smoke] OK graceful fallback verified")

print("\n[smoke] ALL OK: end-to-end loop + degradation path verified on real paper data")
