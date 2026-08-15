"""Phase 2-5 real test for 2601.17470 (parse already cached in DB; skip MinIO stall)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s", datefmt="%H:%M:%S")

from src.storage.sqlite import PaperDatabase
from src.agents.understand_agent import run_understand_paper
from src.agents.poster_planner import generate_blueprint, normalize_analysis_for_poster
from src.renderers.html_renderer import HtmlPosterRenderer
from src.agents.html_optimizer import optimize_html_with_llm
from src.agents.poster_harness import run_poster_harness
from src.schemas.poster_harness import HarnessConfig
from src.utils.output_paths import resolve_paper_output_dir
from pathlib import Path as P

ARXIV = "2601.17470"
out = resolve_paper_output_dir(P("output"), ARXIV)
out.mkdir(parents=True, exist_ok=True)

# --- Phase 2: Understand (real LLM; reuse DB result if already present) ---
db = PaperDatabase()
analysis = db.get_analysis_by_arxiv(ARXIV)
db.close()
if analysis is None:
    print(f"[t] Phase 2: understand {ARXIV}", flush=True)
    analysis = run_understand_paper(ARXIV)
    analysis = normalize_analysis_for_poster(analysis)
    (out / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    print(f"[t] analysis ok: {analysis.problem_statement[:80]!r}", flush=True)
else:
    analysis = normalize_analysis_for_poster(analysis)
    print(f"[t] analysis reused from DB: {analysis.problem_statement[:80]!r}", flush=True)

# --- Phase 3: Plan ---
db = PaperDatabase()
doc = db.get_paper_by_arxiv(ARXIV)
db.close()
print(f"[t] doc: {doc.title}", flush=True)
blueprint = generate_blueprint(doc, analysis)
print(f"[t] blueprint: {len(blueprint.sections)} sections, {len(blueprint.figure_placements)} figures", flush=True)

# --- Phase 4: Render draft ---
draft = out / "poster_draft.html"
HtmlPosterRenderer().render_to_file(blueprint, doc, draft, optimize_with_llm=False)
print(f"[t] draft: {draft} ({draft.stat().st_size} bytes)", flush=True)

# --- Phase 5: Harness (real; capture will be unavailable in sandbox -> fallback) ---
print("[t] Phase 5: harness", flush=True)
def _fallback(old: P, new: P):
    optimize_html_with_llm(html_path=old, prompt_path=_ROOT / "LLM-up.txt", output_path=new)
    print(f"[t] fallback optimize -> {new} ({new.stat().st_size} bytes)", flush=True)

result = run_poster_harness(
    doc, analysis, blueprint, draft, out,
    config=HarnessConfig(threshold=8, max_rounds=5, zoom_crops=True, enable_qa_eval=True),
    fallback_optimizer=_fallback,
)
print(f"[t] harness: passed={result.passed} stop={result.stop_reason} fallback={result.fallback}", flush=True)
print(f"[t] harness: rounds={[r.quality_score for r in result.rounds]} final_html={Path(result.final_html).exists()}", flush=True)
if result.report_path:
    print(f"[t] report: {result.report_path}", flush=True)
if result.qa_eval_path:
    qa = json.loads(Path(result.qa_eval_path).read_text(encoding="utf-8"))
    print(f"[t] QA: {qa['correct_count']}/{qa['total_count']} accuracy={qa['accuracy']:.2f}", flush=True)
print("[t] DONE", flush=True)
