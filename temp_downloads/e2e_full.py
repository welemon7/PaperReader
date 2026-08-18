"""Full end-to-end run: real arXiv paper through the new scene + 100-pt harness."""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s", datefmt="%H:%M:%S")

from src.agents.parse_agent import run_parse_paper
from src.agents.understand_agent import run_understand_paper
from src.agents.poster_planner import generate_blueprint, normalize_analysis_for_poster
from src.agents.poster_harness import run_poster_harness
from src.agents.scene_harness import SceneHarnessAdapter
from src.schemas.poster_harness import HarnessConfig
from src.llm.client import LLMClient
from src.config import settings

ARXIV_ID = sys.argv[1] if len(sys.argv) > 1 else "2412.17630"
OUT = Path("output") / ARXIV_ID
OUT.mkdir(parents=True, exist_ok=True)

print(f"=== Phase 1: parse {ARXIV_ID} ===")
doc = run_parse_paper(ARXIV_ID, force=True)
print(f"    sections={len(doc.sections)} figures={len(doc.figures)}")

print("=== Phase 2: understand ===")
analysis = None
for attempt in range(1, 4):
    try:
        analysis = run_understand_paper(ARXIV_ID)
        break
    except Exception as exc:
        print(f"    attempt {attempt} failed: {exc}")
analysis = normalize_analysis_for_poster(analysis)
(OUT / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
print(f"    contributions={len(analysis.contributions)} formulas={len(analysis.key_formulas)}")

print("=== Phase 3: blueprint + scene ===")
blueprint = generate_blueprint(doc, analysis)
llm = LLMClient() if LLMClient.is_configured() else None
adapter = SceneHarnessAdapter(blueprint, doc, analysis, theme=settings.poster_theme, llm=llm)
draft = adapter.render_html(blueprint, doc, OUT)
(OUT / "poster_draft.html").write_text(draft, encoding="utf-8")
print("    draft rendered, scene panels:", len(adapter.scene.panels))

print("=== Phase 4/5: visual harness (100-pt contract) ===")
config = HarnessConfig(
    threshold=settings.harness_threshold,
    max_rounds=6,
    enable_qa_eval=True,
    qa_threshold=settings.harness_qa_threshold,
    zoom_crops=True,
    max_crops=4,
    vision_model=settings.harness_vision_model or None,
    advanced_visual=True,
    pass_total=settings.harness_pass_total,
    pass_dim_fraction=settings.harness_pass_dim_fraction,
    plateau_rounds=settings.harness_plateau_rounds,
    improvement_delta=settings.harness_improvement_delta,
    max_figure_crops=settings.harness_max_figure_crops,
)

result = run_poster_harness(
    doc=doc, analysis=analysis, blueprint=blueprint,
    html_path=OUT / "poster_draft.html", output_dir=OUT, config=config,
    on_round=lambda r, t, s, n, m: print(f"    round {r}/{t}: total={s}/100 needs_improvement={n} - {m}"),
    render_html=adapter.render_html,
    apply_feedback_override=adapter.apply_feedback,
    rollback_patches=adapter.rollback,
)
adapter.snapshot_after()

print("\n=== RESULT ===")
print(f"passed: {result.passed}")
print(f"stop_reason: {result.stop_reason}")
print(f"stop_label: {result.stop_label}")
print(f"rounds: {result.total_rounds}")
for r in result.rounds:
    print(f"  R{r.round_no}: total={r.total_score} hard_errors={r.hard_error_count} "
          f"needs_improvement={r.needs_improvement} verdict_passed={r.verdict.get('passed') if r.verdict else None} "
          f"actions={r.applied_actions[:3]}")
print(f"best_round: {result.best_round_no} total={result.best_total}")
print(f"final_html: {result.final_html}")
print(f"final_png: {result.final_png}")
print(f"qa_eval: {result.qa_eval_path}")
print(f"report: {result.report_path}")
