"""Real pipeline test for arXiv 2601.17470 (mirrors the app.py generate flow)."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import app as app_module
from app import TaskStatus

ARXIV_ID = "2601.17470"
task_id = "test2601"

task = TaskStatus(task_id)
task.arxiv_id = ARXIV_ID
app_module.tasks[task_id] = task

print(f"[driver] starting pipeline for {ARXIV_ID}", flush=True)
thread = threading.Thread(
    target=app_module.generate_poster_task,
    args=(task_id, ARXIV_ID, None, 8, 5, True),
    daemon=True,
)
thread.start()

while task.status in ("pending", "running"):
    print(f"[driver] {task.status} | {task.progress}% | {task.message}", flush=True)
    time.sleep(5)

print(f"[driver] FINAL status={task.status}", flush=True)
if task.status == "complete":
    print("[driver] result:", json.dumps(task.result, ensure_ascii=False, indent=2)[:1500], flush=True)
    print(f"[driver] harness_status={task.harness_status} rounds={len(task.harness_rounds)}", flush=True)
    for r in task.harness_rounds:
        print(f"[driver]   round {r['round_no']}: score={r['quality_score']} needs={r['needs_improvement']} | {r['summary']}", flush=True)
    report = Path(task.harness_report) if task.harness_report else None
    if report and report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        print("[driver] report: stop_reason=%s passed=%s fallback=%s scores=%s best=%s" % (
            data.get("stop_reason"), data.get("passed"), data.get("fallback"),
            data.get("scores"), data.get("best_score")), flush=True)
else:
    print(f"[driver] ERROR: {task.error}", flush=True)
