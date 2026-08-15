"""Diagnose the understand-phase LLM call for 2601.17470."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s", datefmt="%H:%M:%S")

from src.storage.sqlite import PaperDatabase
from src.llm.client import LLMClient
from src.agents.understand_agent import _SYSTEM_PROMPT, _build_analysis_prompt

db = PaperDatabase()
doc = db.get_paper_by_arxiv("2601.17470")
db.close()
print("[diag] doc loaded:", doc.title if doc else None)

prompt = _build_analysis_prompt(doc)
print("[diag] prompt chars:", len(prompt))

client = LLMClient()
try:
    result = client.chat_json(system=_SYSTEM_PROMPT, user=prompt)
    print("[diag] chat_json OK, keys:", len(result))
    print("[diag] top keys:", list(result.keys())[:15])
except Exception as e:
    print(f"[diag] chat_json FAILED: {type(e).__name__}: {str(e)[:600]}")
    # 再试一次原始 chat，看返回内容
    try:
        content = client.chat(system=_SYSTEM_PROMPT, user=prompt)
        print("[diag] raw chat OK, content head:", content[:300].replace("\n", " "))
    except Exception as e2:
        print(f"[diag] raw chat FAILED: {type(e2).__name__}: {str(e2)[:300]}")
