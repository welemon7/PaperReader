"""Probe the text LLM endpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import LLMClient, LLMError

client = LLMClient()
try:
    out = client.chat(system="You are a one-line responder.", user="Reply with the single word: ok")
    print("TEXT LLM OK:", out[:80])
except LLMError as exc:
    print("TEXT LLM FAILED:", str(exc)[:200])
