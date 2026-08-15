from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

# Ensure src is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
   sys.path.insert(0, str(_root))


@pytest.fixture()
def tmp_path() -> Path:
    """Workspace-based temp dir (avoids sandbox restrictions on the system temp root)."""
    base = _root / "temp_downloads" / "pytest-tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"case-{uuid.uuid4().hex[:10]}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
