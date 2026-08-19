from types import SimpleNamespace

from src.agents.understand_agent import _validate_final_tables


def test_final_table_review_is_source_grounded_and_bounded():
    candidates = [{
        "table_id": "table-001",
        "headers": ["Method", "PSNR", "SSIM"],
        "rows": [
            {"row_index": 1, "group": "Ours", "cells": ["Ours", "32.4", "0.94"]},
            {"row_index": 2, "group": "Ours", "cells": ["Ours", "33.1", "0.95"]},
        ],
    }]
    result = _validate_final_tables(
        SimpleNamespace(tables=[]),
        candidates,
        [{
            "table_id": "table-001",
            "row_indices": [1, 2, 999],
            "column_indices": [0, 1, 2, 3, 4, 5, 6],
            "headers": ["invented"],
            "rows": [["invented"]],
        }],
    )
    assert result[0]["headers"] == ["Method", "PSNR", "SSIM"]
    assert result[0]["rows"] == [["Ours", "32.4", "0.94"], ["Ours", "33.1", "0.95"]]
    assert 999 not in result[0]["row_indices"]


