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


def test_final_table_review_preserves_metadata_and_table_shape():
    candidates = [{
        "table_id": "table-001",
        "caption": "Comparison on benchmarks",
        "headers": ["Dataset", "Method", "PSNR", "SSIM"],
        "row_groups": ["ISTD+", "ISTD+"],
        "rows": [
            {"row_index": 0, "group": "ISTD+", "cells": ["ISTD+", "Ours", "32.4", "0.94"]},
            {"row_index": 1, "group": "ISTD+", "cells": ["ISTD+", "Ours-Large", "33.1", "0.95"]},
        ],
    }]
    result = _validate_final_tables(
        SimpleNamespace(tables=[]),
        candidates,
        [{
            "table_id": "table-001",
            "row_indices": [0, 1],
            "column_indices": [0, 1, 2, 3],
            "column_groups": [[0], [1], [2, 3]],
            "datasets": ["ISTD+"],
            "metrics": ["PSNR", "SSIM"],
            "row_groups": ["ISTD+"],
            "notes": "paper method only",
            "headers": ["Dataset", "Method", "PSNR / SSIM"],
            "rows": [["ISTD+", "Ours", "32.4 / 0.94"], ["ISTD+", "Ours-Large", "33.1 / 0.95"]],
        }],
    )
    assert result[0]["caption"] == "Comparison on benchmarks"
    assert result[0]["datasets"] == ["ISTD+"]
    assert result[0]["metrics"] == ["PSNR", "SSIM"]
    assert result[0]["row_groups"] == ["ISTD+"]
    assert result[0]["notes"] == "paper method only"

