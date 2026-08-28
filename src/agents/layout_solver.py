from __future__ import annotations

from dataclasses import dataclass

from src.schemas.poster_v2 import LayoutNode, LayoutResult


GRID_COLUMNS = 12
SOLVER_VERSION = "semantic-12col-v1"


@dataclass
class _Placed:
    row: int
    col: int
    row_span: int
    col_span: int

    def overlaps(self, other: "_Placed") -> bool:
        return not (
            self.col + self.col_span <= other.col
            or other.col + other.col_span <= self.col
            or self.row + self.row_span <= other.row
            or other.row + other.row_span <= self.row
        )


def _priority(node: LayoutNode) -> int:
    return node.constraints.priority or round(node.importance * 4)


def _spans(node: LayoutNode) -> tuple[int, int]:
    priority = _priority(node)
    composition = node.composition_type
    if priority >= 4 or node.target_area_ratio >= 0.22:
        return (2 if node.content_density != "high" else 1, 6 if composition in {"benchmark_chart", "evidence_grid", "process_diagram", "metric_callout"} else 4)
    if priority >= 3 or node.target_area_ratio >= 0.14:
        return (1, 4 if composition in {"benchmark_chart", "evidence_grid", "process_diagram"} else 3)
    if node.semantic_role in {"footer", "project", "contributions", "takeaway"} or node.composition_type in {"takeaway_card", "project_card", "footer_strip"}:
        return (1, 4)
    return (1, 3 if node.content_density == "high" else 2)


def _fits(candidate: _Placed, placed: list[_Placed]) -> bool:
    return candidate.col >= 1 and candidate.col + candidate.col_span - 1 <= GRID_COLUMNS and not any(
        candidate.overlaps(item) for item in placed
    )


def solve_layout(nodes: list[LayoutNode], canvas_width: int = 1920, canvas_height: int = 1080) -> list[LayoutResult]:
    """Place semantic nodes in reading order on a bounded 12-column grid.

    The solver is intentionally deterministic. It grows rows as needed, favors
    wide spans for high-value visual nodes, and never returns overlapping boxes.
    """
    del canvas_width, canvas_height
    candidates = [node for node in nodes if node.node_type not in {"root", "container", "spacer"}]
    ordered = sorted(candidates, key=lambda node: (-_priority(node), -node.visual_weight, -node.importance, node.reading_order, node.node_id))
    placed: list[_Placed] = []
    results: list[LayoutResult] = []
    for node in ordered:
        row_span, col_span = _spans(node)
        found = None
        for row in range(1, max(2, len(placed) * 2 + 4)):
            for col in range(1, GRID_COLUMNS - col_span + 2):
                candidate = _Placed(row, col, row_span, col_span)
                if _fits(candidate, placed):
                    found = candidate
                    break
            if found:
                break
        if found is None:
            found = _Placed(max((item.row + item.row_span for item in placed), default=1), 1, 1, min(2, GRID_COLUMNS))
        placed.append(found)
        results.append(LayoutResult(
            node_id=node.node_id,
            row=found.row,
            col=found.col,
            row_span=found.row_span,
            col_span=found.col_span,
            width_ratio=round(found.col_span / GRID_COLUMNS, 4),
            height_ratio=round(found.row_span / max(1, len(ordered)), 4),
            overflow=False,
        ))
    return sorted(results, key=lambda item: item.row * 100 + item.col)
