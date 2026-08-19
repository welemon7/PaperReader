from __future__ import annotations

import re
from typing import Any


def extract_tables(latex: str, section_id_for_position=None) -> list[dict[str, Any]]:
    """Extract complete table/tabular environments without truncating rows."""
    tables: list[dict[str, Any]] = []
    covered: list[tuple[int, int]] = []
    next_id = 1
    for env, start, end, body in _find_environments(latex, {"table", "table*"}):
        nested = _first_nested_tabular(body)
        parsed = _parse_tabular(nested[3]) if nested else None
        if not parsed:
            continue
        covered.append((start, end))
        tables.append(_record(next_id, env, start, end, parsed,
                              _read_command_arg(body, "caption"),
                              _read_command_arg(body, "label"),
                              section_id_for_position(start) if section_id_for_position else "",
                              latex[start:end]))
        next_id += 1
    for env, start, end, body in _find_environments(latex, {"tabular", "tabular*", "tabularx", "array"}):
        if any(left <= start < right for left, right in covered):
            continue
        parsed = _parse_tabular(body)
        if not parsed:
            continue
        tables.append(_record(next_id, env, start, end, parsed, "", "",
                              section_id_for_position(start) if section_id_for_position else "",
                              latex[start:end]))
        next_id += 1
    return sorted(tables, key=lambda item: item["start"])


def _record(index, env, start, end, parsed, caption, label, section_id, raw_latex):
    headers, rows, column_spec, row_groups = parsed
    return {"table_id": f"table-{index:03d}", "environment": env,
            "caption": caption, "label": label, "section_id": section_id,
            "column_spec": column_spec, "headers": headers, "rows": rows,
            "row_groups": row_groups,
            "raw_latex": raw_latex, "start": start, "end": end}


def _find_environments(text: str, names: set[str]):
    for match in re.finditer(r"\\begin\{([^}]+)\}", text):
        env = match.group(1)
        if env not in names:
            continue
        close = re.search(rf"\\end\{{{re.escape(env)}\}}", text[match.end():])
        if close:
            body_start = match.end()
            body_end = body_start + close.start()
            yield env, match.start(), body_end + len(close.group(0)), text[body_start:body_end]


def _first_nested_tabular(body: str):
    return next(iter(_find_environments(body, {"tabular", "tabular*", "tabularx", "array"})), None)


def _parse_tabular(body: str):
    spec_match = re.match(r"^\s*(?:\[[^]]*\])?\s*\{([^{}]*)\}", body, re.DOTALL)
    column_spec = spec_match.group(1).strip() if spec_match else ""
    content = re.sub(r"^\s*(?:\[[^]]*\])?\s*(?:\{(?:[^{}]|\{[^{}]*\})*\}){1,2}", "", body, count=1, flags=re.DOTALL)
    rows = []
    group_labels = []
    pending_multirows: dict[int, tuple[int, str]] = {}
    for raw_row in re.split(r"(?<!\\)\\\\(?:\s*\[[^]]*\])?", content):
        raw_row = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|addlinespace|cline\{[^}]*\})", "", raw_row)
        cells, row_group = _expand_row(raw_row, pending_multirows)
        if any(cells):
            rows.append(cells)
            group_labels.append(row_group)
    if len(rows) < 2:
        return None
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return rows[0], rows[1:], column_spec, group_labels[1:]


def _expand_row(raw_row: str, pending_multirows: dict[int, tuple[int, str]]):
    """Expand multicolumn cells and carry multirow labels into later rows."""
    raw_row = re.sub(r"\\(?:rowcolor|cellcolor)\s*(?:\[[^]]*\])?\s*\{[^{}]*\}", "", raw_row)
    tokens = _split_cells(raw_row)
    cells: list[str] = []
    group_label = ""
    col = 0
    for token in tokens:
        multi = _command_args(token, "multicolumn", 3)
        if multi and multi[0].isdigit():
            span, value = int(multi[0]), _clean_cell(multi[2])
            cells.extend([value] * span)
            col += span
            continue
        multirow = _command_args(token, "multirow", 3)
        if multirow and multirow[0].isdigit():
            span, value = int(multirow[0]), _clean_cell(multirow[2])
            cells.append(value)
            if not group_label:
                group_label = value
            if span > 1:
                pending_multirows[col] = (span - 1, value)
            col += 1
            continue
        cells.append(_clean_cell(token))
        col += 1

    for position, (remaining, value) in list(pending_multirows.items()):
        if position < len(cells) and not cells[position]:
            cells[position] = value
        elif position >= len(cells):
            cells.extend([""] * (position - len(cells)))
            cells.append(value)
        if not group_label:
            group_label = value
        if remaining <= 0:
            pending_multirows.pop(position, None)
        else:
            pending_multirows[position] = (remaining - 1, value)
    return cells, group_label


def _command_args(token: str, command: str, count: int) -> list[str] | None:
    match = re.match(rf"\s*\\{command}\b", token)
    if not match:
        return None
    args: list[str] = []
    pos = match.end()
    while len(args) < count:
        while pos < len(token) and token[pos].isspace():
            pos += 1
        if pos >= len(token) or token[pos] != "{":
            return None
        value, pos = _read_balanced(token, pos)
        args.append(value)
    return args if not token[pos:].strip() else None


def _read_balanced(text: str, start: int) -> tuple[str, int]:
    depth, pos = 1, start + 1
    while pos < len(text) and depth:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start + 1:pos - 1], pos


def _split_cells(raw_row: str) -> list[str]:
    cells, start, depth, i = [], 0, 0, 0
    while i < len(raw_row):
        if raw_row[i] == "{":
            depth += 1
        elif raw_row[i] == "}":
            depth = max(0, depth - 1)
        elif raw_row[i] == "&" and depth == 0 and (i == 0 or raw_row[i - 1] != "\\"):
            cells.append(raw_row[start:i])
            start = i + 1
        i += 1
    cells.append(raw_row[start:])
    return cells


def _read_command_arg(text: str, command: str) -> str:
    match = re.search(rf"\\{command}\s*\{{", text)
    if not match:
        return ""
    start, depth, end = match.end() - 1, 1, match.end()
    while end < len(text) and depth:
        depth += text[end] == "{"
        depth -= text[end] == "}"
        end += 1
    return _clean_cell(text[start + 1:end - 1]) if depth == 0 else ""


def _clean_cell(value: str) -> str:
    # Formatting/color commands are presentation noise, never table data.
    value = re.sub(r"\\(?:cellcolor|rowcolor|textcolor|color)\s*(?:\[[^]]*\])?\s*\{[^{}]*\}", "", value)
    value = re.sub(r"\\(?:textbf|textit|emph|mathrm|mathbf|mathit|underline|textrm|textsf|texttt|scriptsize|small|footnotesize|tiny)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\(?:cite|ref|eqref|label|autoref|pageref|cref|Cref)\s*\{[^{}]*\}", "", value)
    value = value.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&").replace(r"\pm", "+/-")
    value = value.replace("$", "")
    # Remove standalone visual glyph commands and table-rule commands.
    value = re.sub(r"\\(?:checkmark|crossmark|ding|cmark|xmark|yes|no|ok|times|tick)\s*(?:\{[^{}]*\})?", "", value)
    value = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline|cmidrule|addlinespace|tabularnewline|arraybackslash)", "", value)
    value = re.sub(r"\\[a-zA-Z@]+", "", value)
    value = value.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", value).strip()
