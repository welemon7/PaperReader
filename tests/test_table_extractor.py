from src.parsers.table_extractor import extract_tables


def test_extracts_all_table_rows_and_metadata():
    latex = r"""
    \section{Method}
    \begin{table}[t]
    \caption{Comparison on benchmarks}\label{tab:results}
    \begin{tabular}{lcc}
    \toprule
    Method & PSNR & SSIM \\
    Baseline & 30.1 & 0.90 \\
    Ours & \textbf{32.4} & 0.94 \\
    \bottomrule
    \end{tabular}
    \end{table}
    """
    tables = extract_tables(latex)
    assert len(tables) == 1
    table = tables[0]
    assert table["caption"] == "Comparison on benchmarks"
    assert table["label"] == "tab:results"
    assert table["headers"] == ["Method", "PSNR", "SSIM"]
    assert table["rows"][-1] == ["Ours", "32.4", "0.94"]


def test_extracts_standalone_tabular():
    tables = extract_tables(r"\begin{tabular}{lc} Name & Score \\ A & 1 \\ B & 2 \end{tabular}")
    assert len(tables) == 1
    assert len(tables[0]["rows"]) == 2


def test_cleans_pollution_and_expands_merged_cells():
    latex = r"""
    \begin{table}
    \begin{tabular}{llc}
    \rowcolor{gray!10} \multicolumn{3}{c|}{\textbf{Ours}} \\
    \multirow{2}{*}{\emph{Ours}} & \textcolor{red}{PSNR} & \checkmark \\
    & 32.4 & \ding{172} \\
    \end{tabular}
    \end{table}
    """
    table = extract_tables(latex)[0]
    assert table["headers"] == ["Ours", "Ours", "Ours"]
    assert table["rows"][0] == ["Ours", "PSNR", ""]
    assert table["rows"][1] == ["Ours", "32.4", ""]
    assert table["row_groups"] == ["Ours", "Ours"]
    assert all("textcolor" not in cell and "ding" not in cell for row in table["rows"] for cell in row)


def test_cleans_more_formatting_and_symbol_commands():
    latex = r"""
    \begin{table}
    \caption{Ablation on datasets}
    \begin{tabular}{lcc}
    Method & PSNR & SSIM \\
    \rowcolor{gray!10} \textbf{Ours} & \textcolor{red}{32.4} & \checkmark \\
    Baseline & \small 29.8 & \ding{172} \\
    \end{tabular}
    \end{table}
    """
    table = extract_tables(latex)[0]
    assert table["caption"] == "Ablation on datasets"
    assert table["rows"][0] == ["Ours", "32.4", ""]
    assert table["rows"][1] == ["Baseline", "29.8", ""]
    assert all("checkmark" not in cell and "ding" not in cell and "textcolor" not in cell for row in table["rows"] for cell in row)
