#!/usr/bin/env python3
"""
Create LaTeX tables from parsed LU experiment results.

Outputs:
  - patch_table.tex
  - reflector_table.tex
"""

import re
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


# ----------------------------- helpers -----------------------------


def _fmt_time(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return "-"


def _fmt_int(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return str(int(x))
    except Exception:
        return "-"


def _fmt_matrix_size(n: Any, m: Any) -> str:
    if n is None or m is None:
        return "-"
    return f"{int(n)} x {int(m)}"


def _extract_patch_size(name: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"patch array with\s*(\d+)\s*x\s*(\d+)\s*patches", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _extract_reflector_freq(name: str) -> Optional[float]:
    # Example: "reflector with struts 25GHz"
    m = re.search(r"reflector with struts\s*([0-9]+(?:\.[0-9]+)?)\s*GHz", name, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


def _latex_table_from_df(
    df: pd.DataFrame,
    caption: str,
    label: str,
    filename: str,
) -> None:
    """
    Emit a LaTeX table with the multi-row header layout:

      & & \multicolumn{2}{c}{Computing L} & ... etc
    """
    cols = df.columns  # MultiIndex (top, sub)

    # Column alignment:
    # first two are textual -> l l, the rest numeric -> r
    align = "ll" + "r" * (len(cols) - 2)

    # Build header lines
    # First header row: top-level group names with multicolumn spans
    top_levels: List[str] = [t for t, _ in cols]
    sub_levels: List[str] = [s for _, s in cols]

    # Determine spans for top-level
    spans = []
    j = 0
    while j < len(top_levels):
        name = top_levels[j]
        k = j
        while k < len(top_levels) and top_levels[k] == name:
            k += 1
        spans.append((name, j, k - j))  # (name, start, span)
        j = k

    # Construct first header row with multicolumns for grouped columns
    header1_parts = []
    for name, start, span in spans:
        if span == 1:
            header1_parts.append(f"\\multicolumn{{1}}{{c}}{{{name}}}")
        else:
            header1_parts.append(f"\\multicolumn{{{span}}}{{c}}{{{name}}}")
    header1 = " & ".join(header1_parts) + " \\\\"

    # Second header row: sub-levels (blank under the first two)
    header2 = " & ".join([s if s else "" for s in sub_levels]) + " \\\\"

    # Column clines (draw partial horizontal lines under grouped headings)
    clines = []
    for name, start, span in spans:
        # cline is 1-indexed columns in LaTeX tabular
        if span > 1:
            c1 = start + 1
            c2 = start + span
            clines.append(f"\\cline{{{c1}-{c2}}}")
    cline_line = " ".join(clines)

    # Body rows
    body_lines = []
    for _, row in df.iterrows():
        vals = [str(v) for v in row.tolist()]
        body_lines.append(" & ".join(vals) + " \\\\")

    latex = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{align}}}",
            "\\hline",
            header1,
            cline_line if cline_line else "",
            header2,
            "\\hline",
            *body_lines,
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex)


# ----------------------------- builders -----------------------------


def build_patch_df(results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(results["name"]):
        ps = _extract_patch_size(name or "")
        if ps is None:
            continue
        p1, p2 = ps

        rows.append(
            {
                ("Size of patch array", ""): f"{p1} x {p2}",
                ("Size of matrix", ""): _fmt_matrix_size(results["matrix_n"][i], results["matrix_m"][i]),
                ("Computing L", "Matrix operations"): _fmt_int(results["L_ops"][i]),
                ("Computing L", "Time (s)"): _fmt_time(results["L_time"][i]),
                ("Computing U", "Matrix operations"): _fmt_int(results["U_ops"][i]),
                ("Computing U", "Time (s)"): _fmt_time(results["U_time"][i]),
                ("Schur updating", "Matrix operations"): _fmt_int(results["Schur_ops"][i]),
                ("Schur updating", "Time (s)"): _fmt_time(results["Schur_time"][i]),
                ("Total", "Matrix operations"): _fmt_int(results["Total_ops"][i]),
                ("Total", "Time (s)"): _fmt_time(results["Total_time"][i]),
            }
        )

    rows.sort(key=lambda r: int(r[("Size of patch array", "")].split(" x ")[0]))

    columns = pd.MultiIndex.from_tuples(
        [
            ("Size of patch array", ""),
            ("Size of matrix", ""),
            ("Computing L", "Matrix operations"),
            ("Computing L", "Time (s)"),
            ("Computing U", "Matrix operations"),
            ("Computing U", "Time (s)"),
            ("Schur updating", "Matrix operations"),
            ("Schur updating", "Time (s)"),
            ("Total", "Matrix operations"),
            ("Total", "Time (s)"),
        ]
    )
    return pd.DataFrame(rows, columns=columns)


def build_reflector_df(results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(results["name"]):
        freq = _extract_reflector_freq(name or "")
        if freq is None:
            continue

        rows.append(
            {
                ("Case", ""): f"{freq:g} GHz",
                ("Size of matrix", ""): _fmt_matrix_size(results["matrix_n"][i], results["matrix_m"][i]),
                ("Computing L", "Matrix operations"): _fmt_int(results["L_ops"][i]),
                ("Computing L", "Time (s)"): _fmt_time(results["L_time"][i]),
                ("Computing U", "Matrix operations"): _fmt_int(results["U_ops"][i]),
                ("Computing U", "Time (s)"): _fmt_time(results["U_time"][i]),
                ("Schur updating", "Matrix operations"): _fmt_int(results["Schur_ops"][i]),
                ("Schur updating", "Time (s)"): _fmt_time(results["Schur_time"][i]),
                ("Total", "Matrix operations"): _fmt_int(results["Total_ops"][i]),
                ("Total", "Time (s)"): _fmt_time(results["Total_time"][i]),
            }
        )

    rows.sort(key=lambda r: float(r[("Case", "")].split()[0]))

    columns = pd.MultiIndex.from_tuples(
        [
            ("Case", ""),
            ("Size of matrix", ""),
            ("Computing L", "Matrix operations"),
            ("Computing L", "Time (s)"),
            ("Computing U", "Matrix operations"),
            ("Computing U", "Time (s)"),
            ("Schur updating", "Matrix operations"),
            ("Schur updating", "Time (s)"),
            ("Total", "Matrix operations"),
            ("Total", "Time (s)"),
        ]
    )
    return pd.DataFrame(rows, columns=columns)


# ----------------------------- main -----------------------------


if __name__ == "__main__":
    import sys
    from parse_results import parse_result_file

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <result_file.txt>")
        raise SystemExit(1)

    results = parse_result_file(sys.argv[1])

    patch_df = build_patch_df(results)
    refl_df = build_reflector_df(results)

    _latex_table_from_df(
        patch_df,
        caption="Block kernel work and runtime for patch-array test cases.",
        label="tab:patch-kernels",
        filename="patch_table.tex",
    )

    _latex_table_from_df(
        refl_df,
        caption="Block kernel work and runtime for reflector-with-struts test cases.",
        label="tab:reflector-kernels",
        filename="reflector_table.tex",
    )

    print("Wrote:")
    print("  patch_table.tex")
    print("  reflector_table.tex")
