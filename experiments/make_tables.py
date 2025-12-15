#!/usr/bin/env python3
import re
from typing import Dict, Any, List, Optional, Tuple


def _extract_patch_size(name: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"patch array with\s*(\d+)\s*x\s*(\d+)\s*patches", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _extract_reflector_freq(name: str) -> Optional[float]:
    m = re.search(r"reflector with struts\s*([0-9]+(?:\.[0-9]+)?)\s*GHz", name, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


def _fmt_patch_size(p1: int, p2: int) -> str:
    return rf"${p1} \times {p2}$"


def _fmt_matrix_size(n: Any, m: Any) -> str:
    if n is None or m is None:
        return "-"
    return rf"${int(n)} \times {int(m)}$"


def _fmt_int(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return str(int(float(x)))
    except Exception:
        return "-"


def _fmt_time(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return "-"


def _fmt_relerr(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.2e}"
    except Exception:
        return "-"


def write_simple_table(
    filename: str,
    caption: str,
    label: str,
    first_col_header: str,
    first_col_values: List[str],
    matrix_sizes: List[str],
    col_L: List[str],
    col_U: List[str],
    col_S: List[str],
    col_T: List[str],
) -> None:
    """
    LaTeX table with full grid lines and centered columns.
    Caption is placed below the tabular environment.
    """
    assert len(first_col_values) == len(matrix_sizes) == len(col_L) == len(col_U) == len(col_S) == len(col_T)

    hdr_L = r"Computing $L_{21}$"
    hdr_U = r"Computing $U_{12}$"
    hdr_S = r"Schur updating"

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|c|c|c|c|c|c|}")
    lines.append("\\hline")
    lines.append(f"{first_col_header} & Size of matrix & {hdr_L} & {hdr_U} & {hdr_S} & Total \\\\")
    lines.append("\\hline")

    for a, ms, l, u, s, t in zip(first_col_values, matrix_sizes, col_L, col_U, col_S, col_T):
        lines.append(f"{a} & {ms} & {l} & {u} & {s} & {t} \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_relerr_table(
    filename: str,
    caption: str,
    label: str,
    first_col_header: str,
    first_col_values: List[str],
    matrix_sizes: List[str],
    rel_errors: List[str],
) -> None:
    """
    3-column LaTeX table with full grid lines and centered columns:
      |c|c|c|
    Caption is placed below the tabular environment.
    """
    assert len(first_col_values) == len(matrix_sizes) == len(rel_errors)

    lines = []
    lines.append("\\begin{table}[H]]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|c|c|c|}")
    lines.append("\\hline")
    lines.append(f"{first_col_header} & Size of matrix & Relative error \\\\")
    lines.append("\\hline")

    for a, ms, e in zip(first_col_values, matrix_sizes, rel_errors):
        lines.append(f"{a} & {ms} & {e} \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _collect_patch_rows(results: Dict[str, Any]):
    rows = []
    for i, name in enumerate(results["name"]):
        if not isinstance(name, str):
            continue
        ps = _extract_patch_size(name)
        if ps is None:
            continue
        p1, p2 = ps
        rows.append(
            {
                "key": p1,
                "first": _fmt_patch_size(p1, p2),
                "matrix": _fmt_matrix_size(results["matrix_n"][i], results["matrix_m"][i]),
                "L_ops": _fmt_int(results["L_ops"][i]),
                "U_ops": _fmt_int(results["U_ops"][i]),
                "S_ops": _fmt_int(results["Schur_ops"][i]),
                "T_ops": _fmt_int(results["Total_ops"][i]),
                "L_time": _fmt_time(results["L_time"][i]),
                "U_time": _fmt_time(results["U_time"][i]),
                "S_time": _fmt_time(results["Schur_time"][i]),
                "T_time": _fmt_time(results["Total_time"][i]),
                "relerr": _fmt_relerr(results.get("rel_error", [None] * len(results["name"]))[i]),
            }
        )
    rows.sort(key=lambda r: r["key"])
    return rows


def _collect_reflector_rows(results: Dict[str, Any]):
    rows = []
    for i, name in enumerate(results["name"]):
        if not isinstance(name, str):
            continue
        freq = _extract_reflector_freq(name)
        if freq is None:
            continue
        rows.append(
            {
                "key": freq,
                "first": f"{freq:g} GHz",
                "matrix": _fmt_matrix_size(results["matrix_n"][i], results["matrix_m"][i]),
                "L_ops": _fmt_int(results["L_ops"][i]),
                "U_ops": _fmt_int(results["U_ops"][i]),
                "S_ops": _fmt_int(results["Schur_ops"][i]),
                "T_ops": _fmt_int(results["Total_ops"][i]),
                "L_time": _fmt_time(results["L_time"][i]),
                "U_time": _fmt_time(results["U_time"][i]),
                "S_time": _fmt_time(results["Schur_time"][i]),
                "T_time": _fmt_time(results["Total_time"][i]),
                "relerr": _fmt_relerr(results.get("rel_error", [None] * len(results["name"]))[i]),
            }
        )
    rows.sort(key=lambda r: r["key"])
    return rows


if __name__ == "__main__":
    import sys
    from parse_results import parse_result_file

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <result_file.txt>")
        raise SystemExit(1)

    results = parse_result_file(sys.argv[1])
    patch = _collect_patch_rows(results)
    refl = _collect_reflector_rows(results)

    # ---- Patch: operations ----
    write_simple_table(
        filename="patch_ops_table.tex",
        caption="Block matrix operation counts for patch-array test cases.",
        label="tab:patch-ops",
        first_col_header="Size of patch array",
        first_col_values=[r["first"] for r in patch],
        matrix_sizes=[r["matrix"] for r in patch],
        col_L=[r["L_ops"] for r in patch],
        col_U=[r["U_ops"] for r in patch],
        col_S=[r["S_ops"] for r in patch],
        col_T=[r["T_ops"] for r in patch],
    )

    # ---- Patch: time ----
    write_simple_table(
        filename="patch_time_table.tex",
        caption="Computation runtimes (s) for patch-array test cases.",
        label="tab:patch-time",
        first_col_header="Size of patch array",
        first_col_values=[r["first"] for r in patch],
        matrix_sizes=[r["matrix"] for r in patch],
        col_L=[r["L_time"] for r in patch],
        col_U=[r["U_time"] for r in patch],
        col_S=[r["S_time"] for r in patch],
        col_T=[r["T_time"] for r in patch],
    )

    # ---- Reflector: operations ----
    write_simple_table(
        filename="reflector_ops_table.tex",
        caption="Block matrix operation counts for reflector-with-struts test cases.",
        label="tab:refl-ops",
        first_col_header="GHz of reflector",
        first_col_values=[r["first"] for r in refl],
        matrix_sizes=[r["matrix"] for r in refl],
        col_L=[r["L_ops"] for r in refl],
        col_U=[r["U_ops"] for r in refl],
        col_S=[r["S_ops"] for r in refl],
        col_T=[r["T_ops"] for r in refl],
    )

    # ---- Reflector: time ----
    write_simple_table(
        filename="reflector_time_table.tex",
        caption="Computation runtimes (s) for reflector-with-struts test cases.",
        label="tab:refl-time",
        first_col_header="GHz of reflector",
        first_col_values=[r["first"] for r in refl],
        matrix_sizes=[r["matrix"] for r in refl],
        col_L=[r["L_time"] for r in refl],
        col_U=[r["U_time"] for r in refl],
        col_S=[r["S_time"] for r in refl],
        col_T=[r["T_time"] for r in refl],
    )

    # ---- Patch: relative error ----
    write_relerr_table(
        filename="patch_relerr_table.tex",
        caption="Relative errors for patch-array test cases.",
        label="tab:patch-relerr",
        first_col_header="Size of patch array",
        first_col_values=[r["first"] for r in patch],
        matrix_sizes=[r["matrix"] for r in patch],
        rel_errors=[r["relerr"] for r in patch],
    )

    # ---- Reflector: relative error ----
    write_relerr_table(
        filename="reflector_relerr_table.tex",
        caption="Relative errors for reflector-with-struts test cases.",
        label="tab:refl-relerr",
        first_col_header="GHz of reflector",
        first_col_values=[r["first"] for r in refl],
        matrix_sizes=[r["matrix"] for r in refl],
        rel_errors=[r["relerr"] for r in refl],
    )

    print("Wrote:")
    print("  patch_ops_table.tex")
    print("  patch_time_table.tex")
    print("  reflector_ops_table.tex")
    print("  reflector_time_table.tex")
    print("  patch_relerr_table.tex")
    print("  reflector_relerr_table.tex")
