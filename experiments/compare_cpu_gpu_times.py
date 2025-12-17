#!/usr/bin/env python3
"""
Compare CPU vs GPU experiment outputs (both parseable by parse_results.py) and
emit LaTeX tables + CPU/GPU comparison plots.

Tables (8):
  patch_time_{L,U,S,T}_table.tex
  reflector_time_{L,U,S,T}_table.tex

Plots (10):
  patch_time_{L,U,S,T}_plot.png
  reflector_time_{L,U,S,T}_plot.png
  patch_lu_time_plot.png
  reflector_lu_time_plot.png

Usage:
  python3 compare_cpu_gpu_times.py experiment_cpu.txt experiment_gpu.txt
"""

import re
from typing import Dict, Any, Optional, Tuple, List
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter



# ----------------------------- selectors -----------------------------


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


# ----------------------------- formatters -----------------------------


def _fmt_patch_size(p1: int, p2: int) -> str:
    return rf"${p1} \times {p2}$"


def _fmt_matrix_size(n: Any, m: Any) -> str:
    if n is None or m is None:
        return "-"
    return rf"${int(n)} \times {int(m)}$"


def _fmt_time(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return "-"


# ----------------------------- indexing -----------------------------


def _index_patch_times(results: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Map (p1,p2) -> {"matrix": "...", "n": int, "L": time, "U": time, "S": time, "T": time, "LU": lu_time}
    """
    idx: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i, name in enumerate(results.get("name", [])):
        if not isinstance(name, str):
            continue
        ps = _extract_patch_size(name)
        if ps is None:
            continue

        n = results.get("matrix_n", [None])[i]
        m = results.get("matrix_m", [None])[i]
        if n is None:
            continue

        p1, p2 = ps
        idx[(p1, p2)] = {
            "matrix": _fmt_matrix_size(n, m),
            "n": int(n),
            "L": results.get("L_time", [None])[i],
            "U": results.get("U_time", [None])[i],
            "S": results.get("Schur_time", [None])[i],
            "T": results.get("Total_time", [None])[i],
            "LU": results.get("lu_time", [None])[i],
        }
    return idx


def _index_reflector_times(results: Dict[str, Any]) -> Dict[float, Dict[str, Any]]:
    """
    Map freq -> {"matrix": "...", "n": int, "L": time, "U": time, "S": time, "T": time, "LU": lu_time}
    """
    idx: Dict[float, Dict[str, Any]] = {}
    for i, name in enumerate(results.get("name", [])):
        if not isinstance(name, str):
            continue
        freq = _extract_reflector_freq(name)
        if freq is None:
            continue

        n = results.get("matrix_n", [None])[i]
        m = results.get("matrix_m", [None])[i]
        if n is None:
            continue

        idx[freq] = {
            "matrix": _fmt_matrix_size(n, m),
            "n": int(n),
            "L": results.get("L_time", [None])[i],
            "U": results.get("U_time", [None])[i],
            "S": results.get("Schur_time", [None])[i],
            "T": results.get("Total_time", [None])[i],
            "LU": results.get("lu_time", [None])[i],
        }
    return idx


# ----------------------------- LaTeX writer -----------------------------


def write_cpu_gpu_time_table(
    filename: str,
    caption: str,
    label: str,
    first_col_header: str,
    first_col_values: List[str],
    matrix_sizes: List[str],
    cpu_times: List[str],
    gpu_times: List[str],
) -> None:
    assert len(first_col_values) == len(matrix_sizes) == len(cpu_times) == len(gpu_times)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|c|c|c|c|}")
    lines.append("\\hline")
    lines.append(f"{first_col_header} & Size of matrix & CPU time & GPU time \\\\")
    lines.append("\\hline")
    for a, ms, ct, gt in zip(first_col_values, matrix_sizes, cpu_times, gpu_times):
        lines.append(f"{a} & {ms} & {ct} & {gt} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------- plot helper -----------------------------


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")

def _log_tick_formatter(x, pos):
    if x == 0:
        return ""
    exp = int(math.floor(math.log10(x)))
    mant = x / (10 ** exp)
    if abs(mant - 1) < 1e-6:
        return rf"$10^{{{exp}}}$"
    elif mant in (2, 5):
        return rf"${int(mant)}\times10^{{{exp}}}$"
    return ""

def plot_cpu_gpu_times(
    outfile: str,
    title: str,
    x_n: List[int],
    y_cpu: List[float],
    y_gpu: List[float],
    ylabel: str = "Computation time (s)",
    dash_last_two: bool = False,   # patch: dashed tail
    dash_all: bool = False,        # reflector: fully dashed
) -> None:
    """
    x-axis: matrix size (n)   [log scale]
    y-axis: time (s)          [log scale]
    two lines: CPU and GPU

    - dash_last_two=True: last segment (covering last two points) is dashed
    - dash_all=True: whole series is dashed
    """

    # Filter out rows where BOTH are NaN or x is missing
    x_f, y_cpu_f, y_gpu_f = [], [], []
    for n, ct, gt in zip(x_n, y_cpu, y_gpu):
        if n is None:
            continue
        if (ct is None or math.isnan(ct)) and (gt is None or math.isnan(gt)):
            continue
        x_f.append(n)
        y_cpu_f.append(ct)
        y_gpu_f.append(gt)

    plt.figure()

    def _plot_series(yvals, label):
        # If fully dashed, just plot once (color chosen automatically)
        if dash_all:
            line, = plt.plot(x_f, yvals, marker="o", linestyle="--", label=label)
            return

        # If not splitting or too few points, plot solid once
        if not dash_last_two or len(x_f) < 3:
            line, = plt.plot(x_f, yvals, marker="o", label=label)
            return

        # Split: solid up to len-2, dashed tail covering last two points
        cut = len(x_f) - 2

        # Solid part (with label) -> capture its color
        solid_line, = plt.plot(x_f[:cut], yvals[:cut], marker="o", label=label)
        c = solid_line.get_color()

        # Dashed tail (no label) with SAME color
        plt.plot(
            x_f[cut - 1:],
            yvals[cut - 1:],
            marker="o",
            linestyle="--",
            color=c,
        )

    _plot_series(y_cpu_f, "CPU")
    _plot_series(y_gpu_f, "GPU")

    plt.xscale("log")
    plt.yscale("log")

    # More labeled ticks on log x-axis (1, 2, 5 × 10^k)
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(_log_tick_formatter))

    # # Y-axis: label 1, 2, 5 × 10^k
    # ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 5.0)))
    # ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_formatter))

    plt.title(title)
    plt.xlabel("Matrix size (n)")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()



# ----------------------------- builders -----------------------------


def _build_patch_rows_for_kernel(
    cpu_idx: Dict[Tuple[int, int], Dict[str, Any]],
    gpu_idx: Dict[Tuple[int, int], Dict[str, Any]],
    kernel_key: str,
):
    keys = sorted(set(cpu_idx.keys()) | set(gpu_idx.keys()), key=lambda k: (k[0], k[1]))

    first_col = [_fmt_patch_size(k[0], k[1]) for k in keys]
    matrix_sizes, cpu_times_s, gpu_times_s = [], [], []
    x_n, y_cpu, y_gpu = [], [], []

    for k in keys:
        c = cpu_idx.get(k, {})
        g = gpu_idx.get(k, {})

        matrix_sizes.append(c.get("matrix") or g.get("matrix") or "-")
        x_n.append(c.get("n") or g.get("n"))

        ct = c.get(kernel_key)
        gt = g.get(kernel_key)

        cpu_times_s.append(_fmt_time(ct))
        gpu_times_s.append(_fmt_time(gt))

        y_cpu.append(_to_float(ct))
        y_gpu.append(_to_float(gt))

    return first_col, matrix_sizes, cpu_times_s, gpu_times_s, x_n, y_cpu, y_gpu


def _build_reflector_rows_for_kernel(
    cpu_idx: Dict[float, Dict[str, Any]],
    gpu_idx: Dict[float, Dict[str, Any]],
    kernel_key: str,
):
    keys = sorted(set(cpu_idx.keys()) | set(gpu_idx.keys()))

    first_col = [f"{k:g} GHz" for k in keys]
    matrix_sizes, cpu_times_s, gpu_times_s = [], [], []
    x_n, y_cpu, y_gpu = [], [], []

    for k in keys:
        c = cpu_idx.get(k, {})
        g = gpu_idx.get(k, {})

        matrix_sizes.append(c.get("matrix") or g.get("matrix") or "-")
        x_n.append(c.get("n") or g.get("n"))

        ct = c.get(kernel_key)
        gt = g.get(kernel_key)

        cpu_times_s.append(_fmt_time(ct))
        gpu_times_s.append(_fmt_time(gt))

        y_cpu.append(_to_float(ct))
        y_gpu.append(_to_float(gt))

    return first_col, matrix_sizes, cpu_times_s, gpu_times_s, x_n, y_cpu, y_gpu

def _build_patch_rows_for_lu_table(cpu_idx, gpu_idx):
    keys = sorted(set(cpu_idx.keys()) | set(gpu_idx.keys()), key=lambda k: (k[0], k[1]))

    first_col = [_fmt_patch_size(k[0], k[1]) for k in keys]
    matrix_sizes, cpu_times, gpu_times = [], [], []

    for k in keys:
        c = cpu_idx.get(k, {})
        g = gpu_idx.get(k, {})

        matrix_sizes.append(c.get("matrix") or g.get("matrix") or "-")
        cpu_times.append(_fmt_time(c.get("LU")))
        gpu_times.append(_fmt_time(g.get("LU")))

    return first_col, matrix_sizes, cpu_times, gpu_times


def _build_reflector_rows_for_lu_table(cpu_idx, gpu_idx):
    keys = sorted(set(cpu_idx.keys()) | set(gpu_idx.keys()))

    first_col = [f"{k:g} GHz" for k in keys]
    matrix_sizes, cpu_times, gpu_times = [], [], []

    for k in keys:
        c = cpu_idx.get(k, {})
        g = gpu_idx.get(k, {})

        matrix_sizes.append(c.get("matrix") or g.get("matrix") or "-")
        cpu_times.append(_fmt_time(c.get("LU")))
        gpu_times.append(_fmt_time(g.get("LU")))

    return first_col, matrix_sizes, cpu_times, gpu_times


def _build_patch_rows_for_lu(cpu_idx, gpu_idx):
    keys = sorted(set(cpu_idx.keys()) | set(gpu_idx.keys()), key=lambda k: (k[0], k[1]))
    x_n, y_cpu, y_gpu = [], [], []
    for k in keys:
        c = cpu_idx.get(k, {})
        g = gpu_idx.get(k, {})
        x_n.append(c.get("n") or g.get("n"))
        y_cpu.append(_to_float(c.get("LU")))
        y_gpu.append(_to_float(g.get("LU")))
    return x_n, y_cpu, y_gpu


def _build_reflector_rows_for_lu(cpu_idx, gpu_idx):
    keys = sorted(set(cpu_idx.keys()) | set(gpu_idx.keys()))
    x_n, y_cpu, y_gpu = [], [], []
    for k in keys:
        c = cpu_idx.get(k, {})
        g = gpu_idx.get(k, {})
        x_n.append(c.get("n") or g.get("n"))
        y_cpu.append(_to_float(c.get("LU")))
        y_gpu.append(_to_float(g.get("LU")))
    return x_n, y_cpu, y_gpu


# ----------------------------- main -----------------------------


if __name__ == "__main__":
    import sys
    from parse_results import parse_result_file

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} experiment_cpu.txt experiment_gpu.txt")
        raise SystemExit(1)

    cpu_file = sys.argv[1]
    gpu_file = sys.argv[2]

    cpu = parse_result_file(cpu_file)
    gpu = parse_result_file(gpu_file)

    cpu_patch = _index_patch_times(cpu)
    gpu_patch = _index_patch_times(gpu)
    cpu_refl = _index_reflector_times(cpu)
    gpu_refl = _index_reflector_times(gpu)

    kernels = [
        ("L", r"$L_{21}$ Kernel"),
        ("U", r"$U_{12}$ Kernel"),
        ("S", "Schur Update Kernel"),
        ("T", "Total Kernel Computation Time"),
    ]

    # ---- Patch tables + plots ----
    for key, title in kernels:
        first_col, matrix_sizes, cpu_times_s, gpu_times_s, x_n, y_cpu, y_gpu = _build_patch_rows_for_kernel(
            cpu_patch, gpu_patch, key
        )

        write_cpu_gpu_time_table(
            filename=f"patch_time_{key}_table.tex",
            caption=f"Computation runtimes (s) for patch-array test cases: {title}.",
            label=f"tab:patch-time-{key}",
            first_col_header="Size of patch array",
            first_col_values=first_col,
            matrix_sizes=matrix_sizes,
            cpu_times=cpu_times_s,
            gpu_times=gpu_times_s,
        )

        plot_cpu_gpu_times(
            outfile=f"patch_time_{key}_plot.png",
            title=f"Patch arrays: {title}",
            x_n=x_n,
            y_cpu=y_cpu,
            y_gpu=y_gpu,
            ylabel="Computation time (s)",
            dash_last_two=True,
            dash_all=False,
        )


    # ---- Reflector tables + plots ----
    for key, title in kernels:
        first_col, matrix_sizes, cpu_times_s, gpu_times_s, x_n, y_cpu, y_gpu = _build_reflector_rows_for_kernel(
            cpu_refl, gpu_refl, key
        )

        write_cpu_gpu_time_table(
            filename=f"reflector_time_{key}_table.tex",
            caption=f"Computation runtimes (s) for reflector-with-struts test cases: {title}.",
            label=f"tab:refl-time-{key}",
            first_col_header="GHz of reflector",
            first_col_values=first_col,
            matrix_sizes=matrix_sizes,
            cpu_times=cpu_times_s,
            gpu_times=gpu_times_s,
        )

        plot_cpu_gpu_times(
            outfile=f"reflector_time_{key}_plot.png",
            title=f"Reflectors with struts: {title}",
            x_n=x_n,
            y_cpu=y_cpu,
            y_gpu=y_gpu,
            ylabel="Computation time (s)",
            dash_last_two=False,
            dash_all=True,
        )


    # ---- LU total time plots (patch + reflector) ----
    x_n, y_cpu, y_gpu = _build_patch_rows_for_lu(cpu_patch, gpu_patch)
    plot_cpu_gpu_times(
        outfile="patch_lu_time_plot.png",
        title="Patch arrays: Total LU Computation Time",
        x_n=x_n,
        y_cpu=y_cpu,
        y_gpu=y_gpu,
        ylabel="Time (s)",
        dash_last_two=True,
        dash_all=False,
    )

    x_n, y_cpu, y_gpu = _build_reflector_rows_for_lu(cpu_refl, gpu_refl)
    plot_cpu_gpu_times(
        outfile="reflector_lu_time_plot.png",
        title="Reflectors with struts: Total LU Computation time",
        x_n=x_n,
        y_cpu=y_cpu,
        y_gpu=y_gpu,
        ylabel="Time (s)",
        dash_last_two=False,
        dash_all=True,
    )

    # ---- LU total time tables (patch + reflector) ----

    first_col, matrix_sizes, cpu_times, gpu_times = _build_patch_rows_for_lu_table(
        cpu_patch, gpu_patch
    )
    write_cpu_gpu_time_table(
        filename="patch_lu_time_table.tex",
        caption="Computation runtimes (s) for total LU factorisation on patch-array test cases.",
        label="tab:patch-lu-time",
        first_col_header="Size of patch array",
        first_col_values=first_col,
        matrix_sizes=matrix_sizes,
        cpu_times=cpu_times,
        gpu_times=gpu_times,
    )

    first_col, matrix_sizes, cpu_times, gpu_times = _build_reflector_rows_for_lu_table(
        cpu_refl, gpu_refl
    )
    write_cpu_gpu_time_table(
        filename="reflector_lu_time_table.tex",
        caption="Computation runtimes (s) for total LU factorisation on reflector-with-struts test cases.",
        label="tab:refl-lu-time",
        first_col_header="GHz of reflector",
        first_col_values=first_col,
        matrix_sizes=matrix_sizes,
        cpu_times=cpu_times,
        gpu_times=gpu_times,
    )


    print("Wrote patch tables:")
    print("  patch_time_L_table.tex  patch_time_U_table.tex  patch_time_S_table.tex  patch_time_T_table.tex")
    print("Wrote reflector tables:")
    print("  reflector_time_L_table.tex  reflector_time_U_table.tex  reflector_time_S_table.tex  reflector_time_T_table.tex")
    print("Wrote kernel plots (loglog):")
    print("  patch_time_L_plot.png  patch_time_U_plot.png  patch_time_S_plot.png  patch_time_T_plot.png")
    print("  reflector_time_L_plot.png  reflector_time_U_plot.png  reflector_time_S_plot.png  reflector_time_T_plot.png")
    print("Wrote LU total plots (loglog):")
    print("  patch_lu_time_plot.png  reflector_lu_time_plot.png")
    print("Wrote LU total tables:")
    print("  patch_lu_time_table.tex  reflector_lu_time_table.tex")

