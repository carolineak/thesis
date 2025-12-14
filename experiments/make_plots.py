#!/usr/bin/env python3
import re
from typing import Dict, Any, Optional, Tuple, List

import matplotlib.pyplot as plt


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


def _collect_patch_points(results: Dict[str, Any]) -> List[dict]:
    pts = []
    for i, name in enumerate(results["name"]):
        if not isinstance(name, str):
            continue
        if _extract_patch_size(name) is None:
            continue
        n = results["matrix_n"][i]
        if n is None:
            continue
        pts.append(
            dict(
                y=int(n),
                L_ops=results["L_ops"][i],
                U_ops=results["U_ops"][i],
                S_ops=results["Schur_ops"][i],
                T_ops=results["Total_ops"][i],
                L_time=results["L_time"][i],
                U_time=results["U_time"][i],
                S_time=results["Schur_time"][i],
                T_time=results["Total_time"][i],
            )
        )
    pts.sort(key=lambda d: d["y"])
    return pts


def _collect_reflector_points(results: Dict[str, Any]) -> List[dict]:
    pts = []
    for i, name in enumerate(results["name"]):
        if not isinstance(name, str):
            continue
        if _extract_reflector_freq(name) is None:
            continue
        n = results["matrix_n"][i]
        if n is None:
            continue
        pts.append(
            dict(
                y=int(n),
                L_ops=results["L_ops"][i],
                U_ops=results["U_ops"][i],
                S_ops=results["Schur_ops"][i],
                T_ops=results["Total_ops"][i],
                L_time=results["L_time"][i],
                U_time=results["U_time"][i],
                S_time=results["Schur_time"][i],
                T_time=results["Total_time"][i],
            )
        )
    pts.sort(key=lambda d: d["y"])
    return pts


def _to_float_list(xs):
    out = []
    for x in xs:
        if x is None:
            out.append(float("nan"))
        else:
            out.append(float(x))
    return out


def plot_ops_vs_size(points: List[dict], title: str, outfile: str) -> None:
    x = [p["y"] for p in points]  # matrix size (n)
    yL = _to_float_list([p["L_ops"] for p in points])
    yU = _to_float_list([p["U_ops"] for p in points])
    yS = _to_float_list([p["S_ops"] for p in points])
    yT = _to_float_list([p["T_ops"] for p in points])

    plt.figure()
    plt.plot(x, yL, marker="o", label=r"Computing $L_{21}$")
    plt.plot(x, yU, marker="o", label=r"Computing $U_{12}$")
    plt.plot(x, yS, marker="o", label="Schur updating")
    plt.plot(x, yT, marker="o", label="Total")

    plt.title(title)
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Block matrix operation count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_time_vs_size(points: List[dict], title: str, outfile: str) -> None:
    x = [p["y"] for p in points]  # matrix size (n)
    yL = _to_float_list([p["L_time"] for p in points])
    yU = _to_float_list([p["U_time"] for p in points])
    yS = _to_float_list([p["S_time"] for p in points])
    yT = _to_float_list([p["T_time"] for p in points])

    plt.figure()
    plt.plot(x, yL, marker="o", label=r"Computing $L_{21}$")
    plt.plot(x, yU, marker="o", label=r"Computing $U_{12}$")
    plt.plot(x, yS, marker="o", label="Schur updating")
    plt.plot(x, yT, marker="o", label="Total")

    plt.title(title)
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Computation time (s)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


if __name__ == "__main__":
    import sys
    from parse_results import parse_result_file

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <result_file.txt>")
        raise SystemExit(1)

    results = parse_result_file(sys.argv[1])

    patch_pts = _collect_patch_points(results)
    refl_pts = _collect_reflector_points(results)

    plot_ops_vs_size(
        patch_pts,
        title="Patch arrays: operations vs matrix size",
        outfile="patch_ops_vs_size.png",
    )
    plot_time_vs_size(
        patch_pts,
        title="Patch arrays: time vs matrix size",
        outfile="patch_time_vs_size.png",
    )

    plot_ops_vs_size(
        refl_pts,
        title="Reflectors with struts: operations vs matrix size",
        outfile="reflector_ops_vs_size.png",
    )
    plot_time_vs_size(
        refl_pts,
        title="Reflectors with struts: time vs matrix size",
        outfile="reflector_time_vs_size.png",
    )

    print("Wrote:")
    print("  patch_ops_vs_size.png")
    print("  patch_time_vs_size.png")
    print("  reflector_ops_vs_size.png")
    print("  reflector_time_vs_size.png")
