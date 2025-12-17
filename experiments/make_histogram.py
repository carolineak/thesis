#!/usr/bin/env python3
"""
Python 3.6-compatible parser for block-sparse .bin files laid out as:
[num_blocks, row_start, row_stop, col_start, col_stop, values]
where:
- num_blocks: 1 x int32
- row_start/row_stop/col_start/col_stop: num_blocks x int32
- values: concatenated complex64 (complex float32)

Plots ONLY the distribution histogram of block-row sizes
(one size per unique (row_start,row_stop) interval).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import math


class BlockMeta(object):
    __slots__ = ("row_start", "row_stop", "col_start", "col_stop")

    def __init__(self, row_start, row_stop, col_start, col_stop):
        self.row_start = int(row_start)
        self.row_stop = int(row_stop)
        self.col_start = int(col_start)
        self.col_stop = int(col_stop)

    @property
    def nrows(self):
        return int(self.row_stop - self.row_start)

    @property
    def ncols(self):
        return int(self.col_stop - self.col_start)

    @property
    def nnz(self):
        return self.nrows * self.ncols


def read_block_sparse_bin(path):  # type: (Path) -> Tuple[List[BlockMeta], List[np.ndarray]]
    with path.open("rb") as f:
        num_blocks_arr = np.fromfile(f, dtype=np.int32, count=1)
        if num_blocks_arr.size != 1:
            raise ValueError("File ended before reading num_blocks.")
        num_blocks = int(num_blocks_arr[0])
        if num_blocks < 0:
            raise ValueError("Invalid num_blocks={}".format(num_blocks))

        row_start = np.fromfile(f, dtype=np.int32, count=num_blocks)
        row_stop = np.fromfile(f, dtype=np.int32, count=num_blocks)
        col_start = np.fromfile(f, dtype=np.int32, count=num_blocks)
        col_stop = np.fromfile(f, dtype=np.int32, count=num_blocks)

        if (row_start.size != num_blocks or row_stop.size != num_blocks or
                col_start.size != num_blocks or col_stop.size != num_blocks):
            raise ValueError("File ended while reading block index arrays.")

        metas = []  # type: List[BlockMeta]
        nnz_per_block = np.empty(num_blocks, dtype=np.int64)

        for i in range(num_blocks):
            rs, re = int(row_start[i]), int(row_stop[i])
            cs, ce = int(col_start[i]), int(col_stop[i])
            if re < rs or ce < cs:
                raise ValueError(
                    "Invalid range for block {}: rows [{},{}), cols [{},{}).".format(i, rs, re, cs, ce)
                )
            meta = BlockMeta(rs, re, cs, ce)
            metas.append(meta)
            nnz_per_block[i] = meta.nnz

        total_vals = int(nnz_per_block.sum())
        values = np.fromfile(f, dtype=np.complex64, count=total_vals)
        if values.size != total_vals:
            raise ValueError(
                "Expected {} complex64 values but only read {}. "
                "Check dtype (complex64) and file integrity.".format(total_vals, values.size)
            )

        trailing = f.read(1)
        if trailing != b"":
            print("Warning: file has trailing bytes after expected values; ignored.")

    # Split values into per-block arrays (column-major inside each block)
    blocks = []  # type: List[np.ndarray]
    offset = 0
    for meta in metas:
        count = meta.nnz
        flat = values[offset:offset + count]
        blk = flat.reshape((meta.nrows, meta.ncols), order="F")
        blocks.append(blk)
        offset += count

    return metas, blocks


def plot_block_row_size_distribution(metas, save_path, show):
    # type: (List[BlockMeta], Optional[Path], bool) -> None
    groups = {}  # type: Dict[Tuple[int, int], List[int]]
    for i, m in enumerate(metas):
        key = (m.row_start, m.row_stop)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    keys_sorted = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    sizes = np.array([k[1] - k[0] for k in keys_sorted], dtype=int)

    if sizes.size == 0:
        raise ValueError("No block-row sizes found. Check that the file contains blocks.")

    s_min = int(sizes.min())
    s_max = int(sizes.max())
    s_range = max(1, s_max - s_min)

    # -----------------------------
    # Choose a sensible bin width, but force it to be a multiple of 10
    # -----------------------------
    q25, q75 = np.percentile(sizes, [25, 75])
    iqr = float(q75 - q25)
    n = float(sizes.size)

    if iqr > 0.0:
        fd_width = 2.0 * iqr / (n ** (1.0 / 3.0))
        raw_width = max(1.0, fd_width)
    else:
        raw_width = 1.0

    # Round UP to next multiple of 10, minimum 10
    def round_up_to_10(x):
        return int(math.ceil(x / 10.0) * 10)

    bin_width = round_up_to_10(raw_width)
    if bin_width < 10:
        bin_width = 10

    # Cap number of bins, keeping multiple-of-10 width
    max_bins = 40
    needed_bins = int(math.ceil((s_range + 1) / float(bin_width)))
    if needed_bins > max_bins:
        bin_width = round_up_to_10((s_range + 1) / float(max_bins))
        if bin_width < 10:
            bin_width = 10

    # Align bins to multiples of 10 (edges on ... -0.5 to capture integer sizes)
    start_edge = (s_min // 10) * 10 - 0.5
    end_edge = int(math.ceil(s_max / 10.0) * 10) + 0.5 + bin_width
    edges = np.arange(start_edge, end_edge, bin_width)

    # -----------------------------
    # Plot histogram
    # -----------------------------
    plt.figure()
    plt.hist(sizes, bins=edges)

    plt.xlabel("Block-row size (rows)")
    plt.ylabel("Count")
    plt.title("Distribution of block-row sizes of 15x15 patch array")

    # Count axis: integers only
    # Y axis: ~8 ticks (let matplotlib choose "nice" real-valued ticks)
    ax = plt.gca()
    ax.locator_params(axis="y", nbins=8)

    # -----------------------------
    # X ticks: multiples of 10, but not every 10 (aim ~8 ticks)
    # -----------------------------
    base = 10
    tick_step = int(math.ceil((s_range / 8.0) / base) * base)
    if tick_step < 10:
        tick_step = 10

    start_tick = (s_min // base) * base
    end_tick = int(math.ceil(s_max / float(base)) * base)

    xticks = list(np.arange(start_tick, end_tick + tick_step, tick_step))

    plt.xticks(xticks)

    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=200)
        print("Saved plot: {}".format(save_path))

    if show:
        plt.show()
    else:
        plt.close()




def main():
    ap = argparse.ArgumentParser(description="Parse block-sparse .bin and plot block-row size distribution.")
    ap.add_argument("binfile", type=Path, help="Path to .bin file")
    ap.add_argument("--save", type=Path, default=None, help="Save plot to this file (e.g., dist.png)")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window with the plot")
    ap.add_argument("--summary", action="store_true", help="Print parsed summary stats")
    args = ap.parse_args()

    metas, _blocks = read_block_sparse_bin(args.binfile)

    if args.summary:
        num_blocks = len(metas)
        max_row = max([m.row_stop for m in metas]) if metas else 0
        max_col = max([m.col_stop for m in metas]) if metas else 0
        nnz = sum([m.nnz for m in metas])

        unique_block_rows = len(set([(m.row_start, m.row_stop) for m in metas]))
        unique_block_cols = len(set([(m.col_start, m.col_stop) for m in metas]))

        print("Blocks: {}".format(num_blocks))
        print("Inferred matrix shape (upper bounds): {} x {}".format(max_row, max_col))
        print("Total dense values stored (sum block areas): {}".format(nnz))
        print("Unique block-rows (unique row intervals): {}".format(unique_block_rows))
        print("Unique block-cols (unique col intervals): {}".format(unique_block_cols))

    plot_block_row_size_distribution(
        metas,
        save_path=args.save,
        show=(not args.no_show),
    )


if __name__ == "__main__":
    main()
