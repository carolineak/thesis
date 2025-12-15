#!/usr/bin/env python3
"""
Parse a block-sparse .bin file laid out as:
[num_blocks, row_start, row_stop, col_start, col_stop, values]
with:
- num_blocks: 1 x int32
- row_start/row_stop/col_start/col_stop: num_blocks x int32
- values: concatenated complex64 (i.e., complex with float32 real/imag)

This script:
1) Parses metadata
2) Splits values per block (column-major inside each block, order='F')
3) Plots "one block size per block-row" (one bar per unique row interval)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BlockMeta:
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    @property
    def nrows(self) -> int:
        return int(self.row_stop - self.row_start)

    @property
    def ncols(self) -> int:
        return int(self.col_stop - self.col_start)

    @property
    def nnz(self) -> int:
        return self.nrows * self.ncols


def read_block_sparse_bin(path: Path) -> Tuple[List[BlockMeta], List[np.ndarray]]:
    """
    Returns:
      metas: list of BlockMeta length num_blocks
      blocks: list of dense numpy arrays with shape (nrows, ncols), complex64
    """
    with path.open("rb") as f:
        num_blocks_arr = np.fromfile(f, dtype=np.int32, count=1)
        if num_blocks_arr.size != 1:
            raise ValueError("File ended before reading num_blocks.")
        num_blocks = int(num_blocks_arr[0])
        if num_blocks < 0:
            raise ValueError(f"Invalid num_blocks={num_blocks}.")

        row_start = np.fromfile(f, dtype=np.int32, count=num_blocks)
        row_stop  = np.fromfile(f, dtype=np.int32, count=num_blocks)
        col_start = np.fromfile(f, dtype=np.int32, count=num_blocks)
        col_stop  = np.fromfile(f, dtype=np.int32, count=num_blocks)

        if any(arr.size != num_blocks for arr in (row_start, row_stop, col_start, col_stop)):
            raise ValueError("File ended while reading block index arrays.")

        metas: List[BlockMeta] = []
        nnz_per_block = np.empty(num_blocks, dtype=np.int64)

        for i in range(num_blocks):
            rs, re = int(row_start[i]), int(row_stop[i])
            cs, ce = int(col_start[i]), int(col_stop[i])
            if re < rs or ce < cs:
                raise ValueError(
                    f"Invalid range for block {i}: "
                    f"rows [{rs},{re}), cols [{cs},{ce})"
                )
            meta = BlockMeta(rs, re, cs, ce)
            metas.append(meta)
            nnz_per_block[i] = meta.nnz

        total_vals = int(nnz_per_block.sum())
        values = np.fromfile(f, dtype=np.complex64, count=total_vals)

        if values.size != total_vals:
            raise ValueError(
                f"Expected {total_vals} complex64 values but only read {values.size}. "
                f"Check dtype (complex64) and file integrity."
            )

        # Optional: detect trailing bytes
        trailing = f.read(1)
        if trailing != b"":
            print("Warning: file has trailing bytes after expected values. "
                  "The parser ignored trailing data.")

    # Split values into per-block arrays (column-major within each block)
    blocks: List[np.ndarray] = []
    offset = 0
    for meta in metas:
        count = meta.nnz
        flat = values[offset:offset + count]
        blk = flat.reshape((meta.nrows, meta.ncols), order="F")
        blocks.append(blk)
        offset += count

    return metas, blocks


def plot_block_row_sizes(metas: List[BlockMeta], save_path: Path | None, show: bool, also_distribution: bool) -> None:
    """
    "One block size per (block) row": group blocks by their (row_start,row_stop).
    Size is the height of that block row (row_stop-row_start).
    """
    # Group by unique row intervals
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, m in enumerate(metas):
        key = (m.row_start, m.row_stop)
        groups.setdefault(key, []).append(i)

    # Sort by row_start for a stable "row index" on the x-axis
    keys_sorted = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    sizes = np.array([k[1] - k[0] for k in keys_sorted], dtype=int)

    # Per-row bar chart (one bar per block-row)
    plt.figure()
    x = np.arange(sizes.size)
    plt.bar(x, sizes)
    plt.xlabel("Block-row index (sorted by row_start)")
    plt.ylabel("Block-row size (rows)")
    plt.title("Block-row sizes (one size per unique row interval)")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

    # Optional distribution histogram
    if also_distribution:
        plt.figure()
        plt.hist(sizes, bins="auto")
        plt.xlabel("Block-row size (rows)")
        plt.ylabel("Count")
        plt.title("Distribution of block-row sizes")
        plt.tight_layout()

        if save_path is not None:
            dist_path = save_path.with_name(save_path.stem + "_distribution" + save_path.suffix)
            plt.savefig(dist_path, dpi=200)
            print(f"Saved plot: {dist_path}")
        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse block-sparse .bin and plot block-row size histogram.")
    ap.add_argument("binfile", type=Path, help="Path to .bin file")
    ap.add_argument("--save", type=Path, default=None, help="Save plot to this file (e.g., sizes.png)")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window with the plot")
    ap.add_argument("--also-distribution", action="store_true", help="Also plot a size distribution histogram")
    ap.add_argument("--summary", action="store_true", help="Print parsed summary stats")
    args = ap.parse_args()

    metas, _blocks = read_block_sparse_bin(args.binfile)

    if args.summary:
        num_blocks = len(metas)
        max_row = max(m.row_stop for m in metas) if metas else 0
        max_col = max(m.col_stop for m in metas) if metas else 0
        nnz = sum(m.nnz for m in metas)

        unique_block_rows = len({(m.row_start, m.row_stop) for m in metas})
        unique_block_cols = len({(m.col_start, m.col_stop) for m in metas})

        print(f"Blocks: {num_blocks}")
        print(f"Inferred matrix shape (upper bounds): {max_row} x {max_col}")
        print(f"Total dense values stored (sum block areas): {nnz}")
        print(f"Unique block-rows (unique row intervals): {unique_block_rows}")
        print(f"Unique block-cols (unique col intervals): {unique_block_cols}")

    plot_block_row_sizes(
        metas,
        save_path=args.save,
        show=(not args.no_show),
        also_distribution=args.also_distribution,
    )


if __name__ == "__main__":
    main()
