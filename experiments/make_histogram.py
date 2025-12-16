#!/usr/bin/env python3
"""
Python 3.6-compatible parser for block-sparse .bin files laid out as:
[num_blocks, row_start, row_stop, col_start, col_stop, values]
where:
- num_blocks: 1 x int32
- row_start/row_stop/col_start/col_stop: num_blocks x int32
- values: concatenated complex64 (complex float32)

Plots one block-row size per unique (row_start,row_stop) interval.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


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


def plot_block_row_sizes(metas, save_path, show, also_distribution):
    # type: (List[BlockMeta], Optional[Path], bool, bool) -> None
    groups = {}  # type: Dict[Tuple[int, int], List[int]]
    for i, m in enumerate(metas):
        key = (m.row_start, m.row_stop)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    keys_sorted = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    sizes = np.array([k[1] - k[0] for k in keys_sorted], dtype=int)

    plt.figure()
    x = np.arange(sizes.size)
    plt.bar(x, sizes)
    plt.xlabel("Block-row index (sorted by row_start)")
    plt.ylabel("Block-row size (rows)")
    plt.title("Block-row sizes (one size per unique row interval)")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(str(save_path), dpi=200)
        print("Saved plot: {}".format(save_path))
    if show:
        plt.show()
    else:
        plt.close()

    if also_distribution:
        plt.figure()
        plt.hist(sizes, bins="auto")
        plt.xlabel("Block-row size (rows)")
        plt.ylabel("Count")
        plt.title("Distribution of block-row sizes")
        plt.tight_layout()

        if save_path is not None:
            dist_path = save_path.with_name(save_path.stem + "_distribution" + save_path.suffix)
            plt.savefig(str(dist_path), dpi=200)
            print("Saved plot: {}".format(dist_path))
        if show:
            plt.show()
        else:
            plt.close()


def main():
    ap = argparse.ArgumentParser(description="Parse block-sparse .bin and plot block-row sizes.")
    ap.add_argument("binfile", type=Path, help="Path to .bin file")
    ap.add_argument("--save", type=Path, default=None, help="Save plot to this file (e.g., sizes.png)")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window with the plot")
    ap.add_argument("--also-distribution", action="store_true", help="Also plot a size distribution histogram")
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

    plot_block_row_sizes(
        metas,
        save_path=args.save,
        show=(not args.no_show),
        also_distribution=args.also_distribution,
    )


if __name__ == "__main__":
    main()
