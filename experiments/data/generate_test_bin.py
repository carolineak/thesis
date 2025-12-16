#!/usr/bin/env python3
import numpy as np

def generate_block_sparse_bin(out_file="matrix.bin",
                              block_rows=2, block_cols=2,
                              seed=1234):

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Your provided block coordinates (ALREADY REPLACED & FINAL)
    # ------------------------------------------------------------------

    row_start = np.array([
        1, 3, 9, 11, 1, 3, 5, 9, 11, 13, 3, 5, 7, 11, 13, 15,
        5, 7, 13, 15, 1, 3, 9, 11, 13, 1, 3, 5, 9, 11, 13, 15,
        3, 5, 7, 9, 11, 13, 15, 5, 7, 11, 13, 15
    ], dtype=np.int32)

    row_end = np.array([
        2, 4, 10, 12, 2, 4, 6, 10, 12, 14, 4, 6, 8, 12, 14, 16,
        6, 8, 14, 16, 2, 4, 10, 12, 14, 2, 4, 6, 10, 12, 14, 16,
        4, 6, 8, 10, 12, 14, 16, 6, 8, 12, 14, 16
    ], dtype=np.int32)

    col_start = np.array([
        1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5,
        7, 7, 7, 7, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11, 11, 11,
        13, 13, 13, 13, 13, 13, 13, 15, 15, 15, 15, 15
    ], dtype=np.int32)

    col_end = np.array([
        2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6,
        8, 8, 8, 8, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12,
        14, 14, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16
    ], dtype=np.int32)

    # ------------------------------------------------------------------

    num_blocks = len(row_start)
    assert num_blocks == 44, f"Expected 44 blocks, got {num_blocks}"

    num_blocks_arr = np.array([num_blocks], dtype=np.int32)

    # Generate random 2×2 block values
    values_per_block = block_rows * block_cols
    total_values = num_blocks * values_per_block

    real_part = rng.random(total_values, dtype=np.float32)
    imag_part = np.zeros_like(real_part, dtype=np.float32) 
    values = (real_part + 1j * imag_part).astype(np.complex64)

    # ------------------------------------------------------------------
    # Write binary file
    # ------------------------------------------------------------------
    with open(out_file, "wb") as f:
        num_blocks_arr.tofile(f)
        row_start.tofile(f)
        row_end.tofile(f)
        col_start.tofile(f)
        col_end.tofile(f)
        values.tofile(f)

    print(f"Wrote '{out_file}' successfully.")
    print(f"num_blocks: {num_blocks}")
    print(f"total complex values: {total_values}")


if __name__ == "__main__":
    generate_block_sparse_bin("sparse_data_example.bin")
