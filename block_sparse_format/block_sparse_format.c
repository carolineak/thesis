#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "block_sparse_format.h"

// ===== Helper functions =====
// Make a range [start,end]
static inline int_range make_range(int start, int end) {
    int_range r;
    r.start = start;
    r.end   = end;
    return r;
}

// Length of a range
static inline int range_length(int_range r) {
    return (r.end >= r.start) ? (r.end - r.start + 1) : 0;
}

// ---------------------------------------------------------------------------
// create
//
// Arguments
//   bsf        : block_sparse_format (output)
//   rows       : array of row indices
//   cols       : array of col indices
//   values     : array of matrix_block
//   num_blocks : number of blocks
//
// Returns 0 on success, <0 on allocation failure
// ---------------------------------------------------------------------------
int create(block_sparse_format *bsf,
           const int *rows,
           const int *cols,
           const matrix_block *values,
           int num_blocks)
{
    int i;
    int num_rows = 0;
    int num_cols = 0;
    int offset;

    // Find max row/col index
    for (i = 0; i < num_blocks; i++) {
        if (rows[i] + 1 > num_rows) num_rows = rows[i] + 1;
        if (cols[i] + 1 > num_cols) num_cols = cols[i] + 1;
    }

    bsf->num_rows   = num_rows;
    bsf->num_cols   = num_cols;
    bsf->num_blocks = num_blocks;

    // Size query
    // ===================================================================
    // Count how many blocks each row and column contains
    bsf->rows = (block_slice*)calloc(num_rows, sizeof(block_slice));
    bsf->cols = (block_slice*)calloc(num_cols, sizeof(block_slice));
    if (!bsf->rows || !bsf->cols) return -1;

    for (i = 0; i < num_blocks; i++) {
        bsf->rows[rows[i]].num_blocks++;
        bsf->cols[cols[i]].num_blocks++;
    }

    // Allocate space for block maps
    for (i = 0; i < num_rows; i++) {
        if (bsf->rows[i].num_blocks > 0) {
            bsf->rows[i].indices = (int*)malloc(bsf->rows[i].num_blocks * sizeof(int));
            bsf->rows[i].num_blocks = 0;
        }
    }
    for (i = 0; i < num_cols; i++) {
        if (bsf->cols[i].num_blocks > 0) {
            bsf->cols[i].indices = (int*)malloc(bsf->cols[i].num_blocks * sizeof(int));
            bsf->cols[i].num_blocks = 0;
        }
    }

    // Save original indices of the blocks
    // ===================================================================
    bsf->row_indices = (int*)malloc(num_blocks * sizeof(int));
    bsf->col_indices = (int*)malloc(num_blocks * sizeof(int));
    if (!bsf->row_indices || !bsf->col_indices) return -2;

    memcpy(bsf->row_indices, rows, num_blocks * sizeof(int));
    memcpy(bsf->col_indices, cols, num_blocks * sizeof(int));

    // Fill arrays
    // ===================================================================
    for (i = 0; i < num_blocks; i++) {
        int rpos = bsf->rows[rows[i]].num_blocks++;
        int cpos = bsf->cols[cols[i]].num_blocks++;
        bsf->rows[rows[i]].indices[rpos] = i;
        bsf->cols[cols[i]].indices[cpos] = i;
    }

    // Copy blocks
    // ===================================================================
    bsf->blocks = (matrix_block*)calloc(num_blocks, sizeof(matrix_block));
    if (!bsf->blocks) return -3;

    for (i = 0; i < num_blocks; i++) {
        matrix_block_init(&bsf->blocks[i], values[i].rows, values[i].cols);
        memcpy(bsf->blocks[i].data, values[i].data,
               values[i].rows * values[i].cols * sizeof(float));
    }

    // Compute size of matrix (m,n)
    // ===================================================================
    // Get info of which indices each block row/column corresponds to
    for (i = 0; i < num_blocks; i++) {
        bsf->rows[rows[i]].range = make_range(0, bsf->blocks[i].rows - 1);
        bsf->cols[cols[i]].range = make_range(0, bsf->blocks[i].cols - 1);
    }

    // Accumulate row offsets
    offset = 0;
    for (i = 0; i < num_rows; i++) {
        bsf->rows[i].range.start += offset;
        bsf->rows[i].range.end   += offset;
        offset = bsf->rows[i].range.end + 1;
    }

    // Accumulate col offsets
    offset = 0;
    for (i = 0; i < num_cols; i++) {
        bsf->cols[i].range.start += offset;
        bsf->cols[i].range.end   += offset;
        offset = bsf->cols[i].range.end + 1;
    }

    // Set size of matrix
    // ===================================================================
    bsf->m = 0;
    bsf->n = 0;
    for (i = 0; i < num_rows; i++) {
        bsf->m += range_length(bsf->rows[i].range);
    }
    for (i = 0; i < num_cols; i++) {
        bsf->n += range_length(bsf->cols[i].range);
    }

    return 0;
}
