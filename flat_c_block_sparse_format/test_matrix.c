#ifndef TEST_MATRIX_H
#define TEST_MATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include "block_sparse_format.h"

// ===========================================================================
// Random float complex in [0,1] + i[0,1]
// ===========================================================================
static inline float complex crand(void) {
    return ((float)rand() / RAND_MAX) + ((float)rand() / RAND_MAX) * I;
}

// ===========================================================================
// Copy one b×b block from a dense matrix into a flat data array
// Each block is stored in COLUMN-MAJOR order
// Arguments:
//   dense : source dense matrix of size n×n
//   n     : leading dimension of dense matrix
//   b     : block size
//   r0,c0 : top-left corner of block in 'dense'
//   k     : index of the block in 'data' (0-based)
//   data  : destination flat buffer with space for NUM_BLOCKS*b*b elements
// ===========================================================================
static void pack_block(const float complex *dense,
                       int n, int b,
                       int r0, int c0,
                       int k,
                       float complex *data)
{
    float complex *dst = data + (size_t)k * (size_t)b * (size_t)b;
    for (int j = 0; j < b; ++j) {
        for (int i = 0; i < b; ++i) {
            dst[(size_t)j*(size_t)b + (size_t)i] =
                dense[(size_t)(r0 + i)*(size_t)n + (size_t)(c0 + j)];
        }
    }
}

// Copy one rlen×clen block (rectangular) from 'dense' into the k-th block slot of 'data'
// The destination slot starts at 'dst_offset' and is COLUMN-MAJOR within the block.
static void pack_block_rect(const float complex *dense,
                            int n,
                            int r0, int c0,
                            int rlen, int clen,
                            size_t dst_offset,
                            float complex *data)
{
    float complex *dst = data + dst_offset;
    for (int j = 0; j < clen; ++j) {
        for (int i = 0; i < rlen; ++i) {
            dst[(size_t)j*(size_t)rlen + (size_t)i] =
                dense[(size_t)(r0 + i)*(size_t)n + (size_t)(c0 + j)];
        }
    }
}


// ===========================================================================
// Creates a test matrix with a given block structure
//
// 0: structure that creates no fill-ins
// 1 - 5 -
// - 3 6 -
// 2 4 7 -
// - - - 8
// ===========================================================================
int create_test_matrix(int n, int b, int block_structure,
                       float complex *dense, block_sparse_format *bsf)
{
    #define NUM_BLOCKS 8

    if (!dense) {
        fprintf(stderr, "No dense matrix allocated\n");
        return 1;
    }

    // Fill dense with random complex numbers
    for (int i = 0; i < n*n; i++) dense[i] = crand();

    if (block_structure == 0) {
        // Zero-out explicitly empty blocks
        const int zero_blocks[][2] = {
            {0,1},{0,3},{1,0},{1,3},{2,3},{3,0},{3,1},{3,2}
        };
        for (size_t k = 0; k < sizeof(zero_blocks)/sizeof(zero_blocks[0]); ++k) {
            const int r0 = zero_blocks[k][0] * b;
            const int c0 = zero_blocks[k][1] * b;
            for (int i = 0; i < b; ++i)
                for (int j = 0; j < b; ++j)
                    dense[(size_t)(r0 + i) * (size_t)n + (size_t)(c0 + j)] = 0.0f + 0.0f * I;
        }

        // Non-zero blocks (same pattern as before)
        const int rows[NUM_BLOCKS] = {0, 2, 1, 2, 0, 1, 2, 3};
        const int cols[NUM_BLOCKS] = {0, 0, 1, 1, 2, 2, 2, 3};

        // Verify divisibility
        const int num_block_rows = n / b;
        if (n % b != 0) {
            fprintf(stderr, "n must be divisible by b\n");
            return 1;
        }

        int *block_lengths = (int*)malloc((size_t)num_block_rows * sizeof *block_lengths);
        if (!block_lengths) {
            fprintf(stderr, "Allocation failed for block_lengths\n");
            return 1;
        }
        for (int r = 0; r < num_block_rows; ++r) block_lengths[r] = b;

        // Allocate flat buffer for all non-zero blocks
        float complex *data = (float complex*)
            malloc((size_t)NUM_BLOCKS * (size_t)b * (size_t)b * sizeof *data);
        if (!data) {
            fprintf(stderr, "Allocation failed for data buffer\n");
            free(block_lengths);
            return 1;
        }

        // Pack each block into flat data buffer
        pack_block(dense, n, b, 0*b, 0*b, 0, data); // 1
        pack_block(dense, n, b, 2*b, 0*b, 1, data); // 2
        pack_block(dense, n, b, 1*b, 1*b, 2, data); // 3
        pack_block(dense, n, b, 2*b, 1*b, 3, data); // 4
        pack_block(dense, n, b, 0*b, 2*b, 4, data); // 5
        pack_block(dense, n, b, 1*b, 2*b, 5, data); // 6
        pack_block(dense, n, b, 2*b, 2*b, 6, data); // 7
        pack_block(dense, n, b, 3*b, 3*b, 7, data); // 8

        // Create the block sparse matrix
        int status = create(bsf,
                            rows,
                            cols,
                            NUM_BLOCKS,
                            block_lengths,
                            data);
        if (status != 0) {
            fprintf(stderr, "Create() failed: %d\n", status);
            free(data);
            free(block_lengths);
            return 1;
        }

        // If create() deep-copies data, free temporary buffers
        free(data);
        free(block_lengths);
        return 0;
    } else if (block_structure == 1) {
        // Zero blocks:
        // (0,1), (0,2), (0,3), (1,0), (2,0), (2,3), (3,0), (3,2)
        const int zero_blocks[][2] = {
            {0,1},{0,2},{0,3},{1,0},{2,0},{2,3},{3,0},{3,2}
        };
        for (size_t k = 0; k < sizeof(zero_blocks)/sizeof(zero_blocks[0]); ++k) {
            const int r0 = zero_blocks[k][0] * b;
            const int c0 = zero_blocks[k][1] * b;
            for (int i = 0; i < b; ++i)
                for (int j = 0; j < b; ++j)
                    dense[(size_t)(r0 + i)*(size_t)n + (size_t)(c0 + j)] = 0.0f + 0.0f * I;
        }

        // Non-zero blocks (order):
        // 1:(0,0)  2:(1,1)  3:(2,1)  4:(3,1)  5:(1,2)  6:(2,2)  7:(1,3)  8:(3,3)
        const int rows[NUM_BLOCKS] = {0, 1, 2, 3, 1, 2, 1, 3};
        const int cols[NUM_BLOCKS] = {0, 1, 1, 1, 2, 2, 3, 3};

        // block_lengths: 4 block rows/cols, each of length b
        if (n % b != 0) { fprintf(stderr, "n must be divisible by b\n"); return 1; }
        const int num_block_rows = n / b;

        int *block_lengths = (int*)malloc((size_t)num_block_rows * sizeof *block_lengths);
        if (!block_lengths) { fprintf(stderr, "Allocation failed for block_lengths\n"); return 1; }
        for (int r = 0; r < num_block_rows; ++r) block_lengths[r] = b;

        // Flat buffer: 8 blocks * b*b each
        float complex *data = (float complex*)
            malloc((size_t)NUM_BLOCKS * (size_t)b * (size_t)b * sizeof *data);
        if (!data) { fprintf(stderr, "Allocation failed for data buffer\n"); free(block_lengths); return 1; }

        // Pack blocks (column-major within each block)
        pack_block(dense, n, b, 0*b, 0*b, 0, data); // 1
        pack_block(dense, n, b, 1*b, 1*b, 1, data); // 2
        pack_block(dense, n, b, 2*b, 1*b, 2, data); // 3
        pack_block(dense, n, b, 3*b, 1*b, 3, data); // 4
        pack_block(dense, n, b, 1*b, 2*b, 4, data); // 5
        pack_block(dense, n, b, 2*b, 2*b, 5, data); // 6
        pack_block(dense, n, b, 1*b, 3*b, 6, data); // 7
        pack_block(dense, n, b, 3*b, 3*b, 7, data); // 8

        int status = create(bsf, rows, cols, NUM_BLOCKS, block_lengths, data);
        if (status != 0) {
            fprintf(stderr, "Create() failed: %d\n", status);
            free(data); free(block_lengths);
            return 1;
        }

        free(data);
        free(block_lengths);
        return 0;
    } else if (block_structure == 2) {
        const int bsizes[4] = { b, b-1, b, b+1 };

        // Validate n matches total of block sizes
        int total = 0; for (int k = 0; k < 4; ++k) total += bsizes[k];
        if (total != n) { fprintf(stderr, "n (%d) != sum(bsizes) (%d)\n", n, total); return 1; }

        // Prefix sums for block offsets
        int off[4]; off[0] = 0; for (int k = 1; k < 4; ++k) off[k] = off[k-1] + bsizes[k-1];

        // Zero blocks (same pattern as structure 1)
        const int zero_blocks[][2] = {
            {0,1},{0,2},{0,3},{1,0},{2,0},{2,3},{3,0},{3,2}
        };
        for (size_t z = 0; z < sizeof(zero_blocks)/sizeof(zero_blocks[0]); ++z) {
            int bi = zero_blocks[z][0], bj = zero_blocks[z][1];
            int r0 = off[bi], c0 = off[bj];
            int rlen = bsizes[bi], clen = bsizes[bj];
            for (int j = 0; j < clen; ++j)
                for (int i = 0; i < rlen; ++i)
                    dense[(size_t)(r0 + i)*(size_t)n + (size_t)(c0 + j)] = 0.0f + 0.0f * I;
        }

        // Non-zero blocks (same order as structure 1)
        // 1:(0,0) 2:(1,1) 3:(2,1) 4:(3,1) 5:(1,2) 6:(2,2) 7:(1,3) 8:(3,3)
        const int rows[NUM_BLOCKS] = {0, 1, 2, 3, 1, 2, 1, 3};
        const int cols[NUM_BLOCKS] = {0, 1, 1, 1, 2, 2, 3, 3};

        // block_lengths: per block row/col
        int *block_lengths = (int*)malloc(4 * sizeof *block_lengths);
        if (!block_lengths) { fprintf(stderr, "Allocation failed for block_lengths\n"); return 1; }
        for (int r = 0; r < 4; ++r) block_lengths[r] = bsizes[r];

        // Compute total required entries for 'data'
        size_t total_entries = 0;
        for (int k = 0; k < NUM_BLOCKS; ++k) {
            total_entries += (size_t)bsizes[rows[k]] * (size_t)bsizes[cols[k]];
        }

        float complex *data = (float complex*)malloc(total_entries * sizeof *data);
        if (!data) { fprintf(stderr, "Allocation failed for data buffer\n"); free(block_lengths); return 1; }

        // Pack each block sequentially with running offset
        size_t off_data = 0;

        // 1:(0,0)
        pack_block_rect(dense, n, off[0], off[0], bsizes[0], bsizes[0], off_data, data);
        off_data += (size_t)bsizes[0] * (size_t)bsizes[0];

        // 2:(1,1)
        pack_block_rect(dense, n, off[1], off[1], bsizes[1], bsizes[1], off_data, data);
        off_data += (size_t)bsizes[1] * (size_t)bsizes[1];

        // 3:(2,1)
        pack_block_rect(dense, n, off[2], off[1], bsizes[2], bsizes[1], off_data, data);
        off_data += (size_t)bsizes[2] * (size_t)bsizes[1];

        // 4:(3,1)
        pack_block_rect(dense, n, off[3], off[1], bsizes[3], bsizes[1], off_data, data);
        off_data += (size_t)bsizes[3] * (size_t)bsizes[1];

        // 5:(1,2)
        pack_block_rect(dense, n, off[1], off[2], bsizes[1], bsizes[2], off_data, data);
        off_data += (size_t)bsizes[1] * (size_t)bsizes[2];

        // 6:(2,2)
        pack_block_rect(dense, n, off[2], off[2], bsizes[2], bsizes[2], off_data, data);
        off_data += (size_t)bsizes[2] * (size_t)bsizes[2];

        // 7:(1,3)
        pack_block_rect(dense, n, off[1], off[3], bsizes[1], bsizes[3], off_data, data);
        off_data += (size_t)bsizes[1] * (size_t)bsizes[3];

        // 8:(3,3)
        pack_block_rect(dense, n, off[3], off[3], bsizes[3], bsizes[3], off_data, data);
        off_data += (size_t)bsizes[3] * (size_t)bsizes[3];

        int status = create(bsf, rows, cols, NUM_BLOCKS, block_lengths, data);
        if (status != 0) {
            fprintf(stderr, "Create() failed: %d\n", status);
            free(data); free(block_lengths);
            return 1;
        }

        free(data);
        free(block_lengths);
        return 0;
    } else if (block_structure == 3) {
        const int bsizes[4] = { b+1, b-(b/2), b-1, b+(b/2) };

        // Validate n matches total of block sizes
        int total = 0; for (int k = 0; k < 4; ++k) total += bsizes[k];
        if (total != n) { fprintf(stderr, "n (%d) != sum(bsizes) (%d)\n", n, total); return 1; }

        // Prefix sums for block offsets
        int off[4]; off[0] = 0; for (int k = 1; k < 4; ++k) off[k] = off[k-1] + bsizes[k-1];

        // Zero blocks (same pattern as structure 1)
        const int zero_blocks[][2] = {
            {0,1},{0,2},{0,3},{1,0},{2,0},{2,3},{3,0},{3,2}
        };
        for (size_t z = 0; z < sizeof(zero_blocks)/sizeof(zero_blocks[0]); ++z) {
            int bi = zero_blocks[z][0], bj = zero_blocks[z][1];
            int r0 = off[bi], c0 = off[bj];
            int rlen = bsizes[bi], clen = bsizes[bj];
            for (int j = 0; j < clen; ++j)
                for (int i = 0; i < rlen; ++i)
                    dense[(size_t)(r0 + i)*(size_t)n + (size_t)(c0 + j)] = 0.0f + 0.0f * I;
        }

        // Non-zero blocks (same order as structure 1)
        const int rows[NUM_BLOCKS] = {0, 1, 2, 3, 1, 2, 1, 3};
        const int cols[NUM_BLOCKS] = {0, 1, 1, 1, 2, 2, 3, 3};

        // block_lengths
        int *block_lengths = (int*)malloc(4 * sizeof *block_lengths);
        if (!block_lengths) { fprintf(stderr, "Allocation failed for block_lengths\n"); return 1; }
        for (int r = 0; r < 4; ++r) block_lengths[r] = bsizes[r];

        // Total entries in flat data
        size_t total_entries = 0;
        for (int k = 0; k < NUM_BLOCKS; ++k) {
            total_entries += (size_t)bsizes[rows[k]] * (size_t)bsizes[cols[k]];
        }

        float complex *data = (float complex*)malloc(total_entries * sizeof *data);
        if (!data) { fprintf(stderr, "Allocation failed for data buffer\n"); free(block_lengths); return 1; }

        // Pack with running offset
        size_t off_data = 0;

        // 1:(0,0)
        pack_block_rect(dense, n, off[0], off[0], bsizes[0], bsizes[0], off_data, data);
        off_data += (size_t)bsizes[0] * (size_t)bsizes[0];

        // 2:(1,1)
        pack_block_rect(dense, n, off[1], off[1], bsizes[1], bsizes[1], off_data, data);
        off_data += (size_t)bsizes[1] * (size_t)bsizes[1];

        // 3:(2,1)
        pack_block_rect(dense, n, off[2], off[1], bsizes[2], bsizes[1], off_data, data);
        off_data += (size_t)bsizes[2] * (size_t)bsizes[1];

        // 4:(3,1)
        pack_block_rect(dense, n, off[3], off[1], bsizes[3], bsizes[1], off_data, data);
        off_data += (size_t)bsizes[3] * (size_t)bsizes[1];

        // 5:(1,2)
        pack_block_rect(dense, n, off[1], off[2], bsizes[1], bsizes[2], off_data, data);
        off_data += (size_t)bsizes[1] * (size_t)bsizes[2];

        // 6:(2,2)
        pack_block_rect(dense, n, off[2], off[2], bsizes[2], bsizes[2], off_data, data);
        off_data += (size_t)bsizes[2] * (size_t)bsizes[2];

        // 7:(1,3)
        pack_block_rect(dense, n, off[1], off[3], bsizes[1], bsizes[3], off_data, data);
        off_data += (size_t)bsizes[1] * (size_t)bsizes[3];

        // 8:(3,3)
        pack_block_rect(dense, n, off[3], off[3], bsizes[3], bsizes[3], off_data, data);
        off_data += (size_t)bsizes[3] * (size_t)bsizes[3];

        int status = create(bsf, rows, cols, NUM_BLOCKS, block_lengths, data);
        if (status != 0) {
            fprintf(stderr, "Create() failed: %d\n", status);
            free(data); free(block_lengths);
            return 1;
        }

        free(data);
        free(block_lengths);
        return 0;
    } else {
        return -1;
    }
}

// ===========================================================================
// Print information for verifying
// ===========================================================================
void print_test_matrix_information(int n, float complex *dense, block_sparse_format *bsf) {
    // Print dense matrix
    printf("\nDense matrix (row-major):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("(%5.2f,%5.2f) ", crealf(dense[i*n + j]), cimagf(dense[i*n + j]));
        printf("\n");
    }
    
    // Print information about the sparse matrix
    printf("Global size: %d x %d\n", bsf->n, bsf->n);
    for (int r = 0; r < bsf->num_rows; r++)
        printf("Block row %d: rows %d to %d\n", r, bsf->rows[r].range.start, bsf->rows[r].range.end);
    for (int b = 0; b < bsf->num_blocks; b++) {
        printf("Block %d: row slice %d, col slice %d, size %d, offset %d\n",
            b, bsf->row_indices[b], bsf->col_indices[b], bsf->block_sizes[b], bsf->offsets[b]);
    }

    // Print blocks
    for (int bidx = 0; bidx < bsf->num_blocks; bidx++) {
        int row = bsf->row_indices[bidx];
        int col = bsf->col_indices[bidx];
        int row_start = bsf->rows[row].range.start;
        int row_end   = bsf->rows[row].range.end;
        int col_start = bsf->cols[col].range.start;
        int col_end   = bsf->cols[col].range.end;
        int rows = row_end - row_start + 1;
        int cols = col_end - col_start + 1;

        printf("\nBlock %d at (%d,%d), size %d x %d:\n",
            bidx, row, col, rows, cols);
        float complex *blk_data = bsf->flat_data + (size_t)bsf->offsets[bidx];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("(%5.2f,%5.2f) ", crealf(blk_data[(size_t)j*(size_t)rows + (size_t)i]),
                                       cimagf(blk_data[(size_t)j*(size_t)rows + (size_t)i]));
            }
            printf("\n");
        }
    }
}


#endif