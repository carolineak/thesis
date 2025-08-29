#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "block_sparse_format.h"

// Random float in [0,1]
static inline float frand(void) { return (float)rand() / (float)RAND_MAX; }

// Copy a b×b submatrix from dense (row-major) into a matrix_block (column-major)
static void copy_block_from_dense(const float *dense, int n,
                                  int r0, int c0, int b,
                                  matrix_block *out)
{
    matrix_block_init(out, (size_t)b, (size_t)b);
    for (int j = 0; j < b; j++)
        for (int i = 0; i < b; i++)
            matrix_block_set(out, (size_t)i, (size_t)j, dense[(r0+i)*n + (c0+j)]);
}

int main(void) {
    // Parameters
    const int n = 8;   // Matrix size
    const int b = 2;   // Block size
    #define NUM_BLOCKS 8 // Number of blocks

    // Generate matrix with the block structure
    // 1 - 5 -
    // - 3 6 -
    // 2 4 7 -
    // - - - 8

    // Create dense matrix and fill it with random floats
    // ===================================================================
    float *dense = (float*)malloc((size_t)n * (size_t)n * sizeof(float));

    for (int i = 0; i < n*n; i++) dense[i] = frand();

    // Fill zero-blocks
    // Pattern of zero blocks:
    // (0,1), (0,3), (1,0), (1,3), (2,3), (3,0), (3,1), (3,2)
    int zero_blocks[][2] = {
        {0,1},{0,3},{1,0},{1,3},{2,3},{3,0},{3,1},{3,2}
    };
    for (size_t k = 0; k < sizeof(zero_blocks)/sizeof(zero_blocks[0]); k++) {
        int row_offset = zero_blocks[k][0] * b;
        int col_offset = zero_blocks[k][1] * b;
        for (int i = 0; i < b; i++)
            for (int j = 0; j < b; j++)
                dense[(row_offset+i)*n + (col_offset+j)] = 0;
    }

    // Generate matrix on the block sparse format
    // ===================================================================
    // Get blocks from dense matrix (all are b×b)
    // 1:(0,0)  2:(2,0)  3:(1,1)  4:(2,1)  5:(0,2)  6:(1,2)  7:(2,2)  8:(3,3)
    // matrix_block *values = (matrix_block)malloc(NUM_BLOCKS * sizeof(matrix_block));



    // matrix_block values[NUM_BLOCKS];
    // copy_block_from_dense(dense, n, 0*b, 0*b, b, &values[0]); // 1
    // copy_block_from_dense(dense, n, 2*b, 0*b, b, &values[1]); // 2
    // copy_block_from_dense(dense, n, 1*b, 1*b, b, &values[2]); // 3
    // copy_block_from_dense(dense, n, 2*b, 1*b, b, &values[3]); // 4
    // copy_block_from_dense(dense, n, 0*b, 2*b, b, &values[4]); // 5
    // copy_block_from_dense(dense, n, 1*b, 2*b, b, &values[5]); // 6
    // copy_block_from_dense(dense, n, 2*b, 2*b, b, &values[6]); // 7
    // copy_block_from_dense(dense, n, 3*b, 3*b, b, &values[7]); // 8

    // // Set block pattern (0-based)
    // int rows[NUM_BLOCKS] = {0, 2, 1, 2, 0, 1, 2, 3};
    // int cols[NUM_BLOCKS] = {0, 0, 1, 1, 2, 2, 2, 3};

    // // Create sparse matrix
    // block_sparse_format bsf;
    // int status = create(&bsf, rows, cols, values, NUM_BLOCKS);
    // if (status != 0) {
    //     fprintf(stderr, "Create() failed: %d\n", status);
    //     for (int i = 0; i < NUM_BLOCKS; i++) matrix_block_free(&values[i]);
    //     free(dense);
    //     return 1;
    // }

    // // Print information for verifying
    // // ===================================================================
    // Print dense matrix
    printf("\nDense matrix (row-major):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%5.2f ", dense[i*n + j]);
        printf("\n");
    }

    // // Print information about the sparse matrix
    // printf("Global size: %d x %d\n", bsf.m, bsf.n);
    // for (int r = 0; r < bsf.num_rows; r++)
    //     printf("Row slice %d: [%d .. %d]\n", r, bsf.rows[r].range.start, bsf.rows[r].range.end);
    // for (int c = 0; c < bsf.num_cols; c++)
    //     printf("Col slice %d: [%d .. %d]\n", c, bsf.cols[c].range.start, bsf.cols[c].range.end);


    // // Print blocks
    // for (int bidx = 0; bidx < bsf.num_blocks; bidx++) {
    //     const matrix_block *Bv = &bsf.blocks[bidx];
    //     printf("\nBlock %d at (%d,%d), size %zu x %zu:\n",
    //            bidx, bsf.row_indices[bidx], bsf.col_indices[bidx], Bv->rows, Bv->cols);
    //     for (size_t i = 0; i < Bv->rows; i++) {
    //         for (size_t j = 0; j < Bv->cols; j++)
    //             printf("%5.2f ", matrix_block_get(Bv, i, j));
    //         printf("\n");
    //     }
    // }

    // // Compute matvec for both dense and sparse matrix to test validity
    // // ===================================================================
    // // TODO

    // // Cleanup
    // // ===================================================================
    // bsf_free(&bsf);
    // free(dense);
    return 0;
}
