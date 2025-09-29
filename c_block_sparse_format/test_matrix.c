#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>
#include "block_sparse_format.h"

// Random float complex in [0,1] + i[0,1]
static inline float complex crand(void) {
    return ((float)rand() / RAND_MAX) + ((float)rand() / RAND_MAX) * I;
}

// Copy a b×b submatrix from dense (row-major) into a matrix_block (column-major)
static void copy_block_from_dense(const float complex *dense, int n,
                                  int row_offset, int col_offset, int b,
                                  matrix_block *out)
{
    matrix_block_init(out, (size_t)b, (size_t)b);
    for (int j = 0; j < b; j++)
        for (int i = 0; i < b; i++)
            matrix_block_set(out, (size_t)i, (size_t)j, dense[(row_offset+i)*n + (col_offset+j)]);
}


// ===========================================================================
// Creates a test matrix with a given block structure

// 0: structure that creates no fill-ins
// 1 - 5 -
// - 3 6 -
// 2 4 7 -
// - - - 8

// 1: structure that creates fill-ins
// 1 - - -
// - 2 5 7
// - 3 6 -
// - 4 - 8
// ===========================================================================
int create_test_matrix(int n, int b, int block_structure, float complex *dense, block_sparse_format *bsf) {

    #define NUM_BLOCKS 8

    // Create dense matrix and fill it with random complex numbers (single precision)
    // ===================================================================
    // float complex *dense = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));

    for (int i = 0; i < n*n; i++) dense[i] = crand();

    // Create block sparse matrix on the chosen structure
    if (block_structure == 0) {
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

        // Get blocks from dense matrix (all are b×b)
        // 1:(0,0)  2:(2,0)  3:(1,1)  4:(2,1)  5:(0,2)  6:(1,2)  7:(2,2)  8:(3,3)
        matrix_block values[NUM_BLOCKS];
        copy_block_from_dense(dense, n, 0*b, 0*b, b, &values[0]); // 1
        copy_block_from_dense(dense, n, 2*b, 0*b, b, &values[1]); // 2
        copy_block_from_dense(dense, n, 1*b, 1*b, b, &values[2]); // 3
        copy_block_from_dense(dense, n, 2*b, 1*b, b, &values[3]); // 4
        copy_block_from_dense(dense, n, 0*b, 2*b, b, &values[4]); // 5
        copy_block_from_dense(dense, n, 1*b, 2*b, b, &values[5]); // 6
        copy_block_from_dense(dense, n, 2*b, 2*b, b, &values[6]); // 7
        copy_block_from_dense(dense, n, 3*b, 3*b, b, &values[7]); // 8

        // Set block pattern (0-based)
        int rows[NUM_BLOCKS] = {0, 2, 1, 2, 0, 1, 2, 3};
        int cols[NUM_BLOCKS] = {0, 0, 1, 1, 2, 2, 2, 3};

        // Create sparse matrix
        int status = create(bsf, rows, cols, values, NUM_BLOCKS);
        if (status != 0) {
            fprintf(stderr, "Create() failed: %d\n", status);
            for (int i = 0; i < NUM_BLOCKS; i++) matrix_block_free(&values[i]);
            free(dense);
            return 1;
        }

    } else if (block_structure == 1) {
        // Pattern of zero blocks:
        // (0,1), (0,2), (0,3), (1,0), (2,0), (2,3), (3,0), (3,2)
        int zero_blocks[][2] = {
            {0,1},{0,2},{0,3},{1,0},{2,0},{2,3},{3,0},{3,2}
        };
        for (size_t k = 0; k < sizeof(zero_blocks)/sizeof(zero_blocks[0]); k++) {
            int row_offset = zero_blocks[k][0] * b;
            int col_offset = zero_blocks[k][1] * b;
            for (int i = 0; i < b; i++)
                for (int j = 0; j < b; j++)
                    dense[(row_offset+i)*n + (col_offset+j)] = 0;
        }

        // Get blocks from dense matrix (all are b×b)
        // 1:(0,0)  2:(1,1)  3:(2,1)  4:(3,1)  5:(1,2)  6:(2,2)  7:(1,3)  8:(3,3)   
        matrix_block values[NUM_BLOCKS];
        copy_block_from_dense(dense, n, 0*b, 0*b, b, &values[0]); // 1
        copy_block_from_dense(dense, n, 1*b, 1*b, b, &values[1]); // 2
        copy_block_from_dense(dense, n, 2*b, 1*b, b, &values[2]); // 3
        copy_block_from_dense(dense, n, 3*b, 1*b, b, &values[3]); // 4
        copy_block_from_dense(dense, n, 1*b, 2*b, b, &values[4]); // 5
        copy_block_from_dense(dense, n, 2*b, 2*b, b, &values[5]); // 6
        copy_block_from_dense(dense, n, 1*b, 3*b, b, &values[6]); // 7
        copy_block_from_dense(dense, n, 3*b, 3*b, b, &values[7]); // 8

        // Set block pattern (0-based)
        int rows[NUM_BLOCKS] = {0, 1, 2, 3, 1, 2, 1, 3};
        int cols[NUM_BLOCKS] = {0, 1, 1, 1, 2, 2, 3, 3};

        // Create sparse matrix
        int status = create(bsf, rows, cols, values, NUM_BLOCKS);
        if (status != 0) {
            fprintf(stderr, "Create() failed: %d\n", status);
            for (int i = 0; i < NUM_BLOCKS; i++) matrix_block_free(&values[i]);
            free(dense);
            return 1;
        }

    } else {
        fprintf(stderr, "Invalid block_structure %d\n", block_structure);
        free(dense);
        return 1;
    }

    return 0;
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
    printf("Global size: %d x %d\n", bsf->m, bsf->n);
    for (int r = 0; r < bsf->num_rows; r++)
    printf("Row slice %d: [%d .. %d]\n", r, bsf->rows[r].range.start, bsf->rows[r].range.end);
    for (int c = 0; c < bsf->num_cols; c++)
    printf("Col slice %d: [%d .. %d]\n", c, bsf->cols[c].range.start, bsf->cols[c].range.end);
    
    
    // Print blocks
    for (int bidx = 0; bidx < bsf->num_blocks; bidx++) {
        const matrix_block *Bv = &bsf->blocks[bidx];
        printf("\nBlock %d at (%d,%d), size %zu x %zu:\n",
            bidx, bsf->row_indices[bidx], bsf->col_indices[bidx], Bv->rows, Bv->cols);
        for (size_t i = 0; i < Bv->rows; i++) {
            for (size_t j = 0; j < Bv->cols; j++)
            printf("(%5.2f,%5.2f) ", 
                crealf(matrix_block_get(Bv, i, j)), 
                cimagf(matrix_block_get(Bv, i, j)));
                printf("\n");
        }
    }
}