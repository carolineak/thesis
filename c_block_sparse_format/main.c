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

// Helper to compute matvec on dense (row-major) matrix
// Y = alpha * A * X + beta * Y
void matvec_dense(int M, int N,
                          const float complex *A, // row-major, len M*N
                          const float complex *X, // len N
                          float complex *Y,       // len M
                          float complex alpha, float complex beta)
{
    cblas_cgemv(CblasRowMajor, CblasNoTrans,
                M, N,
                &alpha, A, N,   // lda = row length for row-major
                X, 1,
                &beta, Y, 1);
}

// Max absolute difference between two complex vectors
static float max_abs_diff(const float complex *a, const float complex *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = cabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Factor A (row-major, n x n) in place: A = P * L * U
// After factorisation, permute rows back to original order using ipiv
static int lu_dense(float complex *A, int n, lapack_int *ipiv) {
    lapack_int info = LAPACKE_cgetrf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n,
                                     (lapack_complex_float*)A, (lapack_int)n, ipiv);
    if (info != 0) return (int)info;

    return 0;
}

// Print L and U from the compact LU storage
static void print_lu(const float complex *A, int n) {
    printf("\nU (upper triangular):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) { printf("    --        "); }
            else       { printf("(%5.2f,%5.2f) ", crealf(A[i*n + j]), cimagf(A[i*n + j])); }
        }
        printf("\n");
    }
    printf("\nL (unit lower triangular):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) { printf("    --        "); }
            else if (i == j) { printf("(%5.2f,%5.2f) ", 1.0f, 0.0f); }
            else { printf("(%5.2f,%5.2f) ", crealf(A[i*n + j]), cimagf(A[i*n + j])); }
        }
        printf("\n");
    }
}

// Print L and U from a bsf matrix after sparse_lu
static void print_bsf_lu(const block_sparse_format *bsf) {
    int n = bsf->m; // assuming square
    float complex *L = (float complex*)calloc((size_t)n * (size_t)n, sizeof(float complex));
    float complex *U = (float complex*)calloc((size_t)n * (size_t)n, sizeof(float complex));
    // Fill L and U from blocks
    for (int bidx = 0; bidx < bsf->num_blocks; bidx++) {
        const matrix_block *blk = &bsf->blocks[bidx];
        int row_block = bsf->row_indices[bidx];
        int col_block = bsf->col_indices[bidx];
        int row_start = bsf->rows[row_block].range.start;
        int col_start = bsf->cols[col_block].range.start;
        for (size_t j = 0; j < blk->cols; j++) {
            for (size_t i = 0; i < blk->rows; i++) {
                int global_row = row_start + (int)i;
                int global_col = col_start + (int)j;
                if (row_block > col_block) {
                    // Strictly lower block: belongs to L
                    L[global_row * n + global_col] = matrix_block_get(blk, i, j);
                } else if (row_block < col_block) {
                    // Strictly upper block: belongs to U
                    U[global_row * n + global_col] = matrix_block_get(blk, i, j);
                } else {
                    // Diagonal block: split into L and U
                    if (global_row > global_col) {
                        L[global_row * n + global_col] = matrix_block_get(blk, i, j);
                    } else if (global_row == global_col) {
                        L[global_row * n + global_col] = 1.0f + 0.0f*I;
                        U[global_row * n + global_col] = matrix_block_get(blk, i, j);
                    } else {
                        U[global_row * n + global_col] = matrix_block_get(blk, i, j);
                    }
                }
            }
        }
    }
    printf("\nU from bsf:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) { printf("    --        "); }
            else       { printf("(%5.2f,%5.2f) ", crealf(U[i*n + j]), cimagf(U[i*n + j])); }
        }
        printf("\n");
    }
    printf("\nL from bsf:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) { printf("    --        "); }
            else if (i == j) { printf("(%5.2f,%5.2f) ", 1.0f, 0.0f); }
            else { printf("(%5.2f,%5.2f) ", crealf(L[i*n + j]), cimagf(L[i*n + j])); }
        }
        printf("\n");
    }
    printf("\n");
    free(L);
    free(U);
}

int main(void) {
    // Parameters
    const int n = 8;   // Matrix size
    const int b = 2;   // Block size
    #define NUM_BLOCKS 8 // Number of blocks
    block_sparse_format bsf;

    int print = 2; 
    // 0 will print nothing
    // 1 will print only results
    // 2 will print everything

    int block_structure = 1;
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

    // Create dense matrix and fill it with random complex numbers (single precision)
    // ===================================================================
    float complex *dense = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));

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
        int status = create(&bsf, rows, cols, values, NUM_BLOCKS);
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
        int status = create(&bsf, rows, cols, values, NUM_BLOCKS);
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

    // Print information for verifying
    // ===================================================================
    // Print dense matrix
    if (print >= 2) {
        printf("\nDense matrix (row-major):\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("(%5.2f,%5.2f) ", crealf(dense[i*n + j]), cimagf(dense[i*n + j]));
            printf("\n");
        }
        
        // Print information about the sparse matrix
        printf("Global size: %d x %d\n", bsf.m, bsf.n);
        for (int r = 0; r < bsf.num_rows; r++)
        printf("Row slice %d: [%d .. %d]\n", r, bsf.rows[r].range.start, bsf.rows[r].range.end);
        for (int c = 0; c < bsf.num_cols; c++)
        printf("Col slice %d: [%d .. %d]\n", c, bsf.cols[c].range.start, bsf.cols[c].range.end);
        
        
        // Print blocks
        for (int bidx = 0; bidx < bsf.num_blocks; bidx++) {
            const matrix_block *Bv = &bsf.blocks[bidx];
            printf("\nBlock %d at (%d,%d), size %zu x %zu:\n",
                bidx, bsf.row_indices[bidx], bsf.col_indices[bidx], Bv->rows, Bv->cols);
            for (size_t i = 0; i < Bv->rows; i++) {
                for (size_t j = 0; j < Bv->cols; j++)
                printf("(%5.2f,%5.2f) ", 
                    crealf(matrix_block_get(Bv, i, j)), 
                    cimagf(matrix_block_get(Bv, i, j)));
                    printf("\n");
            }
        }
    }

    // Compute matvec for both dense and sparse matrix to test validity
    // ===================================================================
    // Create input vector x and outputs y_dense, y_bsf
    float complex *x       = (float complex*)malloc((size_t)n * sizeof(float complex));
    float complex *y_dense = (float complex*)malloc((size_t)n * sizeof(float complex));
    float complex *y_bsf   = (float complex*)malloc((size_t)n * sizeof(float complex));
    if (!x || !y_dense || !y_bsf) {
        fprintf(stderr, "Alloc x/y failed\n");
        return 1;
    }

    // Fill x with random complex; zero outputs
    for (int i = 0; i < n; i++) {
        x[i] = crand();
        y_dense[i] = 0.0f + 0.0f*I;
        y_bsf[i]   = 0.0f + 0.0f*I;
    }

    // Compute y_dense = dense * x  (alpha=1, beta=0)
    matvec_dense(n, n, dense, x, y_dense, 1.0f + 0.0f*I, 0.0f + 0.0f*I);

    // Compute y_bsf = bsf * x using sparse_matvec
    if (sparse_matvec(&bsf, x, n, y_bsf, n) != 0) {
        fprintf(stderr, "sparse_matvec failed\n");
        return 1;
    }

    // Compare results
    float err = max_abs_diff(y_dense, y_bsf, n);
    if (print >= 1) printf("\nMax |y_dense - y_bsf| = %.6f\n", err);

    // Compute relative error
    float norm_y_dense = 0.0f;
    for (int i = 0; i < n; i++) norm_y_dense += cabsf(y_dense[i]) * cabsf(y_dense[i]);
    norm_y_dense = sqrtf(norm_y_dense);
    float rel_err1 = err / norm_y_dense;
    if (print >= 1) printf("\nRelative error |y_dense - y_bsf| / |y_dense| = %.8f\n", rel_err1);

    // Print vectors
    if (print >= 2) {
        printf("\nFirst few entries:\n");
        for (int i = 0; i < (n < 8 ? n : 8); i++) {
            printf("y_dense[%d] = (%5.2f,%5.2f)   y_bsf[%d] = (%5.2f,%5.2f)\n",
                i, crealf(y_dense[i]), cimagf(y_dense[i]),
                i, crealf(y_bsf[i]),   cimagf(y_bsf[i]));
        }
    }  

    // LU factorisation of dense matrix
    // ===================================================================
    lapack_int *ipiv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));
    if (!ipiv) { fprintf(stderr, "Alloc ipiv failed\n"); return 1; }

    int info = lu_dense(dense, n, ipiv);
    if (info < 0) {
        fprintf(stderr, "cgetrf: illegal argument %d\n", -info);
    } else if (info > 0) {
        fprintf(stderr, "cgetrf: U(%d,%d) is exactly zero (singular)\n", info, info);
    } else {
        if (print >= 1) printf("\nDense LU factorisation successful.\n");
        // if (print >= 2) print_lu(dense, n);
    }

    // LU factorisation of block sparse matrix
    // ===================================================================
    // In order to check correctness, we would like to compute A*x = b where A is the original matrix and x is some random vector.
    // We can then compute the same vector using the LU factors from the block sparse format and compare the solutions b1 and b2

    // Generate x filled with ones
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f + 0.0f*I;
    }

    // Compute b1 = A*x using sparse_matvec
    float complex *b1 = (float complex*)malloc((size_t)n * sizeof(float complex));
    if (!b) { fprintf(stderr, "Alloc b failed\n"); return 1; }
    if (sparse_matvec(&bsf, x, n, b1, n) != 0) {
        fprintf(stderr, "sparse_matvec failed\n");
        return 1;
    }

    float complex *fillin_matrix = (float complex*)calloc((size_t)n * (size_t)n, sizeof(float complex));
    if (!fillin_matrix) { fprintf(stderr, "Alloc fillin_matrix failed\n"); return 1; }

    // Now factor the block sparse matrix
    if (block_structure == 0) {
        int bsf_lu_status = sparse_lu(&bsf);
        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
        } else {
            if (print >= 1) printf("\nBlock sparse LU factorisation successful.\n");
        }
    } else if (block_structure == 1) {
        int bsf_lu_status = sparse_lu_with_fill_ins(&bsf, fillin_matrix);
        // int bsf_lu_status = sparse_lu(&bsf);
        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu_with_fill_ins failed: %d\n", bsf_lu_status);
        } else {
            if (print >= 1) printf("\nBlock sparse LU factorisation successful.\n");
        }
    } else {
        fprintf(stderr, "Invalid block_structure %d\n", block_structure);
        free(dense);
        return 1;
    }

    
    // if (print >= 2) print_bsf_lu(&bsf);   

    // Save x in b2
    float complex *b2 = (float complex*)malloc((size_t)n * sizeof(float complex));
    if (!b2) { fprintf(stderr, "Alloc b2 failed\n"); return 1; }
    memcpy(b2, x, (size_t)n * sizeof(float complex));

    if (block_structure == 0) {
        // Compute b2 = pLUx using sparse_trimul - first on U then on L
        if (sparse_trimul(&bsf, b2, 'U') != 0) {
            fprintf(stderr, "sparse_trimul failed\n");
            return 1;
        }
        if (sparse_trimul(&bsf, b2, 'L') != 0) {
            fprintf(stderr, "sparse_trimul failed\n");
            return 1;
        }

        if (print >= 2) {
            printf("\nFirst few entries of b1 and b2:\n");
            for (int i = 0; i < (n < 8 ? n : 8); i++) {
                printf("b1[%d] = (%5.2f,%5.2f)   b2[%d] = (%5.2f,%5.2f)\n",
                    i, crealf(b1[i]), cimagf(b1[i]),
                    i, crealf(b2[i]), cimagf(b2[i]));
            }
        }
    } else if (block_structure == 1) {
        // Compute b2 = L_sL_dU_dU_sx

        // print b2 before
        if (print >= 2) {
            printf("\nb2 before applying LU factors:\n");
            for (int i = 0; i < (n < 8 ? n : 8); i++) {
                printf("b2[%d] = (%5.2f,%5.2f)\n",
                    i, crealf(b2[i]), cimagf(b2[i]));
            }
        }

        // trimul b2 = U_sparse * b2
        sparse_trimul(&bsf, b2, 'U');
        if (sparse_trimul(&bsf, b2, 'U') != 0) {
            fprintf(stderr, "sparse_trimul failed\n");
            return 1;
        }

        // print b2 after U_sparse
        if (print >= 2) {
            printf("\nb2 after applying U_sparse:\n");
            for (int i = 0; i < (n < 8 ? n : 8); i++) {
                printf("b2[%d] = (%5.2f,%5.2f)\n",
                    i, crealf(b2[i]), cimagf(b2[i]));
            }
        }

        // // trimul b2 = U_dense * b2
        // cblas_ctrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
        //             (lapack_int)n,
        //             (const float complex*)fillin_matrix, (lapack_int)n,
        //             (float complex*)b2, (lapack_int)1);

        // // print b2 after U_dense
        // if (print >= 2) {
        //     printf("\nb2 after applying U_dense:\n");
        //     for (int i = 0; i < (n < 8 ? n : 8); i++) {
        //         printf("b2[%d] = (%5.2f,%5.2f)\n",
        //             i, crealf(b2[i]), cimagf(b2[i]));
        //     }
        // }
        
        // // trimul b2 = L_dense * b2
        // cblas_ctrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit,
        //             (lapack_int)n,
        //             (const float complex*)fillin_matrix, (lapack_int)n,
        //             (float complex*)b2, (lapack_int)1);

        // // print b2 after L_dense
        // if (print >= 2) {
        //     printf("\nb2 after applying L_dense:\n");
        //     for (int i = 0; i < (n < 8 ? n : 8); i++) {
        //         printf("b2[%d] = (%5.2f,%5.2f)\n",
        //             i, crealf(b2[i]), cimagf(b2[i]));
        //     }
        // }        

        // trimul b2 = L_sparse * b2
        sparse_trimul(&bsf, b2, 'L');
        if (sparse_trimul(&bsf, b2, 'L') != 0) {
            fprintf(stderr, "sparse_trimul failed\n");
            return 1;
        }
        
        // print b2 after L_sparse
        if (print >= 2) {
            printf("\nb2 after applying L_sparse:\n");
            for (int i = 0; i < (n < 8 ? n : 8); i++) {
                printf("b2[%d] = (%5.2f,%5.2f)\n",
                    i, crealf(b2[i]), cimagf(b2[i]));
            }
        }
    }

    // Compare b1 and b2
    // (||b1 - b2||) / (||b1||) should be zero (or very small)
    float norm_b1 = 0.0f;
    for (int i = 0; i < n; i++) norm_b1 += cabsf(b1[i]) * cabsf(b1[i]);
    norm_b1 = sqrtf(norm_b1);
    float norm_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = cabsf(b1[i] - b2[i]);
        norm_diff += d * d;
    }
    norm_diff = sqrtf(norm_diff);
    float rel_err = norm_diff / norm_b1;
    if (print >= 1) printf("\nRelative error ||b1 - b2|| / ||b1|| = %.8f\n", rel_err);

    if (print >= 2) {
        printf("\nFirst few entries of b1 and b2:\n");
        for (int i = 0; i < (n < 8 ? n : 8); i++) {
            printf("b1[%d] = (%5.2f,%5.2f)   b2[%d] = (%5.2f,%5.2f)\n",
                i, crealf(b1[i]), cimagf(b1[i]),
                i, crealf(b2[i]), cimagf(b2[i]));
        }
    }

    float complex *test = (float complex*)malloc((size_t)2 * (size_t)2 * sizeof(float complex));
    test[0] = 0.97f + 0.49f*I;
    test[1] = 0.77f + 0.29f*I;
    test[2] = 0.66f + 0.19f*I;
    test[3] = 0.35f + 0.89f*I;

    lu_dense(test, 2, ipiv);
    printf("\nTest matrix LU:\n");
    print_lu(test, 2);
    free(test);


    // Cleanup
    // ===================================================================
    bsf_free(&bsf);
    free(dense);
    free(x);
    free(y_dense);
    free(y_bsf);
    free(ipiv);
    free(b1);
    free(b2);
    return 0;
}

