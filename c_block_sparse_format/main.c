#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
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
void dense_cgemv_rowmajor(int M, int N,
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
static int lu_factor_rowmajor(float complex *A, int n, lapack_int *ipiv) {
    lapack_int info = LAPACKE_cgetrf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n,
                                     (lapack_complex_float*)A, (lapack_int)n, ipiv);
    return (int)info; // info = 0 success; >0 means singular at U(info,info)
}

// Print L and U from the compact LU storage
static void print_lu(const float complex *A, int n) {
    printf("\nU (upper triangular):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) { printf("   --        "); }
            else       { printf("(%5.2f,%5.2f) ", crealf(A[i*n + j]), cimagf(A[i*n + j])); }
        }
        printf("\n");
    }
    printf("\nL (unit lower triangular):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) { printf("   --        "); }
            else if (i == j) { printf("(%5.2f,%5.2f) ", 1.0f, 0.0f); }
            else { printf("(%5.2f,%5.2f) ", crealf(A[i*n + j]), cimagf(A[i*n + j])); }
        }
        printf("\n");
    }
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

    // Create dense matrix and fill it with random complex numbers (single precision)
    // ===================================================================
    float complex *dense = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));

    for (int i = 0; i < n*n; i++) dense[i] = crand();

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
    block_sparse_format bsf;
    int status = create(&bsf, rows, cols, values, NUM_BLOCKS);
    if (status != 0) {
        fprintf(stderr, "Create() failed: %d\n", status);
        for (int i = 0; i < NUM_BLOCKS; i++) matrix_block_free(&values[i]);
        free(dense);
        return 1;
    }

    // Print information for verifying
    // ===================================================================
    // Print dense matrix
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
    dense_cgemv_rowmajor(n, n, dense, x, y_dense, 1.0f + 0.0f*I, 0.0f + 0.0f*I);

    // Compute y_bsf = bsf * x using sparse_matvec
    if (sparse_matvec(&bsf, x, n, y_bsf, n) != 0) {
        fprintf(stderr, "sparse_matvec failed\n");
        return 1;
    }

    // Compare results
    float err = max_abs_diff(y_dense, y_bsf, n);
    printf("\nMax |y_dense - y_bsf| = %.6f\n", err);

    // Print vectors
    printf("\nFirst few entries:\n");
    for (int i = 0; i < (n < 8 ? n : 8); i++) {
        printf("y_dense[%d] = (%5.2f,%5.2f)   y_bsf[%d] = (%5.2f,%5.2f)\n",
               i, crealf(y_dense[i]), cimagf(y_dense[i]),
               i, crealf(y_bsf[i]),   cimagf(y_bsf[i]));
    }

    // LU factorization of dense matrix
    // ===================================================================
    lapack_int *ipiv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));
    if (!ipiv) { fprintf(stderr, "Alloc ipiv failed\n"); return 1; }

    int info = lu_factor_rowmajor(dense, n, ipiv);
    if (info < 0) {
        fprintf(stderr, "cgetrf: illegal argument %d\n", -info);
    } else if (info > 0) {
        fprintf(stderr, "cgetrf: U(%d,%d) is exactly zero (singular)\n", info, info);
    } else {
        printf("\nLU factorization successful.\n");
        // Print L and U
        print_lu(dense, n);

        // Show pivot indices
        printf("Pivot indices: ");
        for (int i = 0; i < n; i++) printf("%d ", (int)ipiv[i]);
        printf("\n");
    }

    // Cleanup
    // ===================================================================
    bsf_free(&bsf);
    free(dense);
    free(x);
    free(y_dense);
    free(y_bsf);
    free(ipiv);
    return 0;
}
