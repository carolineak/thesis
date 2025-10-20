#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <sys/time.h>
#include "block_sparse_format.h"
#include "test_matrix.h"

// ===========================================================================
// Compute matvec on dense matrix
// Y = alpha * A * X + beta * Y
// ===========================================================================
void dense_matvec(int M, int N,
                          const float complex *A, // row-major, len M*N
                          const float complex *X, // len N
                          float complex *Y,       // len M
                          float complex alpha, 
                          float complex beta,
                          const enum CBLAS_ORDER order)
{
        cblas_cgemv(order, CblasNoTrans,
                    M, N,
                    &alpha, A, N,   // lda = row length for row-major
                    X, 1,
                    &beta, Y, 1);
}

void run_data_structure_test(int n, int b, int block_structure) {
    block_sparse_format bsf;
    float complex *dense = malloc((size_t)n * n * sizeof(float complex));

    create_test_matrix(n, b, block_structure, dense, &bsf);

    print_test_matrix_information(n, dense, &bsf);

    printf("\nThe sparse matrix A_s:\n");
    sparse_print_matrix(&bsf);

    bsf_free(&bsf);
    free(dense);
}

// ===========================================================================
// Compute the relative error of two complex vectors
// ===========================================================================
double relative_error(const float complex *y_bsf,
                             const float complex *y_dense,
                             int n) {
    if (!y_bsf || !y_dense || n <= 0) return NAN;

    float complex *tmp = (float complex*)malloc((size_t)n * sizeof *tmp);
    if (!tmp) return NAN;

    // tmp <- y_dense
    cblas_ccopy(n, y_dense, 1, tmp, 1);

    // tmp <- tmp + (-1) * y_bsf  == y_dense - y_bsf
    const float complex minus_one = -1.0f + 0.0f*I;
    cblas_caxpy(n, &minus_one, y_bsf, 1, tmp, 1);

    // 2-norms
    float num   = cblas_scnrm2(n, tmp,    1);
    float denom = cblas_scnrm2(n, y_dense,1);

    free(tmp);

    if (denom == 0.0f) return (num == 0.0f) ? 0.0 : INFINITY;
    return (double)num / (double)denom;
}

void run_matvec_test(int n, int b, int block_structure, int print, double tolerance, int *passed) {
    block_sparse_format bsf;
    float complex *dense = malloc((size_t)n * n * sizeof(float complex));
    float complex *x = malloc((size_t)n * sizeof(float complex));
    float complex *y_dense = malloc((size_t)n * sizeof(float complex));
    float complex *y_bsf = malloc((size_t)n * sizeof(float complex));
    if (!dense || !x || !y_dense || !y_bsf) {
        fprintf(stderr, "Allocation failed\n");
        return;
    }
    
    create_test_matrix(n, b, block_structure, dense, &bsf);

    for (int i = 0; i < n; i++) {
        x[i] = 1.0f + 0.0f*I;
        y_dense[i] = 0.0f + 0.0f*I;
        y_bsf[i] = 0.0f + 0.0f*I;
    }

    dense_matvec(n, n, dense, x, y_dense, 1.0f + 0.0f*I, 0.0f + 0.0f*I, CblasRowMajor);
    sparse_matvec(&bsf, x, n, y_bsf, n);

    // |y_dense - y_bsf| / |y_dense|
    double rel_error = relative_error(y_bsf, y_dense, n);
    if (print >= 1) {
        if (rel_error < tolerance) {
            printf("TEST PASSED: Relative error of matvec is %.2e\n", rel_error);
            (*passed)++;
        } else {
            printf("TEST FAILED: Relative error of matvec is %.2e\n", rel_error);
        }
    }

    bsf_free(&bsf);
    free(dense);
    free(x);
    free(y_dense);
    free(y_bsf);
}
