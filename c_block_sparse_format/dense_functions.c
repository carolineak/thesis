#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>
#include "dense_functions.h"

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

// ===========================================================================
// Factor A (col-major, n x n) in place: A = P * L * U
// After factorisation, permute rows back to original order using ipiv
// ===========================================================================
int dense_lu(float complex *A, int n, lapack_int *ipiv) {
    lapack_int info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, (lapack_int)n, (lapack_int)n,
                                     (lapack_complex_float*)A, (lapack_int)n, ipiv);
    if (info != 0) return (int)info;

    return 0;
}

// ===========================================================================
// Print L and U from the compact LU storage
// ===========================================================================
void dense_print_lu(const float complex *A, int n) {
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
// ===========================================================================
// Print a dense matrix
// ===========================================================================
void dense_print_matrix(const float complex *A, int n) {
    printf("\nDense matrix (row-major):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("(%5.2f,%5.2f) ", crealf(A[i*n + j]), cimagf(A[i*n + j]));
        printf("\n");
    }
}

// ===========================================================================
// Computes B = P^TL_dU_dI
// ===========================================================================
void dense_identity_test(int n, float complex *A, float complex *B, int *piv, int lu_factorise_dense) {

    // Allocate work vectors
    float complex *v = (float complex*)calloc((size_t)n, sizeof(float complex));
    float complex *y = (float complex*)calloc((size_t)n, sizeof(float complex));

    // Build columns of B by applying A to each unit basis vector
    for (int j = 0; j < n; ++j) {
        // Set v = e_j
        for (int i = 0; i < n; ++i) v[i] = 0.0f + 0.0f*I;
        v[j] = 1.0f + 0.0f*I;

        if (lu_factorise_dense){
            // v := U_d * v
            cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        (lapack_int)n,
                        (const float complex*)A, (lapack_int)n,
                        (float complex*)v, (lapack_int)1);
    
            // v := L_d * v
            cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                (lapack_int)n,
                (const float complex*)A, (lapack_int)n,
                (float complex*)v, (lapack_int)1);
    
            // v := P^T * v
            apply_pivot_to_vector(v, n, piv);
    
            // Store as column j of dense A (col-major storage)
            for (int i = 0; i < n; ++i) {
                B[j*(size_t)n + i] = v[i];
            }
        } else {
            // y := A * v
            dense_matvec(n, n, A, v, y, (float complex)1, (float complex)0, CblasColMajor);

            // Store as column j of dense A (col-major storage)
            for (int i = 0; i < n; ++i) {
                B[j*(size_t)n + i] = y[i];
            }
        }
    }

    free(v);
    free(y);
}