#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>
#include "block_sparse_format.h"
#include "dense_functions.h"
#include "check_correctness.h"

// Compute the relative error of two complex vectors
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

void all_factors_identity_test(int n, const block_sparse_format *bsf, float complex *A, 
                                      float complex *B, int *piv, int lu_factorise_dense) {

    // Allocate work vectors
    float complex *v = (float complex*)calloc((size_t)n, sizeof(float complex));
    float complex *y = (float complex*)calloc((size_t)n, sizeof(float complex));

    // Build columns of B by applying A to each unit basis vector
    for (int j = 0; j < n; ++j) {
        // Set v = e_j
        for (int i = 0; i < n; ++i) v[i] = 0.0f + 0.0f*I;
        v[j] = 1.0f + 0.0f*I;

        // v := U_s * v
        if (sparse_trimul(bsf, v, 'U') != 0) {
            fprintf(stderr, "sparse_identity_test: sparse_trimul('U') failed at column %d\n", j);
            free(v);
        }
        
        if (lu_factorise_dense) {
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
        } else {
            // y := A * v
            dense_matvec(n, n, A, v, y, (float complex)1, (float complex)0, CblasColMajor);
            
            for (int i = 0; i < n; i ++) {
                v[i] = y[i];
            }         
        }

        // v := L_s * v
        if (sparse_trimul(bsf, v, 'L') != 0) {
            fprintf(stderr, "sparse_identity_test: sparse_trimul('L') failed at column %d\n", j);
            free(v);
        }

        // Store as column j of dense A (col-major storage)
        for (int i = 0; i < n; ++i) {
            B[j*(size_t)n + i] = v[i];
        }
    }

    free(v);
    free(y);
}

void all_factors_vector_test(int n, const block_sparse_format *bsf, float complex *dense, 
                                    float complex *vec_in, complex float *vec_out, int *piv, 
                                    int lu_factorise_dense) {

    float complex *vec_tmp = (float complex*)calloc((size_t)n, sizeof(float complex));

    // v := U_s * v
    if (sparse_trimul(bsf, vec_in, 'U') != 0) {
        fprintf(stderr, "sparse_identity_test: sparse_trimul('U') failed");
    }

    if (lu_factorise_dense) {
        // v := U_d * v
        cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    (lapack_int)n,
                    (const float complex*)dense, (lapack_int)n,
                    (float complex*)vec_in, (lapack_int)1);

        // v := L_d * v
        cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
            (lapack_int)n,
            (const float complex*)dense, (lapack_int)n,
            (float complex*)vec_in, (lapack_int)1);

        // v := P^T * v
        apply_pivot_to_vector(vec_in, n, piv);
    } else {   
        // y := A_d * v
        dense_matvec(n, n, dense, vec_in, vec_tmp, (float complex)1, (float complex)0, CblasColMajor);

        for (int i = 0; i < n; i ++) {
            vec_in[i] = vec_tmp[i];
        }
    }

    // v := L_s * v
    if (sparse_trimul(bsf, vec_in, 'L') != 0) {
        fprintf(stderr, "sparse_identity_test: sparse_trimul('L') failed");
    }

    // Store in vec_out
    for (int i = 0; i < n; ++i) {
        vec_out[i] = vec_in[i];
    }
}