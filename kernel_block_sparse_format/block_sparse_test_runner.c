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
#include "load_binary_data.h"

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
// Run data structure test for synthetic test matrices
// ===========================================================================
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

// ===========================================================================
// Computes b = P^TL_sP^TL_dU_dU_sx
// ===========================================================================
void sparse_dense_trimul(int n, const block_sparse_format *bsf, int dense_size, float complex *dense, 
                                    float complex *vec_in, complex float *vec_out, int *piv, 
                                    int lu_factorise_dense, int *received_fill_in) {

    float complex *vec_tmp = (float complex*)calloc((size_t)dense_size, sizeof(float complex));
    float complex *vec_tmp_unfactorised = (float complex*)calloc((size_t)dense_size, sizeof(float complex));

    // v := U_s * v
    if (sparse_trimul(bsf, vec_in, 'U') != 0) {
        fprintf(stderr, "sparse_identity_test: sparse_trimul('U') failed");
    }

    // From vec_in, move the blocks that received fill-in to a temporary vector
    int offset = 0;
    for (int i = 0; i < bsf->num_rows; i++) {
        if (received_fill_in[i]) {
            const int_range row_rng = bsf->rows[i].range;
            const int M = range_length(row_rng);
            const int dense_start = row_rng.start;
            for (int r = 0; r < M; r++) {
                vec_tmp[offset + r] = vec_in[dense_start + r];
                vec_in[dense_start + r] = 0.0f + 0.0f*I;
            }
            offset += M;
        }
    }

    if (lu_factorise_dense) {
        // v := U_d * v
        cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    (lapack_int)dense_size,
                    (const float complex*)dense, (lapack_int)dense_size,
                    (float complex*)vec_tmp, (lapack_int)1);

        // v := L_d * v
        cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
            (lapack_int)dense_size,
            (const float complex*)dense, (lapack_int)dense_size,
            (float complex*)vec_tmp, (lapack_int)1);


        // v := P^T * v
        apply_inverse_pivot_to_vector(vec_tmp, dense_size, piv);

    } else {   
        // y := A_d * v
        dense_matvec(dense_size, dense_size, dense, vec_tmp, vec_tmp_unfactorised, (float complex)1, (float complex)0, CblasColMajor);

        for (int i = 0; i < dense_size; i ++) {
            vec_tmp[i] = vec_tmp_unfactorised[i];
        }
    }

    // Move the blocks that received fill-in back to vec_in
    offset = 0;
    for (int i = 0; i < bsf->num_rows; i++) {
        if (received_fill_in[i]) {
            const int_range row_rng = bsf->rows[i].range;
            const int M = range_length(row_rng);
            const int dense_start = row_rng.start;
            for (int r = 0; r < M; r++) {
                vec_in[dense_start + r] = vec_tmp[offset + r];;
            }
            offset += M;
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

    free(vec_tmp);
}

// ===========================================================================
// Computes B = P^TL_dU_dI
// ===========================================================================
void dense_identity_test(int n, int A_n, float complex *A, float complex *B, int *piv, int lu_factorise_dense) {

    // Allocate work vectors
    float complex *v = (float complex*)calloc((size_t)n, sizeof(float complex));
    float complex *y = (float complex*)calloc((size_t)n, sizeof(float complex));

    int A_start = n - A_n;

    // Build columns of B by applying A to each unit basis vector
    for (int j = 0; j < n; ++j) {
        // Set v = e_j
        for (int i = 0; i < n; ++i) v[i] = 0.0f + 0.0f*I;
        v[j] = 1.0f + 0.0f*I;

        if (lu_factorise_dense){
            // v := U_d * v
            cblas_ctrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        (lapack_int)A_n,
                        (const float complex*)A, (lapack_int)A_n,
                        (float complex*)v + A_start, (lapack_int)1);
    
            // v := L_d * v
            cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                (lapack_int)A_n,
                (const float complex*)A, (lapack_int)A_n,
                (float complex*)v + A_start, (lapack_int)1);
    
            // v := P^T * v
            apply_pivot_to_vector(v + A_start, A_n, piv);
    
            // Store as column j of dense A (col-major storage)
            for (int i = 0; i < n; ++i) {
                B[j*(size_t)n + i] = v[i];
            }
        } else {
            // y := A * v
            dense_matvec(A_n, A_n, A, v + A_start, y + A_start, (float complex)1, (float complex)0, CblasColMajor);

            // Store as column j of dense A (col-major storage)
            for (int i = 0; i < n; ++i) {
                B[j*(size_t)n + i] = y[i];
            }
        }
    }

    free(v);
    free(y);
}

// ===========================================================================
// Computes B = P^TL_sP^TL_dU_dU_sI
// ===========================================================================
void sparse_dense_identity_test(int n, const block_sparse_format *bsf, int A_n, float complex *A, 
                                      float complex *B, int *piv, int lu_factorise_dense) {

    // Allocate work vectors
    float complex *v = (float complex*)calloc((size_t)n, sizeof(float complex));
    float complex *y = (float complex*)calloc((size_t)A_n, sizeof(float complex));

    int A_start = n - A_n;

    // Build columns of B by applying bsf and A to each unit basis vector
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
                (lapack_int)A_n,
                (const float complex*)A, (lapack_int)A_n,
                (float complex*)v + A_start, (lapack_int)1);
                
            // v := L_d * v
            cblas_ctrmv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                (lapack_int)A_n,
                (const float complex*)A, (lapack_int)A_n,
                (float complex*)v + A_start, (lapack_int)1);
                
            // v := P^T * v
            apply_inverse_pivot_to_vector(v + A_start, A_n, piv);
        } else {
            // y := A * v
            dense_matvec(A_n, A_n, A, v + A_start, y, (float complex)1, (float complex)0, CblasColMajor);
            
            for (int i = 0; i < A_n; i ++) {
                v[i + A_start] = y[i];
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

// ==================================================================
// Run matvec test for synthetic test matrices
// ==================================================================
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

// ==================================================================
// Run LU and triangular multiplication test for synthetic test matrices
// ==================================================================
void run_lu_trimul_test(int n, int b, int block_structure, int print, double tolerance, int *passed) {
    block_sparse_format bsf = {0};
    float complex *dense = malloc((size_t)n * n * sizeof(float complex));
    float complex *x = malloc((size_t)n * sizeof(float complex));
    float complex *b1 = malloc((size_t)n * sizeof(float complex));
    float complex *b2 = malloc((size_t)n * sizeof(float complex));
    float complex *fill_in_matrix = NULL;
    int fill_in_matrix_size = 0;
    int *received_fill_in = NULL;
    int *fill_in_piv = NULL;
    int error = 0;
    struct timeval start, end;

    if (!dense || !x || !b1 || !b2) {
        fprintf(stderr, "Allocation failed\n");
        error = 1;
    }

    if (!error) {
        create_test_matrix(n, b, block_structure, dense, &bsf);

        // Fill x with random values between 0 and 1
        for (int i = 0; i < n; i++) {
            float real_part = (float)rand() / (float)RAND_MAX;
            float imag_part = (float)rand() / (float)RAND_MAX;
            x[i] = real_part + imag_part * I;
        }

        // Compute b1 = A*x using sparse_matvec
        if (sparse_matvec(&bsf, x, n, b1, n) != 0) {
            fprintf(stderr, "sparse_matvec failed\n");
            error = 1;
        }
    }

    // Factorise block sparse matrix
    int lu_factorise_dense = 1;
    if (!error) {
        if (block_structure == 0) {
            gettimeofday(&start, NULL);
            int bsf_lu_status = sparse_lu(&bsf, &fill_in_matrix, &fill_in_matrix_size, &received_fill_in, print);
            gettimeofday(&end, NULL);

            if (bsf_lu_status != 0) {
                fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
                error = 1;
            } else if (print >= 1) {
                printf("\nBlock sparse LU factorisation successful.\n");
            }
        } else if (block_structure >= 1) {
            gettimeofday(&start, NULL);
            int bsf_lu_status = sparse_lu(&bsf, &fill_in_matrix, &fill_in_matrix_size, &received_fill_in, print);
            gettimeofday(&end, NULL);

            if (bsf_lu_status != 0) {
                fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
                error = 1;

            } else {
                fill_in_piv = malloc((size_t)fill_in_matrix_size * sizeof(int));
                if (!fill_in_piv) {
                    fprintf(stderr, "Alloc fill_in_piv failed\n");
                    error = 1;
                } else if (lu_factorise_dense) {
                    dense_lu(fill_in_matrix, fill_in_matrix_size, fill_in_piv);
                }
                if (print >= 1) printf("\nBlock sparse LU factorisation successful.\n");
            }
        } else {
            fprintf(stderr, "Invalid block_structure %d\n", block_structure);
            error = 1;
        }
    }

    // Save x in b2
    if (!error) {
        memcpy(b2, x, (size_t)n * sizeof(float complex));

        if (block_structure == 0) {
            // Compute b2 = pLUx using sparse_trimul - first on U then on L
            if (sparse_trimul(&bsf, b2, 'U') != 0) {
                fprintf(stderr, "sparse_trimul failed\n");
                error = 1;
            }
            if (!error && sparse_trimul(&bsf, b2, 'L') != 0) {
                fprintf(stderr, "sparse_trimul failed\n");
                error = 1;
            }
        } else if (block_structure >= 1) {
            float complex *vec_out = calloc((size_t)n, sizeof(float complex));
            if (!vec_out) {
                fprintf(stderr, "Alloc vec_out failed\n");
                error = 1;
            } else {
                if (fill_in_matrix == NULL) {
                    // No fill-in matrix allocated, just use sparse_trimul
                    if (sparse_trimul(&bsf, b2, 'U') != 0) {
                        fprintf(stderr, "sparse_trimul failed\n");
                        error = 1;
                    }
                    if (!error && sparse_trimul(&bsf, b2, 'L') != 0) {
                        fprintf(stderr, "sparse_trimul failed\n");
                        error = 1;
                    }
                } else {
                    // Use sparse_dense_trimul to include fill-in matrix
                    sparse_dense_trimul(n, &bsf, fill_in_matrix_size, fill_in_matrix, b2, vec_out, fill_in_piv, lu_factorise_dense, received_fill_in);
                    for (int i = 0; i < n; i++) b2[i] = vec_out[i];
                }
                free(vec_out);
            }
        }
    }

    // Compare b1 and b2
    // ||b1 - b2|| / ||b1||
    if (!error) {
        double norm_b1 = 0.0;
        for (int i = 0; i < n; i++) norm_b1 += cabsf(b1[i]) * cabsf(b1[i]);
        norm_b1 = sqrtf(norm_b1);
        double norm_diff = 0.0;
        for (int i = 0; i < n; i++) {
            float d = cabsf(b1[i] - b2[i]);
            norm_diff += d * d;
        }
        norm_diff = sqrtf(norm_diff);
        double rel_err = norm_diff / (norm_b1 > 0.0 ? norm_b1 : 1.0);

        if (print >= 1) {
            if (rel_err < tolerance) {
                printf("TEST PASSED: Relative error of computing b = LUx is %.2e\n", rel_err);
                (*passed)++;
            } else {
                printf("TEST FAILED: Relative error of computing b = LUx is %.2e\n", rel_err);
            }
        }

        if (print >= 2) {
            printf("\nFirst few entries of b1 and b2:\n");
            for (int i = 0; i < (n < 8 ? n : 8); i++) {
                printf("b1[%d] = (%5.2f,%5.2f)   b2[%d] = (%5.2f,%5.2f)\n",
                    i, crealf(b1[i]), cimagf(b1[i]),
                    i, crealf(b2[i]), cimagf(b2[i]));
            }
        }
    }

    double seconds = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Time spend computing LU: %.6f s\n", seconds);

    bsf_free(&bsf);
    free(dense);
    free(x);
    free(b1);
    free(b2);
    free(fill_in_matrix);
    free(received_fill_in);
    free(fill_in_piv);
}

// ==================================================================
// Run LU and triangular multiplication test on binary data file
// ==================================================================
void run_lu_trimul_test_on_bin_data(int print, double tolerance, int *passed, char *data) {
    block_sparse_format bsf = {0};
    float complex *fill_in_matrix = NULL;
    int fill_in_matrix_size = 0;
    int *received_fill_in = NULL;
    int *fill_in_piv = NULL;
    int error = 0;
    struct timeval start, end;

    int err = load_block_sparse_from_bin(data, &bsf);
    if (err != 0) {
        fprintf(stderr, "Failed to load sparse_data.bin, error %d\n", err);
        error = 1;
    }

    // Print information about the loaded block sparse matrix
    // check_block_sparse_format(&bsf);

    int n = bsf.n;

    float complex *x = malloc((size_t)n * sizeof(float complex));
    float complex *b1 = malloc((size_t)n * sizeof(float complex));
    float complex *b2 = malloc((size_t)n * sizeof(float complex));

    // Fill x with random values between 0 and 1
    for (int i = 0; i < n; i++) {
        float real_part = (float)rand() / (float)RAND_MAX;
        float imag_part = (float)rand() / (float)RAND_MAX;
        x[i] = real_part + imag_part * I;
    }

    // Compute b1 = A*x using sparse_matvec
    if (sparse_matvec(&bsf, x, n, b1, n) != 0) {
        fprintf(stderr, "sparse_matvec failed\n");
        error = 1;
    }

    // Print size of matrix A
    printf("Matrix size: %d x %d\n", n, n);

    // Factorize block sparse matrix
    int lu_factorise_dense = 0;
    if (!error) {
        gettimeofday(&start, NULL);
        int bsf_lu_status = sparse_lu(&bsf, &fill_in_matrix, &fill_in_matrix_size, &received_fill_in, print);
        gettimeofday(&end, NULL);

        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu: %d\n", bsf_lu_status);
            error = 1;

        } else {
            fill_in_piv = malloc((size_t)fill_in_matrix_size * sizeof(int));
            if (!fill_in_piv) {
                fprintf(stderr, "Alloc fill_in_piv failed\n");
                error = 1;
            } else if (lu_factorise_dense) {
                dense_lu(fill_in_matrix, fill_in_matrix_size, fill_in_piv);
            }
            if (print >= 1) printf("\nBlock sparse LU factorisation successful.\n");
        }
    }

    // Save x in b2
    if (!error) {
        memcpy(b2, x, (size_t)n * sizeof(float complex));

        float complex *vec_out = calloc((size_t)n, sizeof(float complex));
        if (!vec_out) {
            fprintf(stderr, "Alloc vec_out failed\n");
            error = 1;
        } else {
            if (fill_in_matrix == NULL) {
                // No fill-in matrix allocated, just use sparse_trimul
                if (sparse_trimul(&bsf, b2, 'U') != 0) {
                    fprintf(stderr, "sparse_trimul failed\n");
                    error = 1;
                }
                if (!error && sparse_trimul(&bsf, b2, 'L') != 0) {
                    fprintf(stderr, "sparse_trimul failed\n");
                    error = 1;
                }
            } else {
                // Use sparse_dense_trimul to include fill-in matrix
                sparse_dense_trimul(n, &bsf, fill_in_matrix_size, fill_in_matrix, b2, vec_out, fill_in_piv, lu_factorise_dense, received_fill_in);
                for (int i = 0; i < n; ++i) b2[i] = vec_out[i];
                free(vec_out);
            }
        }
    }

    // Compare b1 and b2
    // ||b1 - b2|| / ||b1||
    if (!error) {
        double norm_b1 = 0.0;
        for (int i = 0; i < n; i++) norm_b1 += cabsf(b1[i]) * cabsf(b1[i]);
        norm_b1 = sqrtf(norm_b1);
        double norm_diff = 0.0;
        for (int i = 0; i < n; i++) {
            float d = cabsf(b1[i] - b2[i]);
            norm_diff += d * d;
        }
        norm_diff = sqrtf(norm_diff);
        double rel_err = norm_diff / (norm_b1 > 0.0 ? norm_b1 : 1.0);

        if (print >= 1) {
            if (rel_err < tolerance) {
                printf("TEST PASSED: Relative error of computing b = LUx is %.2e\n", rel_err);
                (*passed)++;
            } else {
                printf("TEST FAILED: Relative error of computing b = LUx is %.2e\n", rel_err);
            }
        }

        if (print >= 2) {
            printf("\nFirst few entries of b1 and b2:\n");
            for (int i = 0; i < (n < 8 ? n : 8); i++) {
                printf("b1[%d] = (%5.2f,%5.2f)   b2[%d] = (%5.2f,%5.2f)\n",
                    i, crealf(b1[i]), cimagf(b1[i]),
                    i, crealf(b2[i]), cimagf(b2[i]));
            }
        }
    }

    double seconds = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Time spend computing LU: %.6f s\n", seconds);

    bsf_free(&bsf);
    free(x);
    free(b1);
    free(b2);
    free(fill_in_matrix);
    free(received_fill_in);
    free(fill_in_piv);
}

// ==================================================================
// Run LU identity test for synthetic test matrices
// ==================================================================
void run_lu_identity_test(int n, int b, int block_structure, int print) {
    block_sparse_format bsf;
    float complex *dense = malloc((size_t)n * n * sizeof(float complex));
    float complex *fill_in_matrix = NULL;
    int fill_in_matrix_size = 0;
    int *received_fill_in = NULL;
    int *fill_in_piv = NULL;

    if (!dense) {
        fprintf(stderr, "Allocation failed\n");
    }

    create_test_matrix(n, b, block_structure, dense, &bsf);

    printf("The original A:\n");
    sparse_print_matrix(&bsf);

    // Factorise block sparse matrix and fill-in matrix
    int lu_factorise_dense = 1;
    if (block_structure == 0) {
        int bsf_lu_status = sparse_lu(&bsf, &fill_in_matrix, &fill_in_matrix_size, &received_fill_in, print);
        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
        }
    } else if (block_structure >= 1) {
        int bsf_lu_status = sparse_lu(&bsf, &fill_in_matrix, &fill_in_matrix_size, &received_fill_in, print);
        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
        }
        fill_in_piv = malloc((size_t)fill_in_matrix_size * sizeof(int));
        if (!fill_in_piv) {
            fprintf(stderr, "Alloc fill_in_piv failed\n");
        }
        if (lu_factorise_dense) {
            dense_lu(fill_in_matrix, fill_in_matrix_size, fill_in_piv);
        }
    } else {
        fprintf(stderr, "Invalid block_structure %d\n", block_structure);
    }

    printf("\nThe sparse matrix A_s (factorised):\n");
    sparse_print_matrix(&bsf);

    if (lu_factorise_dense) {
        printf("\nThe dense matrix A_d (factorised):\n");
    } else {
        printf("\nThe dense matrix A_d (unfactorised):\n");
    }
    for (int r = 0; r < fill_in_matrix_size; ++r) {
        for (int c = 0; c < fill_in_matrix_size; ++c) {
            printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
        }
        printf("\n");
    }

    float complex *A_sparse_reconstructed = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));
    sparse_identity_test(&bsf, A_sparse_reconstructed);

    printf("\nMatrix reconstructed from L_s*U_s*I:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float complex z = A_sparse_reconstructed[j*(size_t)n + i];
            printf("(%5.2f,%5.2f) ", crealf(z), cimagf(z));
        }
        printf("\n");
    }

    float complex *A_dense_reconstructed = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));
    dense_identity_test(n, fill_in_matrix_size, fill_in_matrix, A_dense_reconstructed, fill_in_piv, lu_factorise_dense);

    if (lu_factorise_dense) {
        printf("\nMatrix reconstructed from L_d*U_d*I:\n");
    } else {
        printf("\nMatrix reconstructed from A_d*I:\n");
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float complex z = A_dense_reconstructed[j*(size_t)n + i];
            printf("(%5.2f,%5.2f) ", crealf(z), cimagf(z));
        }
        printf("\n");
    }

    float complex *A_all_factors_reconstructed = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));
    sparse_dense_identity_test(n, &bsf, fill_in_matrix_size, fill_in_matrix, A_all_factors_reconstructed, fill_in_piv, lu_factorise_dense);

    if (lu_factorise_dense) {
        printf("\nA reconstructed from L_s*L_d*U_d*U_s*I:\n");
    } else {
        printf("\nA reconstructed from L_s*A_d*U_s*I:\n");
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float complex z = A_all_factors_reconstructed[j*(size_t)n + i];
            printf("(%5.2f,%5.2f) ", crealf(z), cimagf(z));
        }
        printf("\n");
    }

    bsf_free(&bsf);
    free(dense);
    free(A_sparse_reconstructed);
    free(A_dense_reconstructed);
    free(A_all_factors_reconstructed);
    free(fill_in_matrix);
    free(received_fill_in);
    free(fill_in_piv);
}

// ==================================================================
// Run LU identity test on binary data file
// ==================================================================
void run_lu_identity_test_with_bin_data(char *data, int print) {
    block_sparse_format bsf;
    float complex *fill_in_matrix = NULL;
    int fill_in_matrix_size = 0;
    int *received_fill_in = NULL;
    int *fill_in_piv = NULL;

    load_block_sparse_from_bin(data, &bsf);
    int n = bsf.n;

    printf("The original A:\n");
    sparse_print_matrix(&bsf);

    // Factorize block sparse matrix and fill-in matrix
    int lu_factorise_dense = 1;
    int bsf_lu_status = sparse_lu(&bsf, &fill_in_matrix, &fill_in_matrix_size, &received_fill_in, print);
    if (bsf_lu_status != 0) {
        fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
    }
    fill_in_piv = malloc((size_t)fill_in_matrix_size * sizeof(int));
    if (!fill_in_piv) {
        fprintf(stderr, "Alloc fill_in_piv failed\n");
    }
    if (lu_factorise_dense) {
        dense_lu(fill_in_matrix, fill_in_matrix_size, fill_in_piv);
    }

    printf("\nThe sparse matrix A_s (factorised):\n");
    sparse_print_matrix(&bsf);

    if (lu_factorise_dense) {
        printf("\nThe dense matrix A_d (factorised):\n");
    } else {
        printf("\nThe dense matrix A_d (unfactorised):\n");
    }
    for (int r = 0; r < fill_in_matrix_size; ++r) {
        for (int c = 0; c < fill_in_matrix_size; ++c) {
            printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
        }
        printf("\n");
    }

    float complex *A_sparse_reconstructed = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));
    sparse_identity_test(&bsf, A_sparse_reconstructed);

    printf("\nMatrix reconstructed from L_s*U_s*I:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float complex z = A_sparse_reconstructed[j*(size_t)n + i];
            printf("(%5.2f,%5.2f) ", crealf(z), cimagf(z));
        }
        printf("\n");
    }

    float complex *A_dense_reconstructed = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));
    dense_identity_test(n, fill_in_matrix_size, fill_in_matrix, A_dense_reconstructed, fill_in_piv, lu_factorise_dense);

    if (lu_factorise_dense) {
        printf("\nMatrix reconstructed from L_d*U_d*I:\n");
    } else {
        printf("\nMatrix reconstructed from A_d*I:\n");
    }
    for (int i = 0; i < fill_in_matrix_size; ++i) {
        for (int j = 0; j < fill_in_matrix_size; ++j) {
            float complex z = A_dense_reconstructed[j*(size_t)n + i];
            printf("(%5.2f,%5.2f) ", crealf(z), cimagf(z));
        }
        printf("\n");
    }

    float complex *A_all_factors_reconstructed = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));
    sparse_dense_identity_test(n, &bsf, fill_in_matrix_size, fill_in_matrix, A_all_factors_reconstructed, fill_in_piv, lu_factorise_dense);

    if (lu_factorise_dense) {
        printf("\nA reconstructed from L_s*L_d*U_d*U_s*I:\n");
    } else {
        printf("\nA reconstructed from L_s*A_d*U_s*I:\n");
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float complex z = A_all_factors_reconstructed[j*(size_t)n + i];
            printf("(%5.2f,%5.2f) ", crealf(z), cimagf(z));
        }
        printf("\n");
    }

    bsf_free(&bsf);
    free(A_sparse_reconstructed);
    free(A_dense_reconstructed);
    free(A_all_factors_reconstructed);
    free(fill_in_matrix);
    free(received_fill_in);
    free(fill_in_piv);
}