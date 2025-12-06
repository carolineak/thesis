#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>
#include <sys/time.h>
#include "block_sparse_format.h"
#include "test_matrix.h"
#include "dense_functions.h"
#include "check_correctness.h"

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
    int *fill_in_piv = NULL;
    int error = 0;
    struct timeval start, end;

    if (!dense || !x || !b1 || !b2) {
        fprintf(stderr, "Allocation failed\n");
        error = 1;
    }

    if (!error) {
        create_test_matrix(n, b, block_structure, dense, &bsf);

        // Generate x filled with ones
        for (int i = 0; i < n; i++) x[i] = 1.0f + 0.0f*I;

        // Compute b1 = A*x using sparse_matvec
        if (sparse_matvec(&bsf, x, n, b1, n) != 0) {
            fprintf(stderr, "sparse_matvec failed\n");
            error = 1;
        }
    }

    // Factorize block sparse matrix
    int lu_factorise_dense = 1;
    if (!error) {
        if (block_structure == 0) {
            gettimeofday(&start, NULL);
            // int bsf_lu_status = sparse_lu(&bsf);
            int bsf_lu_status = sparse_lu_with_fill_ins(&bsf, &fill_in_matrix, &fill_in_matrix_size);
            gettimeofday(&end, NULL);

            if (bsf_lu_status != 0) {
                fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
                error = 1;
            } else if (print >= 1) {
                // printf("\nBlock sparse LU factorisation successful.\n");
            }
        } else if (block_structure >= 1) {
            gettimeofday(&start, NULL);
            int bsf_lu_status = sparse_lu_with_fill_ins(&bsf, &fill_in_matrix, &fill_in_matrix_size);
            gettimeofday(&end, NULL);

            if (bsf_lu_status != 0) {
                fprintf(stderr, "sparse_lu_with_fill_ins failed: %d\n", bsf_lu_status);
                error = 1;

            } else {
                fill_in_piv = malloc((size_t)fill_in_matrix_size * sizeof(int));
                if (!fill_in_piv) {
                    fprintf(stderr, "Alloc fill_in_piv failed\n");
                    error = 1;
                } else if (lu_factorise_dense) {
                    dense_lu(fill_in_matrix, fill_in_matrix_size, fill_in_piv);
                }
                // if (print >= 1) printf("\nBlock sparse LU factorisation successful.\n");
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
                sparse_dense_trimul(n, &bsf, fill_in_matrix_size, fill_in_matrix, b2, vec_out, fill_in_piv, lu_factorise_dense);
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
            for (int i = 0; i < (n < 20 ? n : 20); i++) {
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
    free(fill_in_piv);
}

// ==================================================================
// Run LU identity test for synthetic test matrices
// ==================================================================
void run_lu_identity_test(int n, int b, int block_structure) {
    block_sparse_format bsf;
    float complex *dense = malloc((size_t)n * n * sizeof(float complex));
    float complex *fill_in_matrix = NULL;
    int fill_in_matrix_size = 0;
    int *fill_in_piv = NULL;

    if (!dense) {
        fprintf(stderr, "Allocation failed\n");
    }

    create_test_matrix(n, b, block_structure, dense, &bsf);

    // Factorize block sparse matrix and fill-in matrix
    int lu_factorise_dense = 1;
    if (block_structure == 0) {
        int bsf_lu_status = sparse_lu(&bsf);
        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
        }
    } else if (block_structure >= 1) {
        int bsf_lu_status = sparse_lu_with_fill_ins(&bsf, &fill_in_matrix, &fill_in_matrix_size);
        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu_with_fill_ins failed: %d\n", bsf_lu_status);
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
    free(fill_in_piv);
}