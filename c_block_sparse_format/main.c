#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <complex.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <string.h>
#include "block_sparse_format.h"
#include "test_matrix.h"
#include "dense_functions.h"
#include "check_correctness.h"

// Random float complex in [0,1] + i[0,1]
static inline float complex crand(void) {
    return ((float)rand() / RAND_MAX) + ((float)rand() / RAND_MAX) * I;
}

int main(void) {
    // Parameters
    const int b = 20;   // Block size
    const int n = b*4;   // Matrix size
    #define NUM_BLOCKS 8 // Number of blocks
    block_sparse_format bsf;
    int lu_factorise_dense = 1;

    int print = 1; 
    // 0 will print nothing
    // 1 will print only results
    // 2 will also print matrices and vectors before and after
    // 3 will also print LU decompositions

    int block_structure = 3;
    // 0: structure that creates no fill-ins
    // 1: structure that creates fill-ins
    // 2 and 3: structures with varying block sizes per row/col

    printf("\nRunning tests on matrix of size %d x %d using block structure no. %d.\n", n, n, block_structure);

    // Create a test matrix
    float complex *dense = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));
    create_test_matrix(n, b, block_structure, dense, &bsf);

    // print_test_matrix_information(n, dense, &bsf);

    if (print >= 2) {
        printf("Matrix before factorisation:\n");
        sparse_print_matrix(&bsf);
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
    dense_matvec(n, n, dense, x, y_dense, 1.0f + 0.0f*I, 0.0f + 0.0f*I, CblasRowMajor);

    // Compute y_bsf = bsf * x using sparse_matvec
    if (sparse_matvec(&bsf, x, n, y_bsf, n) != 0) {
        fprintf(stderr, "sparse_matvec failed\n");
        return 1;
    } else {
        if (print >= 1) printf("\nMatrix-vector product succesfully computed.\n");
    }

    // Compute relative error
    double rel_error = relative_error(y_bsf, y_dense, n);
    if (print >= 1) printf("\nRelative error |y_dense - y_bsf| / |y_dense| = %.8f\n", rel_error);

    // Print vectors
    if (print >= 2) {
        printf("\nFirst few entries:\n");
        for (int i = 0; i < (n < 8 ? n : 8); i++) {
            printf("y_dense[%d] = (%5.2f,%5.2f)   y_bsf[%d] = (%5.2f,%5.2f)\n",
                i, crealf(y_dense[i]), cimagf(y_dense[i]),
                i, crealf(y_bsf[i]),   cimagf(y_bsf[i]));
        }
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

    // Prepare for fill-ins
    float complex *fill_in_matrix = (float complex*)calloc((size_t)n * (size_t)n, sizeof(float complex));
    if (!fill_in_matrix) { fprintf(stderr, "Alloc fill_in_matrix failed\n"); return 1; }

    int *fill_in_piv = (int*)malloc((size_t)n * sizeof(int));
    if (!fill_in_piv) { fprintf(stderr, "Alloc fill_in_piv failed\n"); return 1; }

    // Now factor the block sparse matrix
    if (block_structure == 0) {
        int bsf_lu_status = sparse_lu(&bsf);
        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu failed: %d\n", bsf_lu_status);
        } else {
            if (print >= 1) printf("\nBlock sparse LU factorisation successful.\n");
        }
    } else if (block_structure >= 1) {
        int bsf_lu_status = sparse_lu_with_fill_ins(&bsf, fill_in_matrix);
        if (bsf_lu_status != 0) {
            fprintf(stderr, "sparse_lu_with_fill_ins failed: %d\n", bsf_lu_status);
        } else {
            // LU factorise the fill-in matrix
            if (lu_factorise_dense){
                dense_lu(fill_in_matrix, n, fill_in_piv);
            }
            if (print >= 1) printf("\nBlock sparse LU factorisation successful.\n");
        }
    } else {
        fprintf(stderr, "Invalid block_structure %d\n", block_structure);
        free(dense);
        return 1;
    }
    
    if (print >= 3) sparse_print_lu(&bsf);   

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
    } else if (block_structure >= 1) {
        // ======================================================================
        // Identity testing
        if (print >= 2) {
            printf("\nThe sparse matrix A_s (factorised):\n");
            sparse_print_matrix(&bsf);

            if (lu_factorise_dense) {
                printf("\nThe dense matrix A_d (factorised):\n");
            } else {
                printf("\nThe dense matrix A_d (unfactorised):\n");
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++)
                printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[j*n + i]), cimagf(fill_in_matrix[j*n + i]));
                printf("\n");
            }

            float complex *A_dense_reconstructed = (float complex*)malloc((size_t)n * (size_t)n * sizeof(float complex));
            dense_identity_test(n, fill_in_matrix, A_dense_reconstructed, fill_in_piv, lu_factorise_dense);

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
            sparse_dense_identity_test(n, &bsf, fill_in_matrix, A_all_factors_reconstructed, fill_in_piv, lu_factorise_dense);

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
        }
        // ======================================================================

        // Compute b2 = P^TL_sP^TL_dU_dU_sx
        float complex *vec_out = (float complex*)calloc((size_t)n, sizeof(float complex));
        sparse_dense_trimul(n, &bsf, fill_in_matrix, b2, vec_out, fill_in_piv, lu_factorise_dense);

        // Store in b2
        for (int i = 0; i < n; ++i) {
            b2[i] = vec_out[i];
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
        for (int i = 0; i < (n < 12 ? n : 12); i++) {
            printf("b1[%d] = (%5.2f,%5.2f)   b2[%d] = (%5.2f,%5.2f)\n",
                i, crealf(b1[i]), cimagf(b1[i]),
                i, crealf(b2[i]), cimagf(b2[i]));
        }
    }

    // ===================================================================
    // Test sparse_trisolve by solving LUx = b

    sparse_trisolve(&bsf, b2, 'L');
    sparse_trisolve(&bsf, b2, 'U');
    if (print >= 2) {
        printf("\nFirst few entries of solution x:\n");
        for (int i = 0; i < (n < 12 ? n : 12); i++) {
            printf("x[%d] = (%5.2f,%5.2f)\n",
                i, crealf(b2[i]), cimagf(b2[i]));
        }
    }

    // Cleanup
    // ===================================================================
    bsf_free(&bsf);
    free(dense);
    free(x);
    free(y_dense);
    free(y_bsf);
    free(b1);
    free(b2);
    return 0;
}

