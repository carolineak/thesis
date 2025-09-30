#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>
#include "block_sparse_format.h"

// ===========================================================================
// Helper functions
// ===========================================================================

// LU factorisation with pivoting for a matrix_block
static int matrix_block_lu(matrix_block *blk, lapack_int *ipiv) {
    int n = (int)blk->rows;
    int lda = n;
    lapack_int info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, n, n, (lapack_complex_float*)blk->data, lda, ipiv);
    return (int)info;
}

// Triangular solve with pivoting
static void matrix_block_trsm(const matrix_block *A, const matrix_block *B, lapack_int *ipiv, int side, int uplo, int trans, int diag) {
    // A: block to be overwritten (solution)
    // B: LU factorized block (diagonal)
    // ipiv: pivot array from LU (optional, can be NULL)

    int m = (int)A->rows;
    int n = (int)A->cols;
    int lda = (int)B->rows;
    int ldb = (int)A->rows;

    // Apply pivots ONLY for the Left/Lower (L) step of an LU solve:
    // This corresponds to B <- P * B before solving L * Y = B.
    if (ipiv && side == CblasLeft && uplo == CblasLower) {
        for (int i = 0; i < m; ++i) {
            int piv = (int)ipiv[i] - 1;   // ipiv is 1-based
            if (piv != i) {
                // swap row i <-> piv across all N columns (column-major)
                for (int j = 0; j < n; ++j) {
                    float complex tmp = A->data[i   + ldb*j];
                    A->data[i   + ldb*j] = A->data[piv + ldb*j];
                    A->data[piv + ldb*j] = tmp;
                }
            }
        }
    }

    // Triangular solve
    cblas_ctrsm(CblasColMajor, side, uplo, trans, diag,
                m, n, &(float complex){1.0f+0.0f*I},
                B->data, lda,
                A->data, ldb);
    
    // For now, we do not reverse row swaps after solve, idk if we need that
}

// Schur complement update (C = C - A * B)
static void matrix_block_schur_update(matrix_block *C, const matrix_block *A, const matrix_block *B) {
    int m = (int)C->rows;
    int n = (int)C->cols;
    int k = (int)A->cols; // A: m x k, B: k x n
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                &(float complex){-1.0f+0.0f*I},
                A->data, m,
                B->data, k,
                &(float complex){1.0f+0.0f*I},
                C->data, m);
}

// Apply P to vector in-place (forward row swaps recorded in ipiv)
void apply_pivot_to_vector(float complex *vec, int n, const lapack_int *ipiv)
{
    if (!vec || !ipiv || n <= 0) return;

    for (int i = 0; i < n; ++i) {
        int j = (int)ipiv[i] - 1;   // LAPACK ipiv is 1-based
        if (j >= 0 && j < n && j != i) {
            float complex t = vec[i];
            vec[i] = vec[j];
            vec[j] = t;
        }
    }
}


// Apply P^T to vector in-place
void apply_inverse_pivot_to_vector(float complex *vec, int n, const lapack_int *ipiv)
{
    if (!vec || !ipiv || n <= 0) return;

    for (int i = n - 1; i >= 0; --i) {
        int j = (int)ipiv[i] - 1;  // LAPACK ipiv is 1-based
        if (j >= 0 && j < n && j != i) {
            float complex t = vec[i];
            vec[i] = vec[j];
            vec[j] = t;
        }
    }
}

// ===========================================================================
// Create a block_sparse_format matrix
//
// Arguments
//   bsf        : block_sparse_format (output)
//   rows       : array of row indices
//   cols       : array of col indices
//   values     : array of matrix_block
//   num_blocks : number of blocks
//
// Returns 0 on success, <0 on allocation failure
// ==========================================================================
int create(block_sparse_format *bsf,
           const int *rows,
           const int *cols,
           const matrix_block *values,
           int num_blocks)
{
    int i;
    int num_rows = 0;
    int num_cols = 0;
    int offset;

    // Find max row/col index
    for (i = 0; i < num_blocks; i++) {
        if (rows[i] + 1 > num_rows) num_rows = rows[i] + 1;
        if (cols[i] + 1 > num_cols) num_cols = cols[i] + 1;
    }

    bsf->num_rows   = num_rows;
    bsf->num_cols   = num_cols;
    bsf->num_blocks = num_blocks;

    // Size query
    // ===================================================================
    // Count how many blocks each row and column contains
    bsf->rows = (block_slice*)calloc(num_rows, sizeof(block_slice));
    bsf->cols = (block_slice*)calloc(num_cols, sizeof(block_slice));
    if (!bsf->rows || !bsf->cols) return -1;

    for (i = 0; i < num_blocks; i++) {
        bsf->rows[rows[i]].num_blocks++;
        bsf->cols[cols[i]].num_blocks++;
    }

    // Allocate space for block maps
    for (i = 0; i < num_rows; i++) {
        if (bsf->rows[i].num_blocks > 0) {
            bsf->rows[i].indices = (int*)malloc(bsf->rows[i].num_blocks * sizeof(int));
            bsf->rows[i].num_blocks = 0;
        }
    }
    for (i = 0; i < num_cols; i++) {
        if (bsf->cols[i].num_blocks > 0) {
            bsf->cols[i].indices = (int*)malloc(bsf->cols[i].num_blocks * sizeof(int));
            bsf->cols[i].num_blocks = 0;
        }
    }

    // Save original indices of the blocks
    // ===================================================================
    bsf->row_indices = (int*)malloc(num_blocks * sizeof(int));
    bsf->col_indices = (int*)malloc(num_blocks * sizeof(int));
    if (!bsf->row_indices || !bsf->col_indices) return -2;

    memcpy(bsf->row_indices, rows, num_blocks * sizeof(int));
    memcpy(bsf->col_indices, cols, num_blocks * sizeof(int));

    // Fill arrays
    // ===================================================================
    for (i = 0; i < num_blocks; i++) {
        int rpos = bsf->rows[rows[i]].num_blocks++;
        int cpos = bsf->cols[cols[i]].num_blocks++;
        bsf->rows[rows[i]].indices[rpos] = i;
        bsf->cols[cols[i]].indices[cpos] = i;
    }

    // Copy blocks
    // ===================================================================
    bsf->blocks = (matrix_block*)calloc(num_blocks, sizeof(matrix_block));
    if (!bsf->blocks) return -3;

    for (i = 0; i < num_blocks; i++) {
        matrix_block_init(&bsf->blocks[i], values[i].rows, values[i].cols);
        memcpy(bsf->blocks[i].data, values[i].data,
               values[i].rows * values[i].cols * sizeof(float complex));
    }

    // Compute size of matrix (m,n)
    // ===================================================================
    // Get info of which indices each block row/column corresponds to
    for (i = 0; i < num_blocks; i++) {
        bsf->rows[rows[i]].range = make_range(0, bsf->blocks[i].rows - 1);
        bsf->cols[cols[i]].range = make_range(0, bsf->blocks[i].cols - 1);
    }

    // Accumulate row offsets
    offset = 0;
    for (i = 0; i < num_rows; i++) {
        bsf->rows[i].range.start += offset;
        bsf->rows[i].range.end   += offset;
        offset = bsf->rows[i].range.end + 1;
    }

    // Accumulate col offsets
    offset = 0;
    for (i = 0; i < num_cols; i++) {
        bsf->cols[i].range.start += offset;
        bsf->cols[i].range.end   += offset;
        offset = bsf->cols[i].range.end + 1;
    }

    // Set size of matrix
    // ===================================================================
    bsf->m = 0;
    bsf->n = 0;
    for (i = 0; i < num_rows; i++) {
        bsf->m += range_length(bsf->rows[i].range);
    }
    for (i = 0; i < num_cols; i++) {
        bsf->n += range_length(bsf->cols[i].range);
    }

    return 0;

    // Set relies_on_fillin flag for each block to 0 (no fill-in reliance)
    for (i = 0; i < num_blocks; i++) {
        bsf->blocks[i].relies_on_fillin = 0;    
    }
}

// ===================================================================
// Prints a block sparse matrix as a dense matrix
// Fill in empty blocks with zeros
// 
// Arguments
//   bsf : Block-sparse matrix
// ===================================================================  
void sparse_print_matrix(const block_sparse_format *bsf) {
    if (!bsf) {
        printf("Block sparse format is NULL\n");
        return;
    }

    // Allocate dense matrix
    float complex *dense = (float complex*)calloc((size_t)(bsf->m * bsf->n), sizeof(float complex));
    if (!dense) {
        printf("Failed to allocate dense matrix\n");
        return;
    }

    // Fill in blocks
    for (int k = 0; k < bsf->num_blocks; ++k) {
        int row_blk = bsf->row_indices[k];
        int col_blk = bsf->col_indices[k];
        const block_slice *row_slice = &bsf->rows[row_blk];
        const block_slice *col_slice = &bsf->cols[col_blk];
        const matrix_block *blk = &bsf->blocks[k];

        int row_start = row_slice->range.start;
        int col_start = col_slice->range.start;

        for (size_t r = 0; r < blk->rows; ++r) {
            for (size_t c = 0; c < blk->cols; ++c) {
                dense[(row_start + r) + (col_start + c) * bsf->m] = blk->data[r + c * blk->rows];
            }
        }
    }

    // Print dense matrix
    // printf("Block sparse matrix as dense (%d x %d):\n", bsf->m, bsf->n);
    for (int r = 0; r < bsf->m; ++r) {
        for (int c = 0; c < bsf->n; ++c) {
            printf("(%5.2f,%5.2f) ", crealf(dense[r + c * bsf->m]), cimagf(dense[r + c * bsf->m]));
        }
        printf("\n");
    }

    free(dense);
}

// ===================================================================
// Print L and U from a LU factorised block sparse matrix
// ===================================================================
void sparse_print_lu(const block_sparse_format *bsf) {
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

// ===================================================================
// Compute a matrix-vector product for a block sparse matrix
//
// Arguments
//   bsf      : Block-sparse matrix (column-major blocks)
//   vec_in   : Dense input vector (length = len_in  == bsf->n)
//   len_in   : Length of vec_in
//   vec_out  : Dense output vector (length = len_out == bsf->m)
//   len_out  : Length of vec_out
//
// Returns 0 on success, <0 on error.
// ===================================================================
int sparse_matvec(const block_sparse_format *bsf,
                  const float complex *vec_in,  int len_in,
                  float complex *vec_out,       int len_out)
{
    // Check sizes match
    if (len_in != bsf->n || len_out != bsf->m) {
        return -1;
    }

    // Set output to zero
    for (int i = 0; i < len_out; ++i) vec_out[i] = 0.0f + 0.0f*I;

    // Remember y = alpha * A * x + beta * y
    const float complex alpha = 1.0f + 0.0f*I;
    const float complex beta  = 1.0f + 0.0f*I;

    // Loop over block-rows, then over blocks in each row
    for (int j = 0; j < bsf->num_rows; ++j) {
        const block_slice *row = &bsf->rows[j];
        const int_range row_idx = row->range;                  
        const int row_start = row_idx.start;
        const int M = row_idx.end - row_idx.start + 1;           

        for (int p = 0; p < row->num_blocks; ++p) {
            int block_idx = row->indices[p];

            // Column slice is determined by the block’s column index
            int block_col = bsf->col_indices[block_idx];
            const int_range col_idx = bsf->cols[block_col].range;   
            const int col_start = col_idx.start;
            const int N = col_idx.end - col_idx.start + 1;          

            // The dense block
            const matrix_block *blk = &bsf->blocks[block_idx];
            
            // Leading dimension
            const int lda = (int)blk->rows;                     

            // vec_out[row_start : row_start+M] += A_block * vec_in[col_start : col_start+N]
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        M, N,
                        &alpha,
                        blk->data, lda,
                        vec_in + col_start, 1,
                        &beta,
                        vec_out + row_start, 1);
        }
    }

    return 0;
}


// ==================================================================
// Sparse LU factorisation of block sparse matrix
//
// Arguments
//   bsf : Block-sparse matrix (column-major blocks), modified in place to contain
//         the LU factors in its blocks.
//
// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_lu(block_sparse_format *bsf) {
    // Only square block matrices
    if (bsf->num_rows != bsf->num_cols) return -1;
    int num_blocks = bsf->num_blocks;

    for (int i = 0; i < bsf->num_rows; ++i) {
        // Find diagonal block index
        int diag_idx = -1;
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int blk = bsf->rows[i].indices[ii];
            if (bsf->row_indices[blk] == i && bsf->col_indices[blk] == i) {
                diag_idx = blk;
                break;
            }
        }
        if (diag_idx < 0) return -3;

        // Allocate pivot vector for the diagonal block if not already allocated
        matrix_block *diag_blk = &bsf->blocks[diag_idx];
        if (!diag_blk->pivot) {
            diag_blk->pivot = (int*)malloc(diag_blk->rows * sizeof(int));
            if (!diag_blk->pivot) return -4;
        }

        // LU factorize diagonal block, store pivots in block's pivot vector
        int info = matrix_block_lu(diag_blk, diag_blk->pivot);
        if (info != 0) { return -5; }

        // Compute L_21 = A_21 * U_11^-1
        for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
            int blk_idx = bsf->cols[i].indices[jj];
            if (bsf->row_indices[blk_idx] <= i || blk_idx == diag_idx) continue;
            matrix_block_trsm(&bsf->blocks[blk_idx], &bsf->blocks[diag_idx], diag_blk->pivot, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit);
        }

        // Compute U_12 = L_11^-1 * P^T * A_12
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int blk_idx = bsf->rows[i].indices[ii];
            if (bsf->col_indices[blk_idx] <= i || blk_idx == diag_idx) continue;
            matrix_block_trsm(&bsf->blocks[blk_idx], &bsf->blocks[diag_idx], diag_blk->pivot, CblasLeft, CblasLower, CblasNoTrans, CblasUnit);
        }

        // Schur complement update
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int U_12_idx = bsf->rows[i].indices[ii];
            if (bsf->col_indices[U_12_idx] <= i || U_12_idx == diag_idx) continue;
            for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
                int L_21_idx = bsf->cols[i].indices[jj];
                if (bsf->row_indices[L_21_idx] <= i || L_21_idx == diag_idx) continue;
                // Find intersecting block (A_22)
                int row_idx = bsf->row_indices[L_21_idx];
                int col_idx = bsf->col_indices[U_12_idx];
                int A_22_idx = -1;
                for (int k = 0; k < num_blocks; ++k) {
                    if (bsf->row_indices[k] == row_idx && bsf->col_indices[k] == col_idx) {
                        A_22_idx = k;
                        break;
                    }
                }
                if (A_22_idx < 0) continue; // Block not present
                matrix_block_schur_update(&bsf->blocks[A_22_idx], &bsf->blocks[L_21_idx], &bsf->blocks[U_12_idx]);
            }
        }

        // print bsf after each outer iteration for debugging
        // printf ("Matrix after processing row %d:\n", i);
        // sparse_print_matrix(bsf);
    }
    return 0;
}

// ==================================================================
// Sparse LU factorisation of block sparse matrix with fill-ins
//
// Arguments
//   bsf : Block-sparse matrix (column-major blocks), modified in place to contain
//         the LU factors in its blocks.
//
// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_lu_with_fill_ins(block_sparse_format *bsf, complex float *fill_in_matrix) {
    // Only square block matrices
    if (bsf->num_rows != bsf->num_cols) return -1;
    int num_blocks = bsf->num_blocks;

    // Initialise a dense identity matrix with size bsf->m x bsf->n
    // This matrix will in most cases be way to large, but for now it is fine
    // We will only fill the blocks that we need, the rest will be identity
    if (!fill_in_matrix) return -2;
    for (int i = 0; i < bsf->m; ++i) fill_in_matrix[i + i*bsf->m] = 1.0f + 0.0f*I;

    for (int i = 0; i < bsf->num_rows; ++i) {
        // Find diagonal block index
        int diag_idx = -1;
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int blk = bsf->rows[i].indices[ii];
            if (bsf->row_indices[blk] == i && bsf->col_indices[blk] == i) {
                diag_idx = blk;
                break;
            }
        }
        if (diag_idx < 0) return -3;

        // Allocate pivot vector for the diagonal block if not already allocated
        matrix_block *diag_blk = &bsf->blocks[diag_idx];
        if (!diag_blk->pivot) {
            diag_blk->pivot = (int*)malloc(diag_blk->rows * sizeof(int));
            if (!diag_blk->pivot) return -4;
        }

        if ((diag_blk->relies_on_fillin)) {
            const int_range row_rng = bsf->rows[i].range;
            const int_range col_rng = bsf->cols[i].range;
            const int row_start = row_rng.start;
            const int col_start = col_rng.start;
            const int M = range_length(row_rng);
            const int N = range_length(col_rng);

            // Copy block to fill-in matrix
            for (int c = 0; c < N; ++c) {
                for (int r = 0; r < M; ++r) {
                    fill_in_matrix[(row_start + r) + (col_start + c)*bsf->m] = diag_blk->data[r + c*M];
                }
            } 

            // Set diagonal matrix to the identity 
            for (size_t r = 0; r < diag_blk->rows; ++r) {
                for (size_t c = 0; c < diag_blk->cols; ++c) {
                    diag_blk->data[r + c*diag_blk->rows] = (r == c) ? (1.0f + 0.0f*I) : (0.0f + 0.0f*I);
                }
            }
        } else {
            // LU factorize diagonal block, store pivots in block's pivot vector
            int info = matrix_block_lu(diag_blk, diag_blk->pivot);
            if (info != 0) { return -5; }
            
            // Compute L_21 = A_21 * U_11^-1
            for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
                int blk_idx = bsf->cols[i].indices[jj];
                if (bsf->row_indices[blk_idx] <= i || blk_idx == diag_idx) continue;
                matrix_block_trsm(&bsf->blocks[blk_idx], &bsf->blocks[diag_idx], diag_blk->pivot, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit);
            }
            
            // Compute U_12 = L_11^-1 * P^T * A_12
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                int blk_idx = bsf->rows[i].indices[ii];
                if (bsf->col_indices[blk_idx] <= i || blk_idx == diag_idx) continue;
                matrix_block_trsm(&bsf->blocks[blk_idx], &bsf->blocks[diag_idx], diag_blk->pivot, CblasLeft, CblasLower, CblasNoTrans, CblasUnit);
            }
            
            // Schur complement update
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                int U_12_idx = bsf->rows[i].indices[ii];
                if (bsf->col_indices[U_12_idx] <= i || U_12_idx == diag_idx) continue;
                for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
                    int L_21_idx = bsf->cols[i].indices[jj];
                    if (bsf->row_indices[L_21_idx] <= i || L_21_idx == diag_idx) continue;
                    // Find intersecting block (A_22)
                    int row_idx = bsf->row_indices[L_21_idx];
                    int col_idx = bsf->col_indices[U_12_idx];
                    int A_22_idx = -1;
                    for (int k = 0; k < num_blocks; ++k) {
                        if (bsf->row_indices[k] == row_idx && bsf->col_indices[k] == col_idx) {
                            A_22_idx = k;
                            break;
                        }
                    }
                    if (A_22_idx < 0) { // Block not present, fill-in will be created and must be moved to dense fill-in matrix
                        // Get row and column ranges
                        const int_range row_rng = bsf->rows[row_idx].range;
                        const int_range col_rng = bsf->cols[col_idx].range;
                        const int row_start = row_rng.start;
                        const int col_start = col_rng.start;
                        const int M = range_length(row_rng);
                        const int N = range_length(col_rng);

                        // Create new block in fill-in matrix
                        matrix_block new_blk;
                        matrix_block_init(&new_blk, M, N);
                        for (int c = 0; c < N; ++c) {
                            for (int r = 0; r < M; ++r) {
                                new_blk.data[r + c*M] = fill_in_matrix[(row_start + r) + (col_start + c)*bsf->m];
                            }
                        }

                        // Perform Schur update on the new block
                        matrix_block_schur_update(&new_blk, &bsf->blocks[L_21_idx], &bsf->blocks[U_12_idx]);

                        // Print updated new block for debugging
                        // printf ("Updated new block at (%d, %d):\n", row_idx, col_idx);
                        // for (int r = 0; r < M; ++r) {
                        //     for (int c = 0; c < N; ++c) {
                        //         printf ("(%5.2f,%5.2f)", crealf(new_blk.data[r + c*M]), cimagf(new_blk.data[r + c*M])); 
                        //     }
                        //     printf ("\n");
                        // }

                        // Copy updated block back to fill-in matrix
                        for (int c = 0; c < N; ++c) {
                            for (int r = 0; r < M; ++r) {
                                fill_in_matrix[(row_start + r) + (col_start + c)*bsf->m] = new_blk.data[r + c*M];
                            }
                        }

                        // Free temporary block
                        matrix_block_free(&new_blk);

                        // Print fill_in_matrix for debugging
                        // printf ("Fill-in matrix after adding block at (%d, %d):\n", row_idx, col_idx);
                        // for (int r = 0; r < bsf->m; ++r) {
                        //     for (int c = 0; c < bsf->n; ++c) {
                        //         printf ("(%5.2f,%5.2f)", crealf(fill_in_matrix[r + c*bsf->m]), cimagf(fill_in_matrix[r + c*bsf->m])); 
                        //     }
                        //     printf ("\n");
                        // }
                        // printf ("\n");   

                        // Now, the blocks in the same row and column as the fill-in block will be affected
                        // These blocks will now rely on the fill-in block for their correct values.
                        // Therefore, we set the variable relies_on_fillin to true for those blocks
                        for (int k = 0; k < num_blocks; ++k) {
                            if ((bsf->row_indices[k] == row_idx && bsf->col_indices[k] > i) ||
                                (bsf->col_indices[k] == col_idx && bsf->row_indices[k] > i)) {
                                bsf->blocks[k].relies_on_fillin = 1; // Mark block as relying on fill-in
                            }
                        } 

                        // print list of blocks that rely on fill-ins 
                        // printf ("Blocks that rely on fill-ins:\n");
                        // for (int k = 0; k < num_blocks; ++k) {
                        //     if (bsf->blocks[k].relies_on_fillin) {
                        //         printf ("Block %d at (%d, %d)\n", k, bsf->row_indices[k], bsf->col_indices[k]);
                        //     }
                        // }
                        // printf ("\n");

                        // Move the blocks that rely on fill-ins to the fill-in matrix
                        for (int k = 0; k < num_blocks; ++k) {
                            if (bsf->blocks[k].relies_on_fillin) {
                                // Get row and column ranges
                                const int_range row_rng = bsf->rows[bsf->row_indices[k]].range;
                                const int_range col_rng = bsf->cols[bsf->col_indices[k]].range;
                                const int row_start = row_rng.start;
                                const int col_start = col_rng.start;
                                const int M = range_length(row_rng);
                                const int N = range_length(col_rng);    

                                // Copy block to fill-in matrix
                                for (int c = 0; c < N; ++c) {
                                    for (int r = 0; r < M; ++r) {
                                        fill_in_matrix[(row_start + r) + (col_start + c)*bsf->m] = bsf->blocks[k].data[r + c*M];
                                    }
                                }   

                                // Print fill_in_matrix for debugging
                                // printf ("Fill-in matrix after adding block at (%d, %d):\n", bsf->row_indices[k], bsf->col_indices[k]);
                                // for (int r = 0; r < bsf->m; ++r) {
                                //     for (int c = 0; c < bsf->n; ++c) {
                                //         printf ("(%5.2f,%5.2f)", crealf(fill_in_matrix[r + c*bsf->m]), cimagf(fill_in_matrix[r + c*bsf->m])); 
                                //     }
                                //     printf ("\n");
                                // }
                                // printf ("\n");   
                            }
                        }

                        continue; // Skip the rest of the loop, as there is no existing block to update
                    }
                    matrix_block_schur_update(&bsf->blocks[A_22_idx], &bsf->blocks[L_21_idx], &bsf->blocks[U_12_idx]);
                }
            }
        }

        // print bsf after each outer iteration for debugging
        // printf ("Matrix after processing row %d:\n", i);
        // sparse_print_matrix(bsf);
    }
    return 0;
}

// ==================================================================
// Compute Ax = b, where A is given in block sparse LU format
//
// Arguments
//   bsf : Block-sparse matrix in LU factorized form
//   b   : Right-hand side vector, solution is written in place
//   uplo : 'L' for lower triangular solve, 'U' for upper triangular solve
//
// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_trimul(const block_sparse_format *bsf, float complex *b, char uplo) {
    // Only square block matrices
    if (bsf->num_rows != bsf->num_cols) return -1;
    int n = bsf->num_rows;

    // Internal vector x to save original b
    float complex *x = (float complex*)malloc(bsf->m * sizeof(float complex));
    if (!x) return -2;
    memcpy(x, b, bsf->m * sizeof(float complex));

    // Forward solve Ly = b
    if (uplo == 'L') {
        for (int i = 0; i < n; ++i) {
            // Find diagonal block index
            int diag_idx = -1;
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                int blk = bsf->rows[i].indices[ii];
                if (bsf->row_indices[blk] == i && bsf->col_indices[blk] == i) {
                    diag_idx = blk;
                    break;
                }
            }
            if (diag_idx < 0) return -2;

            // Solve L_ii * x_i = b_i
            int row_start = bsf->rows[i].range.start;
            int M = range_length(bsf->rows[i].range);
            cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                M, 1,
                &(float complex){1.0f+0.0f*I},
                bsf->blocks[diag_idx].data, (int)bsf->blocks[diag_idx].rows,
                b + row_start, M);

            // Apply pivoting to b 
            if (bsf->blocks[diag_idx].pivot) {
                apply_inverse_pivot_to_vector(b + row_start, M, bsf->blocks[diag_idx].pivot);
            }

            // After applying the diagonal block, apply the blocks in the same row but only on the lower side of the diagonal
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                int blk_idx = bsf->rows[i].indices[ii];
                if (bsf->col_indices[blk_idx] > i || blk_idx == diag_idx) continue;
                int col_start = bsf->cols[bsf->col_indices[blk_idx]].range.start;
                int N = range_length(bsf->cols[bsf->col_indices[blk_idx]].range);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            M, N,
                            &(float complex){1.0f+0.0f*I},
                            bsf->blocks[blk_idx].data, (int)bsf->blocks[blk_idx].rows,
                            x + col_start, 1,
                            &(float complex){1.0f+0.0f*I},
                            b + row_start, 1);
            }
            
        }
    }
    // Backward solve Ux = y
    else if (uplo == 'U') {
        for (int i = n - 1; i >= 0; --i) {
            // Find diagonal block 
            int diag_idx = -1;
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                int blk = bsf->rows[i].indices[ii];
                if (bsf->row_indices[blk] == i && bsf->col_indices[blk] == i) {
                    diag_idx = blk;
                    break;
                }
            }
            if (diag_idx < 0) return -3;

            // Solve U_ii * x_i = y_i
            int row_start = bsf->rows[i].range.start;
            int M = range_length(bsf->rows[i].range);
            cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                        M, 1,
                        &(float complex){1.0f+0.0f*I},
                        bsf->blocks[diag_idx].data, (int)bsf->blocks[diag_idx].rows,
                        b + row_start, M);
            
            // After updating the diagonal block, update the blocks in the same row
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                int blk_idx = bsf->rows[i].indices[ii];
                if (bsf->col_indices[blk_idx] <= i || blk_idx == diag_idx) continue;
                int col_start = bsf->cols[bsf->col_indices[blk_idx]].range.start;
                int N = range_length(bsf->cols[bsf->col_indices[blk_idx]].range);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            M, N,
                            &(float complex){1.0f+0.0f*I},
                            bsf->blocks[blk_idx].data, (int)bsf->blocks[blk_idx].rows,
                            x + col_start, 1,
                            &(float complex){1.0f+0.0f*I}, 
                            b + row_start, 1);
            }
        }
    } else {
        return -4; 
    }
    return 0;
}

// ==================================================================
// Solve Ax = b, where A is block-sparse triangular in LU format
//
// Arguments
//   bsf  : Block-sparse matrix in LU-factorized form
//   b    : Right-hand side vector; solution is written in place
//   uplo : 'L' -> solve L_s * x = b   (forward solve, applies P)
//           'U' -> solve U   * x = b   (backward solve, no pivots)
//
// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_trisolve(const block_sparse_format *bsf, float complex *b, char uplo) {
    // Only square block matrices
    if (bsf->num_rows != bsf->num_cols) return -1;

    const int n = bsf->num_rows;
    const float complex one      = 1.0f + 0.0f*I;
    const float complex minus_one= -1.0f + 0.0f*I;

    if (uplo == 'L') {
        // Forward solve: L_s * x = b
        for (int i = 0; i < n; ++i) {
            const int row_start = bsf->rows[i].range.start;
            const int M = range_length(bsf->rows[i].range);

            // Subtract contributions from strictly lower blocks in row i:
            // b_i <- b_i - sum_{j<i} L_{i,j} * x_j
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                const int blk_idx = bsf->rows[i].indices[ii];
                const int j = bsf->col_indices[blk_idx];
                if (j >= i) continue; // strictly lower only

                const int col_start = bsf->cols[j].range.start;
                const int N = range_length(bsf->cols[j].range);

                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            M, N,
                            &minus_one,
                            bsf->blocks[blk_idx].data, (int)bsf->blocks[blk_idx].rows,
                            b + col_start, 1,
                            &one,
                            b + row_start, 1);
            }

            // Find the diagonal block (i,i)
            int diag_idx = -1;
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                const int blk = bsf->rows[i].indices[ii];
                if (bsf->row_indices[blk] == i && bsf->col_indices[blk] == i) {
                    diag_idx = blk; break;
                }
            }
            if (diag_idx < 0) return -2;

            // Apply pivot P to the RHS slice
            if (bsf->blocks[diag_idx].pivot) {
                apply_pivot_to_vector(b + row_start, M, bsf->blocks[diag_idx].pivot);
            }

            // Solve L_ii * x_i = b_i   (L_ii is unit-lower inside the LU-packed block)
            cblas_ctrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                        M,
                        bsf->blocks[diag_idx].data, (int)bsf->blocks[diag_idx].rows,
                        b + row_start, 1);
        }
    }
    else if (uplo == 'U') {
        // Backward solve: U * x = b
        for (int i = n - 1; i >= 0; --i) {
            const int row_start = bsf->rows[i].range.start;
            const int M = range_length(bsf->rows[i].range);

            // Subtract contributions from strictly upper blocks in row i:
            // b_i <- b_i - sum_{j>i} U_{i,j} * x_j
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                const int blk_idx = bsf->rows[i].indices[ii];
                const int j = bsf->col_indices[blk_idx];
                if (j <= i) continue; // strictly upper only

                const int col_start = bsf->cols[j].range.start;
                const int N = range_length(bsf->cols[j].range);

                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            M, N,
                            &minus_one,
                            bsf->blocks[blk_idx].data, (int)bsf->blocks[blk_idx].rows,
                            b + col_start, 1,
                            &one,
                            b + row_start, 1);
            }

            // Find the diagonal block (i,i)
            int diag_idx = -1;
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                const int blk = bsf->rows[i].indices[ii];
                if (bsf->row_indices[blk] == i && bsf->col_indices[blk] == i) {
                    diag_idx = blk; break;
                }
            }
            if (diag_idx < 0) return -3;

            // Solve U_ii * x_i = b_i   (U_ii is upper, non-unit)
            cblas_ctrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        M,
                        bsf->blocks[diag_idx].data, (int)bsf->blocks[diag_idx].rows,
                        b + row_start, 1);
        }
    }
    else {
        return -4;
    }

    return 0;
}


// ===================================================================
// sparse_identity_test
//
// Computes A * I = A for an LU-factorized block-sparse matrix A = L*U.
// For each column e_j of the identity, applies:
//   v := U * e_j   then   v := L * v
// The resulting v is column j of A. Prints the dense A.
//
// Returns 0 on success, <0 on error.
// ===================================================================
int sparse_identity_test(const block_sparse_format *bsf, float complex *A) {
    // Basic checks
    if (!bsf) return -1;
    if (bsf->m != bsf->n) {
        fprintf(stderr, "sparse_identity_test: Matrix is not square (m=%d, n=%d)\n", bsf->m, bsf->n);
        return -2;
    }
    const int n = bsf->n;

    // Allocate work vector and output dense matrix
    float complex *v = (float complex*)malloc((size_t)n * sizeof(float complex));
    if (!v || !A) {
        free(v); free(A);
        fprintf(stderr, "sparse_identity_test: Allocation failed\n");
        return -3;
    }

    // Build columns of A by applying U then L to each unit basis vector
    for (int j = 0; j < n; ++j) {
        // Set v = e_j
        for (int i = 0; i < n; ++i) v[i] = 0.0f + 0.0f*I;
        v[j] = 1.0f + 0.0f*I;

        // v := U * v
        if (sparse_trimul(bsf, v, 'U') != 0) {
            fprintf(stderr, "sparse_identity_test: sparse_trimul('U') failed at column %d\n", j);
            free(v); free(A);
            return -4;
        }

        // v := L * v
        if (sparse_trimul(bsf, v, 'L') != 0) {
            fprintf(stderr, "sparse_identity_test: sparse_trimul('L') failed at column %d\n", j);
            free(v); free(A);
            return -5;
        }

        // Store as column j of dense A (col-major storage)
        for (int i = 0; i < n; ++i) {
            A[j*(size_t)n + i] = v[i];
        }
    }

    free(v);
    return 0;
}

// ===================================================================
// sparse_dense_add
//
// Compute: sum = Dense + BSF
//
// Arguments
//   bsf   : Block-sparse matrix (m x n)
//   dense : Input dense matrix (m x n), row-major
//   sum     : Output dense matrix (m x n), row-major
//
// Both dense and sum must be preallocated arrays of size m*n.
// ===================================================================
int sparse_dense_add(const block_sparse_format *bsf,
                     const float complex *dense,
                     float complex *sum)
{
    if (!bsf || !dense || !sum) return -1;
    const int m = bsf->m;
    const int n = bsf->n;

    // Step 1: Copy dense into sum
    memcpy(sum, dense, (size_t)m * (size_t)n * sizeof(float complex));

    // Step 2: Loop over blocks and add them into sum
    for (int blk = 0; blk < bsf->num_blocks; ++blk) {
        const matrix_block *B = &bsf->blocks[blk];

        // Get global ranges
        const int row_start = bsf->rows[ bsf->row_indices[blk] ].range.start;
        const int col_start = bsf->cols[ bsf->col_indices[blk] ].range.start;

        for (size_t jj = 0; jj < B->cols; ++jj) {
            for (size_t ii = 0; ii < B->rows; ++ii) {
                // Block entry (ii,jj), column-major in B
                float complex val = B->data[ii + jj * (int)B->rows];
                int gi = row_start + ii;
                int gj = col_start + jj;
                sum[gi * n + gj] += val; // row-major
            }
        }
    }

    return 0;
}
// ===================================================================
// sparse_to_dense
//
// Turn a bsf into a dense matrix (row-major)
//
// Arguments
//   bsf   : Block-sparse matrix 
//   dense : Output dense matrix
//
// ===================================================================
int sparse_to_dense(const block_sparse_format *bsf, float complex *dense) {
    memset(dense, 0, (size_t)bsf->m * (size_t)bsf->n * sizeof(float complex));
    for (int k = 0; k < bsf->num_blocks; k++) {
        const int_range row_rng = bsf->rows[bsf->row_indices[k]].range;
        const int_range col_rng = bsf->cols[bsf->col_indices[k]].range;
        const int row_start = row_rng.start;
        const int col_start = col_rng.start;
        const matrix_block *blk = &bsf->blocks[k];
        for (size_t j = 0; j < blk->cols; j++) {
            for (size_t i = 0; i < blk->rows; i++) {
                dense[(row_start + (int)i) * bsf->n + (col_start + (int)j)] = matrix_block_get(blk, i, j);
            }
        }
    }

    return 0;
}