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

// Make a range
static inline int_range make_range(int start, int end) {
    int_range r;
    r.start = start;
    r.end   = end;
    return r;
}

// Get length of a range
static inline int range_length(int_range r) {
    return (r.end >= r.start) ? (r.end - r.start + 1) : 0;
}

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
    // side: CblasLeft or CblasRight
    // uplo: CblasUpper or CblasLower
    // trans: CblasNoTrans, CblasTrans, CblasConjTrans
    // diag: CblasUnit or CblasNonUnit

    // Here, we assume A and B are square and of same size

    int n = (int)A->rows;

    // Apply row swaps to A according to ipiv if present and side == CblasLeft
    if (ipiv && side == CblasLeft) {
        // LAPACK ipiv uses 1-based indices
        for (int i = 0; i < n; ++i) {
            int piv = ipiv[i] - 1;
            if (piv != i) {
                // Swap row i and piv in A->data (column-major)
                for (int j = 0; j < n; ++j) {
                    float complex tmp = A->data[i + n * j];
                    A->data[i + n * j] = A->data[piv + n * j];
                    A->data[piv + n * j] = tmp;
                }
            }
        }
    }
    // Triangular solve
    cblas_ctrsm(CblasColMajor, side, uplo, trans, diag,
                n, n, &(float complex){1.0f+0.0f*I},
                B->data, n,
                A->data, n);
    
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


// Apply P^T to vector in-place
static void apply_inverse_pivot_to_vector(float complex *vec, int n, const lapack_int *ipiv)
{
    if (!vec || !ipiv || n <= 0) return;

    // Build permutation p (0-based) by applying the recorded swaps forward.
    int *p = (int*)malloc((size_t)n * sizeof *p);
    int *pinv = (int*)malloc((size_t)n * sizeof *pinv);
    float complex *tmp = (float complex*)malloc((size_t)n * sizeof *tmp);
    if (!p || !pinv || !tmp) { free(p); free(pinv); free(tmp); return; }

    for (int i = 0; i < n; ++i) p[i] = i;

    for (int i = 0; i < n; ++i) {
        int j = (int)ipiv[i] - 1; 
        if (j >= 0 && j < n && j != i) {
            int t = p[i]; p[i] = p[j]; p[j] = t;
        }
    }

    // Invert permutation: pinv[p[i]] = i
    for (int i = 0; i < n; ++i) pinv[p[i]] = i;

    // Apply P^T: new_vec[i] = old_vec[ pinv[i] ]
    for (int i = 0; i < n; ++i) tmp[i] = vec[ pinv[i] ];
    memcpy(vec, tmp, (size_t)n * sizeof *vec);

    free(p); 
    free(pinv); 
    free(tmp);
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
    }
    return 0;
}

// ==================================================================
// Solve a linear system Ax = b, where A is given in block sparse LU format
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

            // Apply inverse pivoting to b 
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
