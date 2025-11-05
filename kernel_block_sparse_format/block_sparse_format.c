#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>
#include "block_sparse_format.h"
#include "kernels.h"

// LU factorisation with pivoting for a dense matrix_block
static int block_lu(float complex *blk, int n, int *ipiv) {
    int lda = n;
    lapack_int info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, n, n, (lapack_complex_float*)blk, lda, (lapack_int*)ipiv);
    return (int)info;
}

// Triangular solve with pivoting
static void block_trsm(float complex *A, float complex *B, int A_m, int A_n, int B_m, int *ipiv, int side, int uplo, int diag) {
    // A: block to be overwritten (solution)
    // B: LU factorized block (diagonal)
    // ipiv: pivot array from LU (optional, can be NULL)

    int m = A_m;
    int n = A_n;
    int lda = B_m;
    int ldb = A_m;

    // Apply pivots ONLY for the Left/Lower (L) step of an LU solve:
    // This corresponds to B <- P * B before solving L * Y = B.
    if (ipiv && side == CblasLeft && uplo == CblasLower) {
        for (int i = 0; i < m; ++i) {
            int piv = (int)ipiv[i] - 1;   // ipiv is 1-based
            if (piv != i) {
                // swap row i <-> piv across all N columns (column-major)
                for (int j = 0; j < n; ++j) {
                    float complex tmp = A[i   + ldb*j];
                    A[i   + ldb*j] = A[piv + ldb*j];
                    A[piv + ldb*j] = tmp;
                }
            }
        }
    }

    // Triangular solve
    cblas_ctrsm(CblasColMajor, side, uplo, CblasNoTrans, diag,
                m, n, &(float complex){1.0f+0.0f*I},
                B, lda,
                A, ldb);
    
}

// Triangular solve with pivoting
static void cuda_block_trsm(float complex *A, float complex *B, int A_m, int A_n, int B_m, int *ipiv, char side, char uplo, char diag) {
    // A: block to be overwritten (solution)
    // B: LU factorized block (diagonal)
    // ipiv: pivot array from LU (optional, can be NULL)

    int m = A_m;
    int n = A_n;
    int lda = B_m;
    int ldb = A_m;
    float complex alpha = {1.0f, 0.0f};

    // Apply pivots ONLY for the Left/Lower (L) step of an LU solve:
    // This corresponds to B <- P * B before solving L * Y = B.
    if (ipiv && side == 'L' && uplo == 'L') {
        for (int i = 0; i < m; ++i) {
            int piv = (int)ipiv[i] - 1;   // ipiv is 1-based
            if (piv != i) {
                // swap row i <-> piv across all N columns (column-major)
                for (int j = 0; j < n; ++j) {
                    float complex tmp = A[i   + ldb*j];
                    A[i   + ldb*j] = A[piv + ldb*j];
                    A[piv + ldb*j] = tmp;
                }
            }
        }
    }

    // Triangular solve
    // cblas_ctrsm(CblasColMajor, side, uplo, CblasNoTrans, diag,
    //             m, n, &(float complex){1.0f+0.0f*I},
    //             B, lda,
    //             A, ldb);

    int rc = trisolve_cu(side, uplo,'N', diag, m, n, &alpha, A, lda, B, ldb);
    if (rc != 0) {
        fprintf(stderr, "trisolve_cu failed\n");
    }
    
}

// Schur complement update (C = C - A * B)
static void block_schur_update(float complex *C, float complex *A, float complex *B, int m, int n, int k) {
    // A: m x k, B: k x n, C: m x n
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                &(float complex){-1.0f+0.0f*I},
                A, m,
                B, k,
                &(float complex){1.0f+0.0f*I},
                C, m);
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
//   bsf            : block_sparse_format (output)
//   row_indices    : array of row indices
//   col_indices    : array of col indices
//   num_blocks     : number of blocks
//   block_lengths  : array of length of each block row/col
//   data           : flattened data of matrix blocks
//
// Returns 0 on success, <0 on allocation failure
// ==========================================================================
int create(block_sparse_format *bsf,
           const int *row_indices,
           const int *col_indices, 
           const int num_blocks,
           const int *block_lengths, 
           const float complex  *data) 
{
    int offset;
    
    // Find max row/col index
    int num_rows = 0;
    int num_cols = 0;
    for (int i = 0; i < num_blocks; i++) {
        if (row_indices[i] + 1 > num_rows) num_rows = row_indices[i] + 1;
        if (col_indices[i] + 1 > num_cols) num_cols = col_indices[i] + 1;
    }

    bsf->num_rows   = num_rows;
    bsf->num_cols   = num_cols;
    bsf->num_blocks = num_blocks;

    // Copy row/col indices
    bsf->row_indices = (int*)malloc(num_blocks * sizeof(int));
    bsf->col_indices = (int*)malloc(num_blocks * sizeof(int));
    if (!bsf->row_indices || !bsf->col_indices) return -2;

    memcpy(bsf->row_indices, row_indices, num_blocks * sizeof(int));
    memcpy(bsf->col_indices, col_indices, num_blocks * sizeof(int));

    // Create slices
    // Count how many blocks each row and column contains
    bsf->rows = (block_slice*)calloc(num_rows, sizeof(block_slice));
    bsf->cols = (block_slice*)calloc(num_cols, sizeof(block_slice));
    if (!bsf->rows || !bsf->cols) return -1;

    for (int i = 0; i < num_blocks; i++) {
        bsf->rows[row_indices[i]].num_blocks++;
        bsf->cols[col_indices[i]].num_blocks++;
    }

    // Allocate space for block slices
    for (int i = 0; i < num_rows; i++) {
        if (bsf->rows[i].num_blocks > 0) {
            bsf->rows[i].indices = (int*)malloc(bsf->rows[i].num_blocks * sizeof(int));
            bsf->rows[i].num_blocks = 0;
        }
    }
    for (int i = 0; i < num_cols; i++) {
        if (bsf->cols[i].num_blocks > 0) {
            bsf->cols[i].indices = (int*)malloc(bsf->cols[i].num_blocks * sizeof(int));
            bsf->cols[i].num_blocks = 0;
        }
    }

    // Fill block slices
    for (int i = 0; i < num_blocks; i++) {
        int rpos = bsf->rows[row_indices[i]].num_blocks++;
        int cpos = bsf->cols[col_indices[i]].num_blocks++;
        bsf->rows[row_indices[i]].indices[rpos] = i;
        bsf->cols[col_indices[i]].indices[cpos] = i;
    }

    // Convert the block lengths into slice ranges of the rows and cols
    offset = 0;
    for (int i = 0; i < num_rows; i++) {
        bsf->rows[i].range.start = offset;
        bsf->rows[i].range.end   = offset + block_lengths[i] - 1;
        offset = bsf->rows[i].range.end + 1;
    }
    offset = 0;
    for (int i = 0; i < num_cols; i++) {
        bsf->cols[i].range.start = offset;
        bsf->cols[i].range.end   = offset + block_lengths[i] - 1;
        offset = bsf->cols[i].range.end + 1;
    }


    // Calculate block sizes and offsets
    bsf->block_sizes = (int*)malloc(num_blocks * sizeof(int));
    bsf->offsets     = (int*)malloc(num_blocks * sizeof(int));
    if (!bsf->block_sizes || !bsf->offsets) return -3;
    int total_size = 0;
    for (int i = 0; i < num_blocks; i++) {
        int row = row_indices[i];
        int col = col_indices[i];
        int row_size = range_length(bsf->rows[row].range);
        int col_size = range_length(bsf->cols[col].range);
        int block_size = row_size * col_size;
        bsf->block_sizes[i] = block_size;
        bsf->offsets[i] = total_size;
        total_size += block_size;
    }

    // Copy flattened data
    bsf->flat_data = (float complex*)malloc(total_size * sizeof(float complex));
    if (!bsf->flat_data) return -4;
    memcpy(bsf->flat_data, data, total_size * sizeof(float complex));

    // Set size of matrix
    bsf->m = 0;
    for (int i = 0; i < num_rows; i++) {
        bsf->m += range_length(bsf->rows[i].range);
    }    
    bsf->n = 0;
    for (int i = 0; i < num_cols; i++) {
        bsf->n += range_length(bsf->cols[i].range);
    }    

    // Relies on fill-in and global pivot are only allocated if used in LU
    bsf->relies_on_fillin = NULL;
    bsf->global_pivot = NULL;

    
    return 0;
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

    // The blocks are stored in a flattened array, so we need to extract them
    for (int k = 0; k < bsf->num_blocks; ++k) {
        int row_blk = bsf->row_indices[k];
        int col_blk = bsf->col_indices[k];
        const block_slice *row_slice = &bsf->rows[row_blk];
        const block_slice *col_slice = &bsf->cols[col_blk];
        const int row_start = row_slice->range.start;
        const int col_start = col_slice->range.start;
        const int row_size = range_length(row_slice->range);
        const int col_size = range_length(col_slice->range);

        // Copy block data into dense matrix
        const float complex *block_data = &bsf->flat_data[ bsf->offsets[k] ];
        for (int r = 0; r < row_size; ++r) {
            for (int c = 0; c < col_size; ++c) {
                dense[(row_start + r) + (col_start + c) * bsf->m] = block_data[r + c * row_size];
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
            
            // Leading dimension
            const int lda = M; // ??? old (int)blk->rows;                     

            // vec_out[row_start : row_start+M] += A_block * vec_in[col_start : col_start+N]
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        M, N,
                        &alpha,
                        bsf->flat_data + bsf->offsets[block_idx], lda,
                        vec_in + col_start, 1,
                        &beta,
                        vec_out + row_start, 1);
        }
    }

    return 0;
}

// ==================================================================
// Sparse LU factorisation of block sparse matrix with fill-ins
//
// Arguments
//   bsf : Block-sparse matrix (column-major blocks), modified in place to contain
//         the LU factors in its blocks.
//   fill_in_matrix_out : Pointer to dense matrix to store fill-ins (output)
//   fill_in_matrix_size_out : Pointer to size of fill-in matrix (output)
//
// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_lu(block_sparse_format *bsf, complex float **fill_in_matrix_out, int *fill_in_matrix_size_out) {

    if (!bsf || !fill_in_matrix_out) return -1;

    // Upload date to the GPU
    int err = bsf_upload_flat_data(bsf);
    if (err != 0) return err;

    // Test kernel
    if (launch_kernel(1) != 0) {
        printf("Launch of kernel failed\n");
        return -1;
    }

    // Download from the GPU - TODO: Move to the end
    err = bsf_download_flat_data(bsf);
    if (err != 0) return err;

    *fill_in_matrix_out = NULL;  
    *fill_in_matrix_size_out = 0; 

    // Only square block matrices
    if (bsf->num_rows != bsf->num_cols) return -1;
    int num_blocks = bsf->num_blocks;

    // Allocate global pivot vector if not already allocated
    bsf->global_pivot = calloc(bsf->n, sizeof(int));
    if (!bsf->global_pivot) return -2;

    // Flag array for keeping track of if row/col received fill-in
    int *received_fill_in = (int*)malloc((int)bsf->num_rows * sizeof(int));
    for (int i = 0; i < bsf->num_rows; ++i) received_fill_in[i] = 0;

    // Array for keeping track of start of block
    int *block_start = (int*)malloc((int)bsf->num_rows * sizeof(int));
    for (int i = 0; i < bsf->num_rows; ++i) block_start[i] = bsf->rows[i].range.start;

    // ========================================================================
    // Dry run to get size of fill-in matrix
    // ========================================================================
    for (int i = 0; i < bsf->num_rows; ++i) {
        // Find intersecting blocks when schur updating A_22
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int U_12_idx = bsf->rows[i].indices[ii];
            if (bsf->col_indices[U_12_idx] <= i) continue;
            for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
                int L_21_idx = bsf->cols[i].indices[jj];
                if (bsf->row_indices[L_21_idx] <= i) continue;
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
                    // Set row and col index of fill-in to true in flag array
                    received_fill_in[row_idx] = 1;
                    received_fill_in[col_idx] = 1;

                    continue;
                }
            }
        }

        // print bsf after each outer iteration for debugging
        // printf ("Matrix after processing row %d:\n", i);
        // sparse_print_matrix(bsf);
        // for (int j = 1; j < bsf->num_rows; j++) {
        //     if (received_fill_in[j]) printf("Row/col %d received fill-in\n", j);
        //     if (block_start[j]) printf("Block in rows/col %d starts at %d\n", j, block_start[j]);
        // }
    }

    // ========================================================================
    // Create fill-in matrix
    // ========================================================================
    // Find first true index in received_fill_in and set the element of that index to be the offset
    int offset = 0;
    for (int j = 0; j < bsf->num_rows; j++) {
        if (received_fill_in[j]) {
            offset = block_start[j];
            break;
        }
    }

    // Number of rows/cols that received fill-in
    int num_fill_in = 0;
    for (int j = 0; j < bsf->num_rows; j++) {
        if (received_fill_in[j]) num_fill_in++;
    }

    int *fill_in_block_start = (int*)malloc((int)bsf->num_rows * sizeof(int));
        for (int j = 0; j < bsf->num_rows; j++) {
            fill_in_block_start[j] = 0;
            if (received_fill_in[j]) {
                fill_in_block_start[j] = block_start[j] - offset;
            }
        }
    
    // // print fill_in_block_start for debugging
    // printf("Fill-in block starts (adjusted by offset %d):\n", offset);
    // for (int j = 0; j < bsf->num_rows; j++) {
    //     printf("%d ", fill_in_block_start[j]);
    // }
    // printf("\n");
    
    // Initialise a dense matrix with the room for the fill-ins
    int fill_in_matrix_size = 0;
    for (int j = 1; j < bsf->num_rows; j++) {
        if (received_fill_in[j]) fill_in_matrix_size += range_length(bsf->rows[j].range);
    }

    complex float *fill_in_matrix = (float complex*)calloc((int)fill_in_matrix_size * (int)fill_in_matrix_size, sizeof(float complex));
    if (!fill_in_matrix) return -2;

    // // print fill_in_matrix for debugging
    // printf("Fill-in matrix (%d x %d):\n", fill_in_matrix_size, fill_in_matrix_size);
    // for (int r = 0; r < fill_in_matrix_size; ++r) {
    //     for (int c = 0; c < fill_in_matrix_size; ++c) {
    //         printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
    //     }
    //     printf("\n");
    // }

    // ========================================================================
    // Computation run with actual-sized fill-in matrix
    // ========================================================================
    for (int i = 0; i < bsf->num_rows; ++i) {

        // printf("\n============== ROW %d ============\n", i);

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

        int diag_offset = bsf->offsets[diag_idx];
        int diag_M = range_length(bsf->rows[bsf->row_indices[diag_idx]].range);
        int diag_N = range_length(bsf->cols[bsf->col_indices[diag_idx]].range); 

        int row_start = bsf->rows[i].range.start;
        int *block_pivot = &bsf->global_pivot[row_start];

        if (received_fill_in[i]) {
            const int M = range_length(bsf->rows[i].range);
            const int N = range_length(bsf->cols[i].range);

            // Copy block to fill-in matrix
            for (int c = 0; c < N; ++c) {
                for (int r = 0; r < M; ++r) {
                    fill_in_matrix[(fill_in_block_start[i] + r) + (fill_in_block_start[i] + c)*fill_in_matrix_size] = bsf->flat_data[diag_offset + r + c*M];
                }
            } 

            // // print fill_in_matrix for debugging
            // printf("Fill-in matrix after moving diagonal:\n");
            // for (int r = 0; r < fill_in_matrix_size; ++r) {
            //     for (int c = 0; c < fill_in_matrix_size; ++c) {
            //         printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
            //     }
            //     printf("\n");
            // }

            // Set diagonal matrix to the identity 
            for (int r = 0; r < range_length(bsf->rows[i].range); ++r) {
                for (int c = 0; c < range_length(bsf->cols[i].range); ++c) {
                    bsf->flat_data[diag_offset + r + c*range_length(bsf->rows[i].range)] = (r == c) ? (1.0f + 0.0f*I) : (0.0f + 0.0f*I);
                }
            }
        } else {
            // LU factorize diagonal block, store pivots in global pivot vector
            int info = block_lu(bsf->flat_data + bsf->offsets[diag_idx], diag_N, block_pivot);
            if (info != 0) { return -5; }

            // // Print matrix after diagonal factorization for debugging
            // printf("Matrix after LU factorization of diagonal block %d:\n", i);
            // sparse_print_matrix(bsf);
            
            // Compute L_21 = A_21 * U_11^-1
            for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
                int blk_idx = bsf->cols[i].indices[jj];
                if (bsf->row_indices[blk_idx] <= i || blk_idx == diag_idx) continue;
                const int M = range_length(bsf->rows[bsf->row_indices[blk_idx]].range);
                const int N = range_length(bsf->cols[bsf->col_indices[blk_idx]].range);  
                // block_trsm(bsf->flat_data + bsf->offsets[blk_idx], bsf->flat_data + bsf->offsets[diag_idx], M, N, diag_M, block_pivot, CblasRight, CblasUpper, CblasNonUnit);
                cuda_block_trsm(bsf->flat_data + bsf->offsets[blk_idx], bsf->flat_data + bsf->offsets[diag_idx], M, N, diag_M, block_pivot, 'R', 'U', 'N');
            }

            // // Print matrix after diagonal factorization for debugging
            // printf("Matrix after computing L_21 in row %d\n", i);
            // sparse_print_matrix(bsf);
            
            // Compute U_12 = L_11^-1 * P^T * A_12
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                int blk_idx = bsf->rows[i].indices[ii];
                if (bsf->col_indices[blk_idx] <= i || blk_idx == diag_idx) continue;
                const int M = range_length(bsf->rows[bsf->row_indices[blk_idx]].range);
                const int N = range_length(bsf->cols[bsf->col_indices[blk_idx]].range);  
                block_trsm(bsf->flat_data + bsf->offsets[blk_idx], bsf->flat_data + bsf->offsets[diag_idx], M, N, diag_M, block_pivot, CblasLeft, CblasLower, CblasUnit);            
                
            }

            // // Print matrix after diagonal factorization for debugging
            // printf("Matrix after computing U_12 in row %d\n", i);
            // sparse_print_matrix(bsf);

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
                        const int M = range_length(bsf->rows[row_idx].range);
                        const int N = range_length(bsf->cols[col_idx].range);

                        // Create new fill-in block
                        float complex *new_blk = (float complex*)malloc((int)M * (int)N * sizeof(float complex));
                        for (int c = 0; c < N; ++c) {
                            for (int r = 0; r < M; ++r) {
                                new_blk[r + c*M] = fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[col_idx]+ c)*fill_in_matrix_size];
                            }
                        }

                        // Perform Schur update on the new block
                        const int K = range_length(bsf->cols[bsf->col_indices[L_21_idx]].range);
                        block_schur_update(new_blk, bsf->flat_data + bsf->offsets[L_21_idx], bsf->flat_data + bsf->offsets[U_12_idx], M, N, K);

                        // Copy updated block back to fill-in matrix
                        for (int c = 0; c < N; ++c) {
                            for (int r = 0; r < M; ++r) {
                                fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[col_idx] + c)*fill_in_matrix_size] = new_blk[r + c*M];
                            }
                        }

                        // // print fill_in_matrix for debugging
                        // printf("Fill-in matrix after moving fill-in block %d x %d:\n", row_idx, col_idx);
                        // for (int r = 0; r < fill_in_matrix_size; ++r) {
                        //     for (int c = 0; c < fill_in_matrix_size; ++c) {
                        //         printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
                        //     }
                        //     printf("\n");
                        // }

                        // Free temporary block
                        free(new_blk);

                        // Now, the blocks in the same row and column as the fill-in block will be affected
                        // These blocks will now rely on the fill-in block for their correct values.
                        // Therefore, we set the variable relies_on_fillin to true for those blocks
                        for (int k = 0; k < num_blocks; ++k) {
                            if (bsf->row_indices[k] == row_idx && bsf->col_indices[k] > i) {
                                // Blocks in same row

                                //  Get row and column ranges
                                const int M = range_length(bsf->rows[bsf->row_indices[k]].range);
                                const int N = range_length(bsf->cols[bsf->col_indices[k]].range);    

                                // Copy block to fill-in matrix
                                for (int c = 0; c < N; ++c) {
                                    for (int r = 0; r < M; ++r) {
                                        fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[row_idx] + c)*fill_in_matrix_size] = bsf->flat_data[bsf->offsets[k] + r + c*M];
                                    }
                                }  

                                // print fill_in_matrix for debugging
                                // printf("Fill-in matrix after moving block %d:\n", k);
                                // for (int r = 0; r < fill_in_matrix_size; ++r) {
                                //     for (int c = 0; c < fill_in_matrix_size; ++c) {
                                //         printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
                                //     }
                                //     printf("\n");
                                // }
                            } else if (bsf->col_indices[k] == col_idx && bsf->row_indices[k] > i) {
                                // Blocks in same column

                                //  Get row and column ranges
                                const int M = range_length(bsf->rows[bsf->row_indices[k]].range);
                                const int N = range_length(bsf->cols[bsf->col_indices[k]].range);    

                                // Copy block to fill-in matrix
                                for (int c = 0; c < N; ++c) {
                                    for (int r = 0; r < M; ++r) {
                                        fill_in_matrix[(fill_in_block_start[col_idx] + r) + (fill_in_block_start[col_idx] + c)*fill_in_matrix_size] = bsf->flat_data[bsf->offsets[k] + r + c*M];
                                    }
                                } 

                                // print fill_in_matrix for debugging
                                // printf("Fill-in matrix after moving block %d:\n", k);
                                // for (int r = 0; r < fill_in_matrix_size; ++r) {
                                //     for (int c = 0; c < fill_in_matrix_size; ++c) {
                                //         printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
                                //     }
                                //     printf("\n");
                                // }
                            }
                        } 

                        continue; // Skip the rest of the loop, as there is no existing block to update
                    }
                    const int M = range_length(bsf->rows[bsf->row_indices[A_22_idx]].range);
                    const int N = range_length(bsf->cols[bsf->col_indices[A_22_idx]].range);
                    const int K = range_length(bsf->cols[bsf->col_indices[L_21_idx]].range);
                    block_schur_update(bsf->flat_data + bsf->offsets[A_22_idx], bsf->flat_data + bsf->offsets[L_21_idx], bsf->flat_data + bsf->offsets[U_12_idx], M, N, K);
                }
            }
        }

        // print bsf after each outer iteration for debugging&bsf->blocks[A_22_idx]
        // printf ("Matrix after processing row %d:\n", i);
        // sparse_print_matrix(bsf);
    }

    // print fill_in_matrix for debugging
    // printf("Fill-in matrix (%d x %d):\n", fill_in_matrix_size, fill_in_matrix_size);
    // for (int r = 0; r < fill_in_matrix_size; ++r) {
    //     for (int c = 0; c < fill_in_matrix_size; ++c) {
    //         printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
    //     }
    //     printf("\n");
    // }

    *fill_in_matrix_out = fill_in_matrix;
    *fill_in_matrix_size_out = fill_in_matrix_size;

    free(received_fill_in);
    free(block_start);
    free(fill_in_block_start);
    // free(fill_in_matrix);

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
            if (diag_idx < 0) { free(x); return -2; }

            int row_start = bsf->rows[i].range.start;
            int M = range_length(bsf->rows[i].range);

            // Use the appropriate slice of the global pivot vector
            int *block_pivot = bsf->global_pivot ? &bsf->global_pivot[row_start] : NULL;

            // Solve L_ii * x_i = b_i
            cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                M, 1,
                &(float complex){1.0f+0.0f*I},
                bsf->flat_data + bsf->offsets[diag_idx], M,
                b + row_start, M);

            // Apply pivoting to b using global pivot vector
            if (block_pivot) {
                apply_inverse_pivot_to_vector(b + row_start, M, block_pivot);
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
                            bsf->flat_data + bsf->offsets[blk_idx], M,
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
            if (diag_idx < 0) { free(x); return -3; }

            int row_start = bsf->rows[i].range.start;
            int M = range_length(bsf->rows[i].range);

            // No pivoting for U (upper)
            cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                        M, 1,
                        &(float complex){1.0f+0.0f*I},
                        bsf->flat_data + bsf->offsets[diag_idx], M,
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
                            bsf->flat_data + bsf->offsets[blk_idx], M,
                            x + col_start, 1,
                            &(float complex){1.0f+0.0f*I}, 
                            b + row_start, 1);
            }
        }
    } else {
        free(x);
        return -4; 
    }
    free(x);
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

static size_t bsf_flat_num_elements(const block_sparse_format *bsf)
{
    if (!bsf || !bsf->block_sizes) return 0;

    size_t total = 0;
    for (int i = 0; i < bsf->num_blocks; ++i) {
        if (bsf->block_sizes[i] > 0)
            total += (size_t)bsf->block_sizes[i];
    }
    return total;
}

int bsf_upload_flat_data(block_sparse_format *bsf)
{
    if (!bsf) return -1;
    if (!bsf->flat_data) return -2;

    size_t numel = bsf_flat_num_elements(bsf);
    if (numel == 0) {
        // Nothing to upload
        if (bsf->d_flat_data) {
            cudaFree(bsf->d_flat_data);
            bsf->d_flat_data = NULL;
        }
        bsf->flat_on_device = 0;
        return 0;
    }

    size_t bytes = numel * sizeof(cuFloatComplex);

    if (bsf->d_flat_data) {
        cudaFree(bsf->d_flat_data);
        bsf->d_flat_data = NULL;
    }

    cudaError_t err = cudaMalloc((void**)&bsf->d_flat_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bsf_upload_flat_data] cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        bsf->flat_on_device = 0;
        return -3;
    }

    // flat_data is float complex; cuFloatComplex has the same layout in practice.
    err = cudaMemcpy(bsf->d_flat_data,
                     (const void*)bsf->flat_data,
                     bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bsf_upload_flat_data] cudaMemcpy H2D failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(bsf->d_flat_data);
        bsf->d_flat_data   = NULL;
        bsf->flat_on_device = 0;
        return -4;
    }

    bsf->flat_on_device = 1;
    return 0;
}

int bsf_download_flat_data(block_sparse_format *bsf)
{
    if (!bsf) return -1;
    if (!bsf->flat_data) return -2;
    if (!bsf->flat_on_device || !bsf->d_flat_data) {
        // Nothing to download or not on device
        return 0;
    }

    size_t numel = bsf_flat_num_elements(bsf);
    if (numel == 0) return 0;

    size_t bytes = numel * sizeof(cuFloatComplex);

    cudaError_t err = cudaMemcpy((void*)bsf->flat_data,
                                 (const void*)bsf->d_flat_data,
                                 bytes,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[bsf_download_flat_data] cudaMemcpy D2H failed: %s\n",
                cudaGetErrorString(err));
        return -3;
    }

    return 0;
}