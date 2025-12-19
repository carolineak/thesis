#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>
#include <sys/time.h>
#include "block_sparse_format.h"

// ==================================================================
// LU factorisation with pivoting for a dense matrix block
// ==================================================================
static int block_lu(float complex *blk, int n, int *ipiv) {
    int lda = n;
    lapack_int info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, n, n, (lapack_complex_float*)blk, lda, (lapack_int*)ipiv);
    return (int)info;
}

// ==================================================================
// Triangular solve with pivoting
// ==================================================================
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

// ==================================================================
// Schur complement update (C = C - A * B)
// ==================================================================
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

// ==================================================================
// Apply P to vector in-place (forward row swaps recorded in ipiv)
// ==================================================================
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

// ==================================================================
// Apply P^T to vector in-place
// ==================================================================
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

void print_block(float complex *block, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("(%6.2f,%6.2f) ", crealf(block[r + c*rows]), cimagf(block[r + c*rows]));
        }
        printf("\n");
    }
}

// ===========================================================================
// Create a block_sparse_format matrix
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
    bsf->is_lower = NULL;
    
    return 0;
}

// ===================================================================
// Prints a block sparse matrix as a dense matrix
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
            const int lda = M;                  

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
// ==================================================================
int sparse_lu(block_sparse_format *bsf, complex float **fill_in_matrix_out, int *fill_in_matrix_size_out, int **received_fill_in_out, int print) {

    if (!bsf || !fill_in_matrix_out || !received_fill_in_out) return -1;

    *fill_in_matrix_out = NULL;  
    *fill_in_matrix_size_out = 0; 
    *received_fill_in_out = NULL;

    // Only square block matrices
    if (bsf->num_rows != bsf->num_cols) return -1;
    int num_blocks = bsf->num_blocks;

    // Allocate global pivot vector 
    bsf->global_pivot = (int*)calloc(bsf->n, sizeof(int));
    if (!bsf->global_pivot) return -1;

    // Allocate is_lower 
    bsf->is_lower = (int*)calloc((int)bsf->num_blocks, sizeof(int));
    if (!bsf->is_lower) return -1;

    // Flag array for keeping track of if row/col received fill-in
    int *received_fill_in = (int*)calloc((int)bsf->num_rows, sizeof(int));
    if (!received_fill_in) return -1;

    // Array for keeping track of start of block
    int *block_start = (int*)malloc((int)bsf->num_rows * sizeof(int));
    if (!block_start) return -1;
    for (int i = 0; i < bsf->num_rows; ++i) block_start[i] = bsf->rows[i].range.start;

    // Array for keeping track of how many matmuls per operation, length = 4
    int matmul_counts[4] = {0, 0, 0, 0}; // {L solve, U solve, Schur update, Total}

    // Array for keeping track of time taken per operation, length = 4
    double time_counts[4] = {0.0, 0.0, 0.0, 0.0}; // {L solve, U solve, Schur update, Total}
    struct timeval start, end;

    // Flag if fill-in matrix is too large to allocate
    int fill_in_too_large = 0;

    // ========================================================================
    // Dry run to get size of fill-in matrix
    // ========================================================================
    for (int i = 0; i < bsf->num_rows; ++i) {
        // Find intersecting blocks when schur updating A_22
        if (received_fill_in[i]) {
            continue;
        }
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int U_12_idx = bsf->rows[i].indices[ii];
            if (bsf->col_indices[U_12_idx] == i) continue;
            if (bsf->col_indices[U_12_idx] < i && !received_fill_in[bsf->col_indices[U_12_idx]]) continue; 
            for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
                int L_21_idx = bsf->cols[i].indices[jj];
                if (bsf->row_indices[L_21_idx] == i) continue;
                if (bsf->row_indices[L_21_idx] < i && !received_fill_in[bsf->row_indices[L_21_idx]]) continue;
                int row_idx = bsf->row_indices[L_21_idx];
                int col_idx = bsf->col_indices[U_12_idx];
                int A_22_idx = -1;
                for (int k = 0; k < num_blocks; ++k) {
                    if (bsf->row_indices[k] == row_idx && bsf->col_indices[k] == col_idx) {
                        A_22_idx = k;
                        break;
                    }
                }
                if (A_22_idx < 0) { // Block not present
                    received_fill_in[row_idx] = 1;
                    received_fill_in[col_idx] = 1;
                    continue;
                }
            }
        }
    }

    // ========================================================================
    // Create fill-in matrix
    // ========================================================================
    // Number of rows/cols that received fill-in
    int num_fill_in = 0;
    for (int j = 0; j < bsf->num_rows; j++) {
        if (received_fill_in[j]) num_fill_in++;
    }

    // The block start of the first row/col that received fill-in is determined by the offset
    // The rest is determined by the block sizes of the previous rows/cols that received fill-in
    // All rows that did not receive fill-in have block start -1
    int offset = 0;
    int *fill_in_block_start = (int*)malloc((int)bsf->num_rows * sizeof(int));
    for (int j = 0; j < bsf->num_rows; j++) {
        if (received_fill_in[j]) {
            fill_in_block_start[j] = offset;
            offset += range_length(bsf->rows[j].range);
        } else {
            fill_in_block_start[j] = -1;
        }
    }
    
    // Initialise a dense matrix with the room for the fill-ins
    int fill_in_matrix_size = 0;
    for (int j = 1; j < bsf->num_rows; j++) {
        if (received_fill_in[j]) fill_in_matrix_size += range_length(bsf->rows[j].range);
    }
    complex float *fill_in_matrix = (float complex*)calloc((int)fill_in_matrix_size * (int)fill_in_matrix_size, sizeof(float complex));
    if (!fill_in_matrix) { 
        printf("Fill-in matrix allocation failed. Size of fill-in matrix: %d x %d\n", fill_in_matrix_size, fill_in_matrix_size); 
        fill_in_too_large = 1;
    }

    // ========================================================================
    // Computation run with actual-sized fill-in matrix
    // ========================================================================
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

        int diag_M = range_length(bsf->rows[bsf->row_indices[diag_idx]].range);
        int diag_N = range_length(bsf->cols[bsf->col_indices[diag_idx]].range); 

        int row_start = bsf->rows[i].range.start;
        int *block_pivot = &bsf->global_pivot[row_start];

        // Skip row/col if it has received fill-in
        if (received_fill_in[i]) {
            continue;
        }

        // LU factorize diagonal block, store pivots in global pivot vector
        int info = block_lu(bsf->flat_data + bsf->offsets[diag_idx], diag_N, block_pivot);
        if (info != 0) { return -5; }
        
        // Compute L_21 = A_21 * U_11^-1
        for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
            int blk_idx = bsf->cols[i].indices[jj];
            if (bsf->row_indices[blk_idx] == i) continue;
            if (bsf->row_indices[blk_idx] < i && !received_fill_in[bsf->row_indices[blk_idx]]) continue;
            // Set is_lower flag
            bsf->is_lower[blk_idx] = 1;
            const int M = range_length(bsf->rows[bsf->row_indices[blk_idx]].range);
            const int N = range_length(bsf->cols[bsf->col_indices[blk_idx]].range);  
            gettimeofday(&start, NULL);
            block_trsm(bsf->flat_data + bsf->offsets[blk_idx], bsf->flat_data + bsf->offsets[diag_idx], M, N, diag_M, block_pivot, CblasRight, CblasUpper, CblasNonUnit);
            gettimeofday(&end, NULL);
            double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
            time_counts[0] += elapsed;
            matmul_counts[0] += 1; // L solve
        }
        
        // Compute U_12 = L_11^-1 * P^T * A_12
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int blk_idx = bsf->rows[i].indices[ii];
            if (bsf->col_indices[blk_idx] == i) continue;
            if (bsf->col_indices[blk_idx] < i && !received_fill_in[bsf->col_indices[blk_idx]]) continue;
            const int M = range_length(bsf->rows[bsf->row_indices[blk_idx]].range);
            const int N = range_length(bsf->cols[bsf->col_indices[blk_idx]].range);  
            gettimeofday(&start, NULL);
            block_trsm(bsf->flat_data + bsf->offsets[blk_idx], bsf->flat_data + bsf->offsets[diag_idx], M, N, diag_M, block_pivot, CblasLeft, CblasLower, CblasUnit); 
            gettimeofday(&end, NULL);
            double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
            time_counts[1] += elapsed;           
            matmul_counts[1] += 1; // U solve
        }

        // Schur complement update
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int U_12_idx = bsf->rows[i].indices[ii];
            if (bsf->col_indices[U_12_idx] == i) continue;
            if (bsf->col_indices[U_12_idx] < i && !received_fill_in[bsf->col_indices[U_12_idx]]) continue;
            for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
                int L_21_idx = bsf->cols[i].indices[jj];
                if (bsf->row_indices[L_21_idx] == i) continue;
                if (bsf->row_indices[L_21_idx] < i && !received_fill_in[bsf->row_indices[L_21_idx]]) continue;
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
                if (A_22_idx < 0) { // Block not present, fill-in will be created and must be moved to dense fill-in matrix}
                    // Get row and column ranges
                    const int M = range_length(bsf->rows[row_idx].range);
                    const int N = range_length(bsf->cols[col_idx].range);

                    float complex *new_blk = (float complex*)calloc((size_t)M * (size_t)N, sizeof(float complex));
                    if (!fill_in_too_large) {
                        // Create new fill-in block
                        for (int c = 0; c < N; ++c) {
                            for (int r = 0; r < M; ++r) {
                                new_blk[r + c*M] = fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[col_idx]+ c)*fill_in_matrix_size];
                            }
                        }
                    } 

                    // Perform Schur update on the new block
                    const int K = range_length(bsf->cols[bsf->col_indices[L_21_idx]].range);
                    gettimeofday(&start, NULL);
                    block_schur_update(new_blk, bsf->flat_data + bsf->offsets[L_21_idx], bsf->flat_data + bsf->offsets[U_12_idx], M, N, K);
                    gettimeofday(&end, NULL);
                    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
                    time_counts[2] += elapsed;
                    matmul_counts[2] += 1; // Schur update

                    if (!fill_in_too_large) {
                        // Copy updated block back to fill-in matrix
                        for (int c = 0; c < N; ++c) {
                            for (int r = 0; r < M; ++r) {
                                fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[col_idx] + c)*fill_in_matrix_size] = new_blk[r + c*M];
                            }
                        }
                    }

                    // Free temporary block
                    free(new_blk);
                    continue; // Skip the rest of the loop, as there is no existing block to update
                }
                const int M = range_length(bsf->rows[bsf->row_indices[A_22_idx]].range);
                const int N = range_length(bsf->cols[bsf->col_indices[A_22_idx]].range);
                const int K = range_length(bsf->cols[bsf->col_indices[L_21_idx]].range);
                gettimeofday(&start, NULL);
                block_schur_update(bsf->flat_data + bsf->offsets[A_22_idx], bsf->flat_data + bsf->offsets[L_21_idx], bsf->flat_data + bsf->offsets[U_12_idx], M, N, K);
                gettimeofday(&end, NULL);
                double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
                time_counts[2] += elapsed;
                matmul_counts[2] += 1; // Schur update
            }
        }
    }        

    // ========================================================================
    // Moving blocks to fill-in matrix
    // ========================================================================
    for (int k = 0; k < bsf->num_blocks; ++k) {
        if (fill_in_too_large) {
            // Skip moving blocks if fill-in matrix could not be allocated
            break;
        }
        int row_idx = bsf->row_indices[k];
        int col_idx = bsf->col_indices[k];
        if (received_fill_in[row_idx] && received_fill_in[col_idx]) {
            const int M = range_length(bsf->rows[row_idx].range);
            const int N = range_length(bsf->cols[col_idx].range);  

            // Copy block to fill-in matrix
            for (int c = 0; c < N; ++c) {
                for (int r = 0; r < M; ++r) {
                    fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[col_idx] + c)*fill_in_matrix_size] = bsf->flat_data[bsf->offsets[k] + r + c*M];
                }
            }

            // Zero the block in the original matrix if not a diagonal block else make identity
            for (int c = 0; c < N; ++c) {
                for (int r = 0; r < M; ++r) {   
                    if (row_idx == col_idx) {
                        bsf->flat_data[bsf->offsets[k] + r + c*M] = (r == c) ? (1.0f + 0.0f*I) : (0.0f + 0.0f*I);
                    } else {
                        bsf->flat_data[bsf->offsets[k] + r + c*M] = 0.0f + 0.0f*I;
                    }
                }
            }
        }
    }

    if (fill_in_too_large) {
        printf("Warning: Fill-in matrix was too large to allocate. Fill-in matrix not returned.\n");
        fill_in_matrix = NULL;
        fill_in_matrix_size = 0;
    }

    matmul_counts[3] = matmul_counts[0] + matmul_counts[1] + matmul_counts[2];
    time_counts[3] = time_counts[0] + time_counts[1] + time_counts[2];
    if (print >= 1) {
        printf("Sparse LU factorization completed.\n");
        printf("Number of block matrix operations:\n");
        printf("  L solves       : %d\n", matmul_counts[0]);
        printf("  U solves       : %d\n", matmul_counts[1]);
        printf("  Schur updates  : %d\n", matmul_counts[2]);
        printf("  Total          : %d\n", matmul_counts[3]);
        printf("Time taken (seconds):\n");
        printf("  L solves       : %f\n", time_counts[0]);
        printf("  U solves       : %f\n", time_counts[1]);
        printf("  Schur updates  : %f\n", time_counts[2]);
        printf("  Total          : %f\n", time_counts[3]);
        printf("Fill-in matrix size: %d x %d\n", fill_in_matrix_size, fill_in_matrix_size);
    }

    *fill_in_matrix_out = fill_in_matrix;
    *fill_in_matrix_size_out = fill_in_matrix_size;
    *received_fill_in_out = received_fill_in;

    free(block_start);
    free(fill_in_block_start);

    return 0;
}

// ==================================================================
// Compute Ax = b, where A is given in block sparse LU format
// ==================================================================
int sparse_trimul(const block_sparse_format *bsf, float complex *b, char uplo) {
    // Only square block matrices
    if (bsf->num_rows != bsf->num_cols) return -1;
    int n = bsf->num_rows;

    // Internal vector x to save original b
    float complex *x = (float complex*)malloc(bsf->m * sizeof(float complex));
    if (!x) return -1;
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
            if (diag_idx < 0) { free(x); return -1; }

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
                // Check if in L
                if (!bsf->is_lower[blk_idx] || blk_idx == diag_idx) continue;
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

            // No pivoting for U

            // Solve U_ii * x_i = b_i
            cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                        M, 1,
                        &(float complex){1.0f+0.0f*I},
                        bsf->flat_data + bsf->offsets[diag_idx], M,
                        b + row_start, M);

            // After updating the diagonal block, update the blocks in the same row
            for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
                int blk_idx = bsf->rows[i].indices[ii];
                // check if in U
                if (bsf->is_lower[blk_idx] || blk_idx == diag_idx) continue;
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
// Compute A * I = A for an LU-factorized block-sparse matrix A = L*U.
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
