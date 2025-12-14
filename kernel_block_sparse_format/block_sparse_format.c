#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>
#include "block_sparse_format.h"
#include "kernels.h"

// ==================================================================
// LU factorisation with pivoting for a dense matrix block
// ==================================================================
static int block_lu(float complex *blk, int n, int *ipiv) {
    int lda = n;
    lapack_int info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, n, n, (lapack_complex_float*)blk, lda, (lapack_int*)ipiv);
    return (int)info;
}

// ==================================================================
// Triangular solve with pivoting on device
// ==================================================================
static void cuda_block_trsm(cuFloatComplex *d_A, 
                            cuFloatComplex *d_B, 
                            int B_m, 
                            int B_n, 
                            int A_m, 
                            int *ipiv, 
                            char side, 
                            char uplo, 
                            char diag) {
    // d_A: device pointer to LU-factored diagonal block (triangular m x m)
    // d_B: device pointer to block to be overwritten (m x n)

    int m = B_m; // number of rows of target block
    int n = B_n; // number of columns of target block
    int lda = A_m; // leading dim of triangular block
    int ldb = B_m; // leading dim of target block

    cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);

    // Only apply pivots if Left/Lower (L) step of LU solve
    if (side == 'L' || side == 'l') {
        if (uplo == 'L' || uplo == 'l') {
            // Download block B to host to apply pivots on host
            float complex *h_B = (float complex*)malloc(m * n * sizeof(float complex));
            if (!h_B) {
                fprintf(stderr, "Failed to allocate host memory for B\n");
                return;
            }
            cudaMemcpy(h_B, d_B, m * n * sizeof(float complex), cudaMemcpyDeviceToHost);

            // Apply pivots on host
            for (int i = 0; i < m; ++i) {
                int piv = (int)ipiv[i] - 1;   // ipiv is 1-based
                if (piv != i) {
                    // swap row i <-> piv across all N columns (column-major)
                    for (int j = 0; j < n; ++j) {
                        float complex tmp = h_B[i   + ldb*j];
                        h_B[i   + ldb*j] = h_B[piv + ldb*j];
                        h_B[piv + ldb*j] = tmp;
                    }
                }
            }

            // Upload back to device
            cudaMemcpy(d_B, h_B, m * n * sizeof(float complex), cudaMemcpyHostToDevice);
            free(h_B);
        }
    }


    // Call trisolve_cu with device pointers for A and B and cuFloatComplex alpha
    int rc = trisolve_cu(side, uplo, 
                         'N', diag, 
                         m, n, 
                         &alpha, 
                         (const cuFloatComplex*)d_A, lda, 
                         (cuFloatComplex*)d_B, ldb);
    if (rc != 0) {
        fprintf(stderr, "trisolve_cu failed: %d\n", rc);
    }
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

// ==================================================================
// Count total number of elements in flat_data
// ==================================================================
static size_t flat_num_elements(const block_sparse_format *bsf)
{
    if (!bsf || !bsf->block_sizes) return 0;

    size_t total = 0;
    for (int i = 0; i < bsf->num_blocks; ++i) {
        if (bsf->block_sizes[i] > 0)
            total += (size_t)bsf->block_sizes[i];
    }
    return total;
}

// ==================================================================
// Upload flat_data to device
// ==================================================================
int upload_flat_data(block_sparse_format *bsf)
{
    if (!bsf) return -1;
    if (!bsf->flat_data) return -1;

    size_t numel = flat_num_elements(bsf);
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

    // Allocate device memory for flat_data
    cudaError_t err = cudaMalloc((void**)&bsf->d_flat_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[upload_flat_data] cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        bsf->flat_on_device = 0;
        return -1;
    }

    // Copy data to device
    err = cudaMemcpy(bsf->d_flat_data,
                     (const void*)bsf->flat_data,
                     bytes,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[upload_flat_data] cudaMemcpy H2D failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(bsf->d_flat_data);
        bsf->d_flat_data   = NULL;
        bsf->flat_on_device = 0;
        return -1;
    }

    bsf->flat_on_device = 1;
    return 0;
}

// ==================================================================
// Download flat_data from device
// ==================================================================
int download_flat_data(block_sparse_format *bsf)
{
    if (!bsf) return -1;
    if (!bsf->flat_data) return -1;
    if (!bsf->flat_on_device || !bsf->d_flat_data) {
        // Nothing to download or not on device
        return 0;
    }

    size_t numel = flat_num_elements(bsf);
    if (numel == 0) return 0;

    size_t bytes = numel * sizeof(cuFloatComplex);

    // Copy data to host
    cudaError_t err = cudaMemcpy((void*)bsf->flat_data,
                                 (const void*)bsf->d_flat_data,
                                 bytes,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[download_flat_data] cudaMemcpy D2H failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// ==================================================================
// Create a block_sparse_format matrix
// ==================================================================
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
    if (!bsf->row_indices || !bsf->col_indices) return -1;

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
    if (!bsf->block_sizes || !bsf->offsets) return -1;
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
    if (!bsf->flat_data) return -1;
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

// ==================================================================
// Prints a block sparse matrix as a dense matrix
// ==================================================================  
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

// ==================================================================
// Compute a matrix-vector product for a block sparse matrix
// ==================================================================
int sparse_matvec(const block_sparse_format *bsf,
                  const float complex *vec_in,  
                  int len_in,
                  float complex *vec_out,       
                  int len_out)
{
    // Check sizes match
    if (len_in != bsf->n || len_out != bsf->m) {
        return -1;
    }

    // Initialise GPU libraries with persistent handles
    if (gpu_init() != 0) {
        fprintf(stderr, "[sparse_matvec] gpu_init failed\n");
        return -1;
    }

    // Upload data to the GPU
    int err = upload_flat_data((block_sparse_format*)bsf);
    if (err != 0) return err;

    // Set output to zero
    for (int i = 0; i < len_out; ++i) vec_out[i] = 0.0f + 0.0f*I;

    // Build per-block metadata (host arrays)
    int nb = bsf->num_blocks;

    int *h_row_start = (int*)malloc(nb * sizeof(int));
    int *h_M = (int*)malloc(nb * sizeof(int));
    int *h_col_start = (int*)malloc(nb * sizeof(int));
    int *h_N = (int*)malloc(nb * sizeof(int));
    int *h_offsets = (int*)malloc(nb * sizeof(int));
    if (!h_row_start || !h_M || !h_col_start || !h_N || !h_offsets) {
        free(h_row_start); free(h_M); free(h_col_start); free(h_N); free(h_offsets);
        return -1;
    }

    for (int k = 0; k < nb; ++k) {
        int row_blk = bsf->row_indices[k];
        int col_blk = bsf->col_indices[k];
        const int_range row_idx = bsf->rows[row_blk].range;
        const int_range col_idx = bsf->cols[col_blk].range;
        h_row_start[k] = row_idx.start;
        h_M[k] = range_length(row_idx);
        h_col_start[k] = col_idx.start;
        h_N[k] = range_length(col_idx);
        h_offsets[k] = bsf->offsets[k];
    }

    cuFloatComplex *d_x = NULL;
    cuFloatComplex *d_y = NULL;
    int rc = 0;

    // Allocate device memory for input and output vectors
    size_t bytes_x = (size_t)len_in * sizeof(cuFloatComplex);
    size_t bytes_y = (size_t)len_out * sizeof(cuFloatComplex);
    
    if (cudaMalloc((void**)&d_x, bytes_x) != cudaSuccess) {
        fprintf(stderr, "[sparse_matvec] cudaMalloc for d_x failed\n");
        rc = -1;
    } else if (cudaMalloc((void**)&d_y, bytes_y) != cudaSuccess) {
        fprintf(stderr, "[sparse_matvec] cudaMalloc for d_y failed\n");
        rc = -1;
    } else {
        // Copy input vector to device
        if (cudaMemcpy(d_x, (const void*)vec_in, bytes_x, cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "[sparse_matvec] cudaMemcpy H2D failed\n");
            rc = -1;
        } else {
            // Initialise d_y to zeros
            if (cudaMemset(d_y, 0, bytes_y) != cudaSuccess) {
                fprintf(stderr, "[sparse_matvec] cudaMemset failed\n");
                rc = -1;
            } else {
                // Call the CUDA block-sparse matvec wrapper
                rc = matvec_cu(bsf->d_flat_data, nb, h_row_start, h_M, h_col_start, h_N, h_offsets, d_x, d_y);
                if (rc != 0) {
                    fprintf(stderr, "[sparse_matvec] matvec_cu failed: %d\n", rc);
                } else {
                    // Copy result back to host
                    if (cudaMemcpy((void*)vec_out, d_y, bytes_y, cudaMemcpyDeviceToHost) != cudaSuccess) {
                        fprintf(stderr, "[sparse_matvec] cudaMemcpy D2H failed\n");
                        rc = -1;
                    }
                }
            }
        }
    }

    // Free device temporaries
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);

    // Free host metadata arrays
    free(h_row_start);
    free(h_M);
    free(h_col_start);
    free(h_N);
    free(h_offsets);

    // Return early on error before downloading
    if (rc != 0) return rc;

    // Download from the GPU 
    err = download_flat_data((block_sparse_format*)bsf);
    if (err != 0) return err;

    return 0;
}

// ==================================================================
// Sparse LU factorisation of block sparse matrix with fill-ins
// ==================================================================
int sparse_lu(block_sparse_format *bsf, 
              complex float **fill_in_matrix_out, 
              int *fill_in_matrix_size_out, 
              int **received_fill_in_out,
              int print) {

    // =======================================================================
    // Check inputs
    // =======================================================================
    if (!bsf || !fill_in_matrix_out || !received_fill_in_out) return -1;

    // Only square block matrices
    if (bsf->num_rows != bsf->num_cols) return -1;
    int num_blocks = bsf->num_blocks;
    
    // =======================================================================
    // Prepare for GPU offloading
    // =======================================================================
    // Upload data to the GPU
    int err = upload_flat_data(bsf);
    if (err != 0) return err;
    
    // Initialise GPU librariesw with persistent handles
    if (gpu_init() != 0) {
        fprintf(stderr, "[sparse_lu] gpu_init failed\n");
        return -1;
    }
    
    // =======================================================================
    // Allocate arrays and initialise outputs
    // =======================================================================
    *fill_in_matrix_out = NULL;  
    *fill_in_matrix_size_out = 0; 
    *received_fill_in_out = NULL;

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

    // Print received_fill_in for debugging
    // printf("Rows/cols that received fill-in:\n");
    // for (int j = 0; j < bsf->num_rows; j++) {
    //     printf("%d ", received_fill_in[j]);
    // }
    // printf("\n");

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

    // =======================================================================
    // Computation run with actual-sized fill-in matrix
    // =======================================================================
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
        if (diag_idx < 0) return -1;

        int diag_M = range_length(bsf->rows[bsf->row_indices[diag_idx]].range);
        int diag_N = range_length(bsf->cols[bsf->col_indices[diag_idx]].range); 

        int row_start = bsf->rows[i].range.start;
        int *block_pivot = &bsf->global_pivot[row_start];

        // Skip row/col if it has received fill-in
        if (received_fill_in[i]) {
            // printf("Row/col %d has received fill-in, skipping\n", i);
            continue;
        }

        // Download diagonal block to host from the device
        size_t diag_elems = (size_t)diag_M * (size_t)diag_N;
        size_t bytes_diag = diag_elems * sizeof(cuFloatComplex);
        cudaError_t err = cudaMemcpy((void*)(bsf->flat_data + bsf->offsets[diag_idx]),
                                      (const void*)(bsf->d_flat_data + bsf->offsets[diag_idx]),
                                      bytes_diag, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "[sparse_lu] cudaMemcpy D2H diag block failed: %s\n", cudaGetErrorString(err));
            return -1;
        }

        // LU factorize diagonal block on host using LAPACK, store pivots in global pivot vector
        int info = block_lu(bsf->flat_data + bsf->offsets[diag_idx], diag_N, block_pivot);
        if (info != 0) {
            fprintf(stderr, "[sparse_lu] block_lu failed info=%d\n", info);
            return -1;
        }
        // printf("LU factorized diagonal block (%d, %d) at index %d on host\n", i, i, diag_idx);

        // Copy the updated LU factors to device so subsequent device triangular solve calls
        // operate on the updated diagonal
        if (bsf->d_flat_data) {
            size_t diag_elems = (size_t)diag_M * (size_t)diag_N;
            size_t bytes_diag = diag_elems * sizeof(cuFloatComplex);
            cudaError_t err = cudaMemcpy(bsf->d_flat_data + bsf->offsets[diag_idx],
                                          (const void*)(bsf->flat_data + bsf->offsets[diag_idx]),
                                          bytes_diag, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "[sparse_lu] cudaMemcpy H2D diag block failed: %s\n", cudaGetErrorString(err));
                return -1;
            }
        }
        
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
            // Perform triangular solve on device
            cuda_block_trsm(bsf->d_flat_data + bsf->offsets[diag_idx], 
                            bsf->d_flat_data + bsf->offsets[blk_idx], 
                            M, N, diag_M, 
                            block_pivot, 
                            'R', 'U', 'N');
            gettimeofday(&end, NULL);
            double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
            time_counts[0] += elapsed;
            matmul_counts[0] += 1; // L solve count
        }
        // L_21 blocks remain on device for device-side Schur update

        // Compute U_12 = L_11^-1 * P^T * A_12
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int blk_idx = bsf->rows[i].indices[ii];
            if (bsf->col_indices[blk_idx] == i) continue;
            if (bsf->col_indices[blk_idx] < i && !received_fill_in[bsf->col_indices[blk_idx]]) continue;
            const int M = range_length(bsf->rows[bsf->row_indices[blk_idx]].range);
            const int N = range_length(bsf->cols[bsf->col_indices[blk_idx]].range);

            gettimeofday(&start, NULL);
            // Perform triangular solve on device
            cuda_block_trsm(bsf->d_flat_data + bsf->offsets[diag_idx], 
                            bsf->d_flat_data + bsf->offsets[blk_idx], 
                            M, N, diag_M, 
                            block_pivot, 
                            'L', 'L', 'U');  
            gettimeofday(&end, NULL);
            double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
            time_counts[1] += elapsed;
            matmul_counts[1] += 1; // U solve count        
        }
        // U_12 blocks remain on device for device-side Schur update

        // Schur complement update
        for (int ii = 0; ii < bsf->rows[i].num_blocks; ++ii) {
            int U_12_idx = bsf->rows[i].indices[ii];
            if (bsf->col_indices[U_12_idx] == i) continue;
            if (bsf->col_indices[U_12_idx] < i && !received_fill_in[bsf->col_indices[U_12_idx]]) continue;
            for (int jj = 0; jj < bsf->cols[i].num_blocks; ++jj) {
                int L_21_idx = bsf->cols[i].indices[jj];
                if (bsf->row_indices[L_21_idx] == i) continue;
                if (bsf->row_indices[L_21_idx] < i && !received_fill_in[bsf->row_indices[L_21_idx]]) continue;

                // Find intersecting block to update (A_22)
                int row_idx = bsf->row_indices[L_21_idx];
                int col_idx = bsf->col_indices[U_12_idx];
                int A_22_idx = -1;
                for (int k = 0; k < num_blocks; ++k) {
                    if (bsf->row_indices[k] == row_idx && bsf->col_indices[k] == col_idx) {
                        A_22_idx = k;
                        break;
                    }
                }
                if (A_22_idx < 0) { // Block not present, fill-in will be created              
                    // We perform Schur update on device and write back to the fill_in_matrix

                    // Dimensions of the fill-in block
                    const int M = range_length(bsf->rows[row_idx].range);
                    const int N = range_length(bsf->cols[col_idx].range);
                    const int K = range_length(bsf->cols[bsf->col_indices[L_21_idx]].range);
                    size_t elems = (size_t)M * (size_t)N;
                    size_t bytes = elems * sizeof(cuFloatComplex);

                    // Allocate device buffer for fill-in block and copy initial values from host fill_in_matrix
                    cuFloatComplex *d_fill_in = NULL;
                    cudaError_t err = cudaMalloc((void**)&d_fill_in, bytes);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "[sparse_lu] cudaMalloc for fill-in block failed: %s\n", cudaGetErrorString(err));
                        return -1;
                    }

                    // Prepare a temporary host buffer to copy fill-in block
                    cuFloatComplex *h_fill_in = (cuFloatComplex*)malloc(bytes);
                    if (!h_fill_in) {
                        cudaFree(d_fill_in);
                        return -1;
                    }
                    if (fill_in_too_large) {
                        // Set h_fill_in to zero if fill-in matrix could not be allocated
                        for (size_t idx = 0; idx < elems; ++idx) {
                            h_fill_in[idx].x = 0.0f;
                            h_fill_in[idx].y = 0.0f;
                        }
                    } else {
                        // Copy initial fill-in block from fill_in_matrix
                        for (int c = 0; c < N; ++c) {
                            for (int r = 0; r < M; ++r) {
                                int idx = r + c*M;
                                float realv = crealf(fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[col_idx]+ c)*fill_in_matrix_size]);
                                float imagv = cimagf(fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[col_idx]+ c)*fill_in_matrix_size]);
                                h_fill_in[idx].x = realv;
                                h_fill_in[idx].y = imagv;
                            }
                        }
                    }

                    // Copy initial fill-in block to device
                    err = cudaMemcpy(d_fill_in, h_fill_in, bytes, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "[sparse_lu] cudaMemcpy H2D fill-in C failed: %s\n", cudaGetErrorString(err));
                        free(h_fill_in);
                        cudaFree(d_fill_in);
                        return -1;
                    }

                    // Device pointers for L_21 and U_12
                    const cuFloatComplex *d_L_21 = bsf->d_flat_data + bsf->offsets[L_21_idx];
                    const cuFloatComplex *d_U_12 = bsf->d_flat_data + bsf->offsets[U_12_idx];

                    gettimeofday(&start, NULL);
                    // Perform Schur update on device: A_22 = A_22 - L_21 * U_12
                    int rc = block_schur_update_cu(d_fill_in, d_L_21, d_U_12, M, N, K);
                    if (rc != 0) {
                        fprintf(stderr, "[sparse_lu] block_schur_update_cu failed: %d\n", rc);
                        free(h_fill_in);
                        cudaFree(d_fill_in);
                        return -1;
                    }
                    gettimeofday(&end, NULL);
                    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
                    time_counts[2] += elapsed;
                    matmul_counts[2] += 1; // Schur update count

                    // Copy result back to the host and store in fill_in_matrix
                    err = cudaMemcpy(h_fill_in, d_fill_in, bytes, cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "[sparse_lu] cudaMemcpy D2H fill-in C failed: %s\n", cudaGetErrorString(err));
                        free(h_fill_in);
                        cudaFree(d_fill_in);
                        return -1;
                    }

                    if (!fill_in_too_large) {
                        // Store updated fill-in block back to fill_in_matrix
                        for (int c = 0; c < N; ++c) {
                            for (int r = 0; r < M; ++r) {
                                int idx = r + c*M;
                                fill_in_matrix[(fill_in_block_start[row_idx] + r) + (fill_in_block_start[col_idx] + c)*fill_in_matrix_size] = h_fill_in[idx].x + h_fill_in[idx].y * I;
                            }
                        }
                    }

                    free(h_fill_in);
                    cudaFree(d_fill_in);
                    continue; 
                }
                const int M = range_length(bsf->rows[bsf->row_indices[A_22_idx]].range);
                const int N = range_length(bsf->cols[bsf->col_indices[A_22_idx]].range);
                const int K = range_length(bsf->cols[bsf->col_indices[L_21_idx]].range);

                gettimeofday(&start, NULL);
                // Perform Schur update on device: A_22 = A_22 - L_21 * U_12
                cuFloatComplex *d_C = bsf->d_flat_data + bsf->offsets[A_22_idx];
                const cuFloatComplex *d_L_21 = bsf->d_flat_data + bsf->offsets[L_21_idx];
                const cuFloatComplex *d_U_12 = bsf->d_flat_data + bsf->offsets[U_12_idx];
                int rc = block_schur_update_cu(d_C, d_L_21, d_U_12, M, N, K);
                if (rc != 0) {
                    fprintf(stderr, "[sparse_lu] block_schur_update_cu failed for block %d: %d\n", A_22_idx, rc);
                    return -1;
                }
                gettimeofday(&end, NULL);
                double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
                time_counts[2] += elapsed;
                matmul_counts[2] += 1; // Schur update count
            }
        }

        // // After processing each row print block (2,2) on both host and device for debugging
        // // printf("Block (2,2) after processing row %d:\n", i);
        // int block_22_idx = -1;
        // for (int k = 0; k < num_blocks; ++k) {
        //     if (bsf->row_indices[k] == 2 && bsf->col_indices[k] == 2) {
        //         block_22_idx = k;
        //         break;
        //     }
        // }
        // if (block_22_idx >= 0) {
        //     // Download block from device
        //     size_t M_22 = range_length(bsf->rows[2].range);
        //     size_t N_22 = range_length(bsf->cols[2].range);
        //     size_t elems_22 = M_22 * N_22;
        //     size_t bytes_22 = elems_22 * sizeof(cuFloatComplex);
        //     cuFloatComplex *d_block_22 = bsf->d_flat_data + bsf->offsets[block_22_idx];
        //     cuFloatComplex *h_block_22 = (cuFloatComplex*)malloc(bytes_22);
        //     if (h_block_22) {
        //         cudaError_t err = cudaMemcpy(h_block_22, d_block_22, bytes_22, cudaMemcpyDeviceToHost);
        //         if (err == cudaSuccess) {
        //             printf("Device block (2,2):\n");
        //             for (size_t r = 0; r < M_22; ++r) {
        //                 for (size_t c = 0; c < N_22; ++c) {
        //                     size_t idx = r + c * M_22;
        //                     printf("(%5.2f,%5.2f) ", h_block_22[idx].x, h_block_22[idx].y);
        //                 }
        //                 printf("\n");
        //             }
        //         }
        //         free(h_block_22);
        //     }
        // }

        // // Print host block (2,2)
        // if (block_22_idx >= 0) {
        //     printf("Host block (2,2):\n");
        //     for (int c = 0; c < range_length(bsf->cols[2].range); ++c) {
        //         for (int r = 0; r < range_length(bsf->rows[2].range); ++r) {
        //             int idx = bsf->offsets[block_22_idx] + r + c * range_length(bsf->rows[2].range);
        //             printf("(%5.2f,%5.2f) ", crealf(bsf->flat_data[idx]), cimagf(bsf->flat_data[idx]));
        //         }
        //         printf("\n");
        //     }
        // }
    }

    // After completing device-side Schur updates, download full flat_data
    // from device so host-side fill-in moving/zeroing uses the updated values
    err = download_flat_data(bsf);
    if (err != 0) return err;

    // ========================================================================
    // Moving blocks to fill-in matrix
    // ========================================================================
    // printf("\n======= AFTER PROCESSING ALL ROWS =======\n");

    // Zero out the blocks in the original matrix that were moved to the fill-in matrix
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

    // printf("Final matrix after moving fill-ins:\n");
    // sparse_print_matrix(bsf);

    // print fill_in_matrix for debugging
    // printf("Fill-in matrix (%d x %d):\n", fill_in_matrix_size, fill_in_matrix_size);
    // for (int r = 0; r < fill_in_matrix_size; ++r) {
    //     for (int c = 0; c < fill_in_matrix_size; ++c) {
    //         printf("(%5.2f,%5.2f) ", crealf(fill_in_matrix[r + c * fill_in_matrix_size]), cimagf(fill_in_matrix[r + c * fill_in_matrix_size]));
    //     }
    //     printf("\n");
    // }

    // printf("Received fill_in final flags:\n");
    // for (int j = 0; j < bsf->num_rows; j++) {
    //     printf("%d ", received_fill_in[j]);
    // }
    // printf("\n");

    // Print block id's that are in lower triangular part
    // printf("Blocks marked as lower triangular:\n");
    // for (int k = 0; k < bsf->num_blocks; ++k) {
    //     if (bsf->is_lower[k]) {
    //         printf("Block %d at (%d, %d) is lower triangular\n", k, bsf->row_indices[k], bsf->col_indices[k]);
    //     }
    // }

    // =======================================================================
    // Set outputs and clean up
    // =======================================================================
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

    gpu_finalise();

    return 0;
}

// ==================================================================
// Compute Ax = b, where A is given in block sparse LU format
// ==================================================================
int sparse_trimul(const block_sparse_format *bsf, 
                  float complex *b, 
                  char uplo) {
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
                // Check if block is in L
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
            if (diag_idx < 0) { free(x); return -1; }

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
                // Check if block is in U
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
        return -1; 
    }
    free(x);
    return 0;
}

// ===================================================================
// Computes A * I = A for an LU-factorized block-sparse matrix A = L*U.
// ===================================================================
int sparse_identity_test(const block_sparse_format *bsf, 
                         float complex *A) {
    if (!bsf) return -1;
    if (bsf->m != bsf->n) {
        fprintf(stderr, "sparse_identity_test: Matrix is not square (m=%d, n=%d)\n", bsf->m, bsf->n);
        return -1;
    }
    const int n = bsf->n;

    // Allocate work vector and output dense matrix
    float complex *v = (float complex*)malloc((size_t)n * sizeof(float complex));
    if (!v || !A) {
        free(v); free(A);
        fprintf(stderr, "sparse_identity_test: Allocation failed\n");
        return -1;
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
            return -1;
        }

        // v := L * v
        if (sparse_trimul(bsf, v, 'L') != 0) {
            fprintf(stderr, "sparse_identity_test: sparse_trimul('L') failed at column %d\n", j);
            free(v); free(A);
            return -1;
        }

        // Store as column j of dense A (col-major storage)
        for (int i = 0; i < n; ++i) {
            A[j*(size_t)n + i] = v[i];
        }
    }

    free(v);
    return 0;
}
