#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>
#include "block_sparse_format.h"

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
