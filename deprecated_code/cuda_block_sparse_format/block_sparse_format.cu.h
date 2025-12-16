#ifndef BLOCK_SPARSE_FORMAT_H
#define BLOCK_SPARSE_FORMAT_H

#include <stddef.h>
#include <stdlib.h>
#include <cuComplex.h>

// Integer range
typedef struct {
    int start;
    int end;
} int_range;


// Block sparse matrix
typedef struct {
    int n;                      // Global size of block matrix (assumed square)
    int num_blocks;             // Number of blocks

    int num_slices;             // Number of block rows/cols (assumed square)

    int *row_indices;           // Row indices of the blocks, len=num_blocks
    int *col_indices;           // Column indices of the blocks, len=num_blocks

    int_range *slice_ranges;    // Range of each block row/col, len=num_slices

    cuFloatComplex *flatData;   // All matrix blocks flattened, len=sum of all matrix block sizes

    int *block_sizes;           // Size of each block, len=num_blocks
    int *offsets;               // Offsets into flattened matrix blocks (offsets(block(i)) = acc(block_sizes(i-1))), len=num_blocks
    
    int *relies_on_fillin;      // Flag array to check if block relies of fill-in, len=num_blocks
    int *global_pivot;          // Global pivot vector for all diagonal blocks, len=sum of row range of each block
} block_sparse_format;  

#endif

// ==== Range helpers ====
static inline int_range make_range(int start, int end) {
    int_range r;
    r.start = start;
    r.end   = end;
    return r;
}

static inline int range_length(int_range r) {
    return (r.end >= r.start) ? (r.end - r.start + 1) : 0;
}


// ===== Block_sparse_format helpers =====
static inline void bsf_free(block_sparse_format *bsf) {
    free(bsf->flatData);
    free(bsf->block_sizes);
    free(bsf->offsets);
    free(bsf->row_indices);
    free(bsf->col_indices);
    free(bsf->relies_on_fillin);
    free(bsf->global_pivot);
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
           const cuFloatComplex *data) 
{
    // Find max row/col index
    int num_slices = 0;
    for (int i = 0; i < num_blocks; i++) {
        if (row_indices[i] + 1 > num_slices) num_slices = row_indices[i] + 1;
    }

    bsf->num_slices = num_slices;
    bsf->num_blocks = num_blocks;

    // Copy row/col indices
    bsf->row_indices = (int*)malloc(num_blocks * sizeof(int));
    bsf->col_indices = (int*)malloc(num_blocks * sizeof(int));
    if (!bsf->row_indices || !bsf->col_indices) return -2;

    memcpy(bsf->row_indices, row_indices, num_blocks * sizeof(int));
    memcpy(bsf->col_indices, col_indices, num_blocks * sizeof(int));

    // Convert the block lengths into slice ranges
    bsf->slice_ranges = (int_range*)malloc(num_slices * sizeof(int_range));
    if (!bsf->slice_ranges) return -3;
    int offset = 0;
    for (int i = 0; i < num_slices; i++) {
        bsf->slice_ranges[i].start = offset;
        bsf->slice_ranges[i].end   = offset + block_lengths[i] - 1;
        offset = bsf->slice_ranges[i].end + 1;
    }

    // Calculate block sizes and offsets
    bsf->block_sizes = (int*)malloc(num_blocks * sizeof(int));
    bsf->offsets     = (int*)malloc(num_blocks * sizeof(int));
    if (!bsf->block_sizes || !bsf->offsets) return -3;
    int total_size = 0;
    for (int i = 0; i < num_blocks; i++) {
        int row = row_indices[i];
        int col = col_indices[i];
        int row_size = range_length(bsf->slice_ranges[row]);
        int col_size = range_length(bsf->slice_ranges[col]);
        int block_size = row_size * col_size;
        bsf->block_sizes[i] = block_size;
        bsf->offsets[i] = total_size;
        total_size += block_size;
    }

    // Copy flattened data
    bsf->flatData = (cuFloatComplex*)malloc(total_size * sizeof(cuFloatComplex));
    if (!bsf->flatData) return -4;
    memcpy(bsf->flatData, data, total_size * sizeof(cuFloatComplex));

    // Set size of matrix
    bsf->n = 0;
    for (int i = 0; i < num_slices; i++) {
        bsf->n += range_length(bsf->slice_ranges[i]);
    }

    // Relies on fill-in and global pivot are only allocated if used in LU
    bsf->relies_on_fillin = NULL;
    bsf->global_pivot = NULL;

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
                  const cuFloatComplex *vec_in,  int len_in,
                  cuFloatComplex *vec_out,       int len_out)
{
    if (!bsf || !vec_in || !vec_out) return -2;
    if (len_in != bsf->n || len_out != bsf->n) return -1; // square by contract

    // y = 0
    for (int i = 0; i < len_out; ++i)
        vec_out[i] = make_cuFloatComplex(0.0f, 0.0f);

    // Iterate over all blocks
    for (int k = 0; k < bsf->num_blocks; ++k) {
        const int br = bsf->row_indices[k]; // block-row (slice index)
        const int bc = bsf->col_indices[k]; // block-col (slice index)

        // Row/col slice ranges (inclusive)
        const int_range r = bsf->slice_ranges[br];
        const int_range c = bsf->slice_ranges[bc];

        const int row_start = r.start;
        const int col_start = c.start;
        const int M = r.end - r.start + 1;   // rows in this block
        const int N = c.end - c.start + 1;   // cols in this block

        if (M <= 0 || N <= 0) continue; // nothing to do

        // Pointer to this block inside the flat buffer
        const cuFloatComplex *A = bsf->flatData + (size_t)bsf->offsets[k];
        const int lda = M; // column-major leading dimension

        // Optional consistency check (can be #ifdef DEBUG):
        // if (bsf->block_sizes[k] != M * N) return -3;

        // y[row_start : row_start+M] += A_block * x[col_start : col_start+N]
        for (int j = 0; j < N; ++j) {
            const cuFloatComplex xj = vec_in[col_start + j];
            const cuFloatComplex *Acol = A + (size_t)j * (size_t)lda;
            cuFloatComplex *y = vec_out + row_start;

            for (int i = 0; i < M; ++i) {
                // y[i] += Acol[i] * xj;
                y[i] = cuCaddf(y[i], cuCmulf(Acol[i], xj));
            }
        }
    }

    return 0;
}