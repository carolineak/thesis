#ifndef BLOCK_SPARSE_FORMAT_H
#define BLOCK_SPARSE_FORMAT_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>

// Integer range
typedef struct {
    int start;
    int end;
} int_range;

// Slice of blocks in a row/col
typedef struct {
    int num_blocks;     // Number of blocks in slice
    int *indices;       // Block indices
    int_range range;    // Index range on the slice
} block_slice;

// Block sparse matrix
typedef struct {
    int m, n;                   // Global size of block matrix
    int num_blocks;             // Number of blocks

    int num_rows;               // Number of block-rows
    int num_cols;               // Number of block-cols

    block_slice *rows;          // Row slices
    block_slice *cols;          // Col slices

    int *row_indices;           // Row indices of the blocks, len=num_blocks
    int *col_indices;           // Column indices of the blocks, len=num_blocks

    float complex *flat_data;   // All matrix blocks flattened, len=sum of all matrix block sizes

    int *block_sizes;           // Size of each block, len=num_blocks
    int *offsets;               // Offsets into flattened matrix blocks (offsets(block(i)) = acc(block_sizes(i-1))), len=num_blocks
    
    int *relies_on_fillin;      // Flag array to check if block relies of fill-in, len=num_blocks
    int *global_pivot;          // Global pivot vector for all diagonal blocks, len=sum of row range of each block
} block_sparse_format;  

// ===== Block_slice helpers =====
static inline void block_slice_free(block_slice *s) {
    free(s->indices);
    s->indices = NULL;
    s->num_blocks = 0;
}

// ===== Block_sparse_format helpers =====
static inline void bsf_free(block_sparse_format *bsf) {
    // Free slices
    if (bsf->rows) {
        for (int r = 0; r < bsf->num_rows; r++)
            block_slice_free(&bsf->rows[r]);
    }
    if (bsf->cols) {
        for (int c = 0; c < bsf->num_cols; c++)
            block_slice_free(&bsf->cols[c]);
    }
    free(bsf->rows);
    free(bsf->cols);

    free(bsf->flat_data);
    free(bsf->block_sizes);
    free(bsf->offsets);
    free(bsf->row_indices);
    free(bsf->col_indices);
    free(bsf->relies_on_fillin);
    free(bsf->global_pivot);
}

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


// ==== Create block sparse matrix ====
int create(block_sparse_format *bsf,
           const int *row_indices,
           const int *col_indices, 
           const int num_blocks,
           const int *block_lengths, 
           const float complex *data);

// ==== Print block sparse matrix as dense matrix ====
void sparse_print_matrix(const block_sparse_format *bsf);

// ==== Compute matvec with bsf ====
int sparse_matvec(const block_sparse_format *bsf,
                  const float complex *vec_in,  
                  int len_in,
                  float complex *vec_out,       
                  int len_out);

// ==== Compute sparse LU factorization with fill-ins ====
int sparse_lu(block_sparse_format *bsf, complex float **fill_in_matrix_out, int *fill_in_matrix_size_out);

#endif