#ifndef BLOCK_SPARSE_FORMAT_H
#define BLOCK_SPARSE_FORMAT_H

#include <stddef.h>
#include <stdlib.h>

// Dense matrix block
typedef struct {
    size_t rows;    // Number of rows
    size_t cols;    // Number of cols
    float *data;    // Size = rows*cols, column-major
} matrix_block;

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
    int m, n;               // Global size of block matrix
    matrix_block *blocks;   // All blocks in matrix
    
    int num_blocks;         // Number of blocks

    int num_rows;           // Number of block-rows
    int num_cols;           // Number of block-cols
    
    block_slice *rows;      // Row slices
    block_slice *cols;      // Col slices

    int *row_indices;       // Row indices of the blocks
    int *col_indices;       // Column indices of the blocks
} block_sparse_format;

// Column-major index (i=row, j=col, 0-based)
#define MB_INDEX(b, i, j) ((size_t)(i) + (size_t)(b)->rows * (size_t)(j))


// ===== Matrix_block helpers =====
static inline void matrix_block_init(matrix_block *b, size_t r, size_t c) {
    b->rows = r;
    b->cols = c;
    b->data = (r && c) ? (float*)calloc(r * c, sizeof(float)) : NULL;
}
static inline void matrix_block_free(matrix_block *b) {
    free(b->data);
    b->data = NULL;
    b->rows = b->cols = 0;
}
static inline float matrix_block_get(const matrix_block *b, size_t i, size_t j) {
    return b->data[MB_INDEX(b, i, j)];
}
static inline void matrix_block_set(matrix_block *b, size_t i, size_t j, float v) {
    b->data[MB_INDEX(b, i, j)] = v;
}

// ===== Block_slice helpers =====
static inline void block_slice_free(block_slice *s) {
    free(s->indices);
    s->indices = NULL;
    s->num_blocks = 0;
}

// ===== Block_sparse_format helpers =====
static inline void bsf_free(block_sparse_format *bsf) {
    // Free each block
    if (bsf->blocks) {
        for (int k = 0; k < bsf->num_blocks; k++)
            matrix_block_free(&bsf->blocks[k]);
    }
    free(bsf->blocks);
    free(bsf->row_indices);
    free(bsf->col_indices);

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

    // Reset
    bsf->blocks = NULL;
    bsf->rows = NULL;
    bsf->cols = NULL;
    bsf->row_indices = NULL;
    bsf->col_indices = NULL;
    bsf->m = bsf->n = 0;
    bsf->num_rows = bsf->num_cols = 0;
    bsf->num_blocks = 0;
}

// ===== Create block sparse matrix =====
int create(block_sparse_format *bsf,
           const int *rows,
           const int *cols,
           const matrix_block *values,
           int num_blocks);

#endif 