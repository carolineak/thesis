#ifndef BLOCK_SPARSE_FORMAT_H
#define BLOCK_SPARSE_FORMAT_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>

// Dense matrix block
typedef struct {
    size_t rows;    // Number of rows
    size_t cols;    // Number of cols
    float complex *data;    // Size = rows*cols, column-major
    // NOTE: right now the struct itself doesn’t own the storage for the matrix data — it just has a pointer
    int relies_on_fillin; // Flag indicating if the block relies on fill-in (1 = yes, 0 = no)
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

    int *global_pivot;      // Global pivot vector for all diagonal blocks
} block_sparse_format;

// Column-major index (i=row, j=col, 0-based)
#define MB_INDEX(b, i, j) ((size_t)(i) + (size_t)(b)->rows * (size_t)(j))


// ===== Matrix_block helpers =====
static inline void matrix_block_init(matrix_block *b, size_t r, size_t c) {
    b->rows = r;
    b->cols = c;
    b->data = (r && c) ? (float complex*)calloc(r * c, sizeof(float complex)) : NULL;
}
static inline void matrix_block_free(matrix_block *b) {
    free(b->data);
    b->data = NULL;
    b->rows = b->cols = 0;
}
static inline float complex matrix_block_get(const matrix_block *b, size_t i, size_t j) {
    return b->data[MB_INDEX(b, i, j)];
}
static inline void matrix_block_set(matrix_block *b, size_t i, size_t j, float complex v) {
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

// ==== Pivot helpers ====
void apply_inverse_pivot_to_vector(float complex *vec, int n, const lapack_int *ipiv);

void apply_pivot_to_vector(float complex *vec, int n, const lapack_int *ipiv);

// ==== Create block sparse matrix ====
int create(block_sparse_format *bsf,
           const int *rows,
           const int *cols,
           const matrix_block *values,
           int num_blocks);

// ==== Print block sparse matrix ====
void sparse_print_matrix(const block_sparse_format *bsf);

// ==== Print factorised block sparse matrix ====
void sparse_print_lu(const block_sparse_format *bsf);

// ==== Compute matvec with bsf ====
int sparse_matvec(const block_sparse_format *bsf,
                  const float complex *vec_in,  int len_in,
                  float complex *vec_out,       int len_out);

// ==== Compute sparse LU factorization ====
int sparse_lu(block_sparse_format *bsf);

// ==== Compute sparse LU factorization with fill-ins ====
int sparse_lu_with_fill_ins(block_sparse_format *bsf, 
                            complex float *fillin_matrix);

// ==== Compute sparse trimul ====
int sparse_trimul(const block_sparse_format *bsf,
                  float complex *b,
                  char uplo);

// ==== Solve Ax = b, where A is block-sparse triangular in LU format ====
int sparse_trisolve(const block_sparse_format *bsf, float complex *b, char uplo);

// ==== Test that LUI = A ====
int sparse_identity_test(const block_sparse_format *bsf, float complex *A);

// Adds a sparse and a dense matrix
int sparse_dense_add(const block_sparse_format *bsf,
                     const float complex *dense,
                     float complex *C);

// ==== Converts bsf to dense ====
int sparse_to_dense(const block_sparse_format *bsf, float complex *dense);
                  

#endif