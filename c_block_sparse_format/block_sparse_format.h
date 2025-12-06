#ifndef BLOCK_SPARSE_FORMAT_H
#define BLOCK_SPARSE_FORMAT_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>

// ==================================================================
// Block sparse matrix format
// ==================================================================
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


// ==================================================================
// Matrix block helpers
// ==================================================================
static inline void matrix_block_init(matrix_block *b, size_t r, size_t c) {
    b->rows = r;
    b->cols = c;
    b->data = (r && c) ? (float complex*)calloc(r * c, sizeof(float complex)) : NULL;
}
static inline void matrix_block_free(matrix_block *b) {
    free(b->data);
    b->data = NULL;
    b->rows = b->cols = 0;
    b->relies_on_fillin = 0;
}
static inline float complex matrix_block_get(const matrix_block *b, size_t i, size_t j) {
    return b->data[MB_INDEX(b, i, j)];
}
static inline void matrix_block_set(matrix_block *b, size_t i, size_t j, float complex v) {
    b->data[MB_INDEX(b, i, j)] = v;
}

// ==================================================================
// Block slice helpers
// ==================================================================
static inline void block_slice_free(block_slice *s) {
    free(s->indices);
    s->indices = NULL;
    s->num_blocks = 0;
}

// ==================================================================
// Block sparse format helpers
// ==================================================================
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

    // Free pivot
    free(bsf->global_pivot); 

    // Reset
    bsf->blocks = NULL;
    bsf->rows = NULL;
    bsf->cols = NULL;
    bsf->row_indices = NULL;
    bsf->col_indices = NULL;
    bsf->global_pivot = NULL;
    bsf->m = bsf->n = 0;
    bsf->num_rows = bsf->num_cols = 0;
    bsf->num_blocks = 0;
}

// ==================================================================
// Integer range helpers
// ==================================================================
static inline int_range make_range(int start, int end) {
    int_range r;
    r.start = start;
    r.end   = end;
    return r;
}

static inline int range_length(int_range r) {
    return (r.end >= r.start) ? (r.end - r.start + 1) : 0;
}


// ==================================================================
// Apply (inverse) pivot array to host vector
// ==================================================================
void apply_inverse_pivot_to_vector(float complex *vec, int n, const lapack_int *ipiv);

void apply_pivot_to_vector(float complex *vec, int n, const lapack_int *ipiv);

// ==================================================================
// Create a block_sparse_format matrix
//
// Arguments
//   bsf           : block_sparse_format (output)
//   rows          : Array of row indices for each block
//   cols          : Array of col indices for each block
//   values        : Array of matrix_block structures for each block
//   num_blocks    : Number of blocks
//
// Returns 0 on success, <0 on allocation failure
// ==================================================================
int create(block_sparse_format *bsf,
           const int *rows,
           const int *cols,
           const matrix_block *values,
           int num_blocks);

// ==================================================================
// Prints a block sparse matrix as a dense matrix
// Fill in empty blocks with zeros
// 
// Arguments
//   bsf : Block-sparse matrix
// ==================================================================  
void sparse_print_matrix(const block_sparse_format *bsf);

// ==================================================================
// Prints a factorised block sparse matrix as a dense matrix
// Fill in empty blocks with zeros
// 
// Arguments
//   bsf : Block-sparse matrix
// ==================================================================  
void sparse_print_lu(const block_sparse_format *bsf);

// ==================================================================
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
// ==================================================================
int sparse_matvec(const block_sparse_format *bsf,
                  const float complex *vec_in,  
                  int len_in,
                  float complex *vec_out,      
                  int len_out);

// ==================================================================
// Sparse LU factorisation of block sparse matrix without fill-ins
//
// Arguments
//   bsf : Block-sparse matrix (column-major blocks), modified in place to contain
//         the LU factors in its blocks.
//
// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_lu(block_sparse_format *bsf);

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
int sparse_lu_with_fill_ins(block_sparse_format *bsf, 
                            complex float **fill_in_matrix_out,
                            int *fill_in_matrix_size_out);

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
int sparse_trimul(const block_sparse_format *bsf,
                  float complex *b,
                  char uplo);

// ==================================================================
// Solve Ax = b, where A is block-sparse triangular in LU format
// 
// Arguments
//   bsf : Block-sparse matrix in LU factorized form
//   b   : Right-hand side vector, solution is written in place
//   uplo : 'L' for lower triangular solve, 'U' for upper triangular solve

// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_trisolve(const block_sparse_format *bsf, 
                    float complex *b, 
                    char uplo);

// ===================================================================
// Compute A * I = A for an LU-factorized block-sparse matrix A = L*U.
// For each column e_j of the identity, applies:
//   v := U * e_j   then   v := L * v
// The resulting v is column j of A. Prints the dense A.
//
// Arguments
//   bsf : Block-sparse matrix in LU factorized form
//   A   : Pre-allocated dense matrix to store the result (output)
//
// Returns 0 on success, <0 on error.
// ===================================================================
int sparse_identity_test(const block_sparse_format *bsf, 
                         float complex *A);

// ==================================================================
// Add a sparse and a dense matrix
// 
// Arguments
//   bsf   : Block-sparse matrix
//   dense : Dense matrix (size = bsf->m x bsf->n, column-major)
//   C     : Output dense matrix (size = bsf->m x bsf->n, column-major)
//
// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_dense_add(const block_sparse_format *bsf,
                     const float complex *dense,
                     float complex *C);

// ==================================================================
// Convert bsf to dense
//
// Arguments
//   bsf   : Block-sparse matrix
//   dense : Output dense matrix (size = bsf->m x bsf->n, column-major)

// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_to_dense(const block_sparse_format *bsf, 
                    float complex *dense);
                  

#endif