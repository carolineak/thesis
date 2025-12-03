#ifndef BLOCK_SPARSE_FORMAT_H
#define BLOCK_SPARSE_FORMAT_H

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include <lapacke.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

// ==================================================================
// Block sparse matrix format
// ==================================================================
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
    int *is_lower;              // Flag arrays to check if block is in lower triangular, len=num_blocks

    cuFloatComplex *d_flat_data;   // Device copy of flat_data
    int flat_on_device;            // 1 if d_flat_data is allocated & up to date

} block_sparse_format;  

// ==================================================================
// Helper for freeing block_slice
// ==================================================================
static inline void block_slice_free(block_slice *s) {
    free(s->indices);
    s->indices = NULL;
    s->num_blocks = 0;
}

// ==================================================================
// Helper for freeing block_sparse_format and its members
// ==================================================================
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
    free(bsf->is_lower);
    if (bsf->flat_on_device && bsf->d_flat_data) {
        cudaFree(bsf->d_flat_data);
        bsf->d_flat_data = NULL;
        bsf->flat_on_device = 0;
    }
}

// ==================================================================
// Helper functions for int_range
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
//   bsf            : block_sparse_format (output)
//   row_indices    : array of row indices
//   col_indices    : array of col indices
//   num_blocks     : number of blocks
//   block_lengths  : array of length of each block row/col
//   data           : flattened data of matrix blocks
//
// Returns 0 on success, <0 on allocation failure
// ==================================================================
int create(block_sparse_format *bsf,
           const int *row_indices,
           const int *col_indices, 
           const int num_blocks,
           const int *block_lengths, 
           const float complex *data);

// ==================================================================
// Prints a block sparse matrix as a dense matrix
// Fill in empty blocks with zeros
// 
// Arguments
//   bsf : Block-sparse matrix
// ==================================================================  
void sparse_print_matrix(const block_sparse_format *bsf);

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
// Sparse LU factorisation of block sparse matrix with fill-ins
//
// Arguments
//   bsf : Block-sparse matrix (column-major blocks), modified in place to contain
//         the LU factors in its blocks.
//   fill_in_matrix_out : Pointer to dense matrix to store fill-ins (output)
//   fill_in_matrix_size_out : Pointer to size of fill-in matrix (output)
//   received_fill_in_out : Pointer to flag array indicating which rows/cols received fill-in (output)
//
// Returns 0 on success, <0 on error.
// ==================================================================
int sparse_lu(block_sparse_format *bsf, 
              complex float **fill_in_matrix_out, 
              int *fill_in_matrix_size_out, 
              int **received_fill_in_out);

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

// ===================================================================
// Computes A * I = A for an LU-factorized block-sparse matrix A = L*U.
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

#endif